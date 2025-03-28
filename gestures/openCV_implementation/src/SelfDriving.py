import asyncio
import cv2
import numpy as np
import websockets
import time
import threading
import torch
import os
import sys
from ultralytics import YOLO

print("Starting to connect")
URI = "ws://kind-control-plane:32085/roversocket"

# Load YOLOv11 Model with GPU Acceleration (If Available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("overheadbest-v5.pt").to(device)

if device == "cuda":
    model.half()  # Enable FP16 inference for speed boost

# Print Model Classes for Debugging
print(f"Model Classes: {model.names}")

# Initialize Camera
capture = cv2.VideoCapture("camera.png")
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering delay
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global Variables
frame = None
running = True
previous_msg = "S"

# Define target objects and special labels
TARGETS = ["Earth", "Saturn", "Rose"]  # 🟢 Green
SPECIAL_LABELS = ["Top", "Center", "Bottom"]  # 🟡 Yellow

def capture_frames():
    """ Continuously grabs frames, dropping old ones to prevent buffering delay. """
    global frame
    while running:
        ret, latest_frame = capture.read()
        if ret:
            frame = cv2.resize(latest_frame, (640, 480))
        time.sleep(0.001)

async def send_msg_if_not_previous(websocket, previous_msg, msg):
    """ Sends msg to the websocket so long as it is not the same as 'previous_msg'. """
    if msg != previous_msg:
        if msg != "S":
            await websocket.send("S")
            print("Sent message:", "S")
        await websocket.send(msg)
        print("Sent message:", msg)
        previous_msg = msg
    return previous_msg

async def process_and_send(websocket):
    """ Runs YOLOv11 detection and sends movement commands asynchronously. """
    global frame, previous_msg
    locked_targets = {}  # Track locked targets for 5 seconds
    saved_target_position = None  # Save target position for 2 seconds
    nearest_target_start_time = None  # Start timer when nearest target is found

    while running:
        if frame is None:
            continue

        start_time = time.time()
        latest_frame = frame.copy()
        # **Fix: Use a clean frame for detection before drawing anything**
        clean_frame = frame.copy()

        frame_height, frame_width = latest_frame.shape[:2]  # Get image dimensions

        # **Use Full-Color Image Instead of Grayscale**
        frame_input = latest_frame.copy()

        # Suppress print statements and stderr if needed
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = sys.stdout  # Optional

        # **Lower Confidence Threshold to Detect Smaller Objects**
        results = model.predict(clean_frame, verbose=False, device=device, half=True if device == "cuda" else False, conf=0.25)

        # Re-enable printing
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__  # Re-enable stderr if it was redirected


        boxes = results[0].boxes if len(results) > 0 else []
        labels = results[0].names if len(results) > 0 else []

        # **Print Detected Labels for Debugging**
        # print(f"Detected Labels: {labels}")

        network_msg = "S"  # Default to Stop
        bottom_object = None
        top_object = None
        center_object = None
        target_centers = []

        # Display detections with color-coded bounding boxes
        for i, box in enumerate(boxes):
            label = labels[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # **Color Logic**
            if label in TARGETS:
                color = (0, 255, 0)  # 🟢 Green for targets (Earth, Saturn, Rose)
                target_centers.append((box_center, label))
            elif label in SPECIAL_LABELS:
                color = (0, 255, 255)  # 🟡 Yellow for "Top", "Center", "Bottom"
                if label == "Top":
                    top_object = ((x1 + x2) // 2, (y1 + y2) // 2)
                elif label == "Bottom":
                    bottom_object = ((x1 + x2) // 2, (y1 + y2) // 2)
                elif label == "Center":
                    center_object = ((x1 + x2) // 2, (y1 + y2) // 2)
            else:
                color = (0, 0, 255)  # 🔴 Red for everything else

            # Draw Bounding Box & Label
            cv2.rectangle(latest_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(latest_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        # **Draw a line from "Bottom" → "Top" → Extend beyond "Top" correctly**
        if bottom_object and top_object:
            dx = top_object[0] - bottom_object[0]  # Direction X
            dy = top_object[1] - bottom_object[1]  # Direction Y

            # Draw the line from "Bottom" to "Top"
            cv2.line(latest_frame, bottom_object, top_object, (0, 255, 255), 2)  # 🟡 Yellow Line

            # Calculate the angle (in degrees) of the line between 'bottom' and 'top'
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)


        # Find the target closest to the "Center" object
        nearest_target = None
        min_distance = float('inf')


        # Find the target closest to the "Center" object
        nearest_target = None
        min_distance = float('inf')

        if center_object:
            for center, label in target_centers:
                distance = np.linalg.norm(np.array(center) - np.array(center_object))
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = (center, label)

            # If a nearest target is found
            if nearest_target:
                current_time = time.time()
                if nearest_target_start_time is None:
                    nearest_target_start_time = time.time()  # Start timer
                elif time.time() - nearest_target_start_time >= 2:
                    if saved_target_position is None:
                        saved_target_position = nearest_target  # Save position after 2 sec
                else:
                    nearest_target_start_time = None

                # Draw a line from the 'Center' object to the closest TARGET object
                cv2.line(latest_frame, center_object, nearest_target[0], (0, 255, 0), 2)  # 🟢 Green Line
                
                
                # If the same target remains the closest for at least 3 seconds, lock it
                if saved_target_position:
                    # Label the nearest target
                    cv2.putText(latest_frame, f"Nearest Target: {nearest_target[1]}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # **Draw a permanent RED dot at the center of the nearest target**
                    cv2.circle(latest_frame, nearest_target[0], 5, (0, 0, 255), -1)  # 🔴 Red Dot
            
                    line_dx = top_object[0] - bottom_object[0]
                    line_dy = top_object[1] - bottom_object[1]

                    target_dx = nearest_target[0][0] - bottom_object[0]
                    target_dy = nearest_target[0][1] - bottom_object[1]

                    line_angle = np.degrees(np.arctan2(line_dy, line_dx))
                    target_angle = np.degrees(np.arctan2(target_dy, target_dx))

                    angle_diff = target_angle - line_angle

                    # Normalize angle to -180 to 180
                    if angle_diff > 180:
                        angle_diff -= 360
                    elif angle_diff < -180:
                        angle_diff += 360

                    # Determine Turn Direction
                    if angle_diff > 10:  # Target is to the right
                        turn_direction = "R"
                        cv2.putText(latest_frame, "Turn Right", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif angle_diff < -10:  # Target is to the left
                        turn_direction = "L"
                        cv2.putText(latest_frame, "Turn Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        turn_direction = "S"  # Stay on course
                        cv2.putText(latest_frame, "Stay Straight", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Display angle difference for debugging
                    angle_text = f"Angle Diff: {angle_diff:.2f}°"
                    cv2.putText(latest_frame, angle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    network_msg = turn_direction

                
                # else:
                #     # If a new target is closest, reset the timer
                #     saved_target_position = nearest_target
                #     nearest_target_start_time = current_time
            else:
                # Reset tracking if no nearest target
                saved_target_position = None
                nearest_target_start_time = None

        if network_msg is None:
            network_msg = "S"





        # Send movement command only when it changes
        previous_msg = await send_msg_if_not_previous(websocket, previous_msg, network_msg)

        cv2.imshow("YOLOv11 Detection", latest_frame)
        cv2.waitKey(1)

        # Print processing latency
        # latency = time.time() - start_time
        # print(f"Latency: {latency:.4f} sec")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def main():
    async with websockets.connect(URI) as websocket:
        print("Connected to WebSocket")
        await process_and_send(websocket)

# Start frame capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# Run WebSocket & YOLO Processing
asyncio.run(main())

# Cleanup
capture.release()
cv2.destroyAllWindows()