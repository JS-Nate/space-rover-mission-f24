#################################################################################
# Copyright (c) 2022 IBM Corporation and others.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Eclipse Public License 2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#################################################################################

# Import all the important libraries
import asyncio
from asyncio import futures
import cv2
from cv2 import imshow
import cvzone
import mediapipe as mp
import websockets
from cvzone.HandTrackingModule import HandDetector
import time
import sys
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Define obstacle and target categories
obstacles = {"Sun", "Asteroids", "Black Hole"}
targets = {"Venus", "Jupiter", "Earth", "Saturn"}

print("Starting to connect")
#uri = "ws://192.168.0.101:9070/roversocket"
URI = "ws://kind-control-plane:32085/roversocket"

# Initialize the camera
cap = cv2.VideoCapture(0)

# Angle threshold for deciding if the car needs to turn
angle_threshold = 2
currently_tracking = None
collected_objects = []
searching_phase = False
searching_start_time = None
stop_display_time = None  # To track when to stop displaying "Stop"

# Function to calculate angle between vectors in degrees
def calculate_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# Asynchronously try to connect to the server


@asyncio.coroutine
def main():
    connecting = True

    while connecting:
        try:

            done, pending = yield from asyncio.wait([websockets.connect(URI)])

            # assert not pending
            future, = done
            print(future.result())
        except:
            print("Unable to connect to the machine")
            time.sleep(5)
        else:
            connecting = False


# Run the main loop until we are able to connect to the server
asyncio.get_event_loop().run_until_complete(main())


async def repl():
    async with websockets.connect(URI) as websocket:

        # Send successful connection message to the server
        print("Connected")

        # Speed Control parameters for the Rover
        previous = "S"

        # Font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Set up for fps counter
        # fps_counter = cvzone.FPS()
        window_name = "Hand Gesture Recognition Live Capture"

        # Use default capture device with default rendering
        capture = cv2.VideoCapture(0)
        # Window name
        cv2.namedWindow(window_name, cv2.WND_PROP_AUTOSIZE)

        # set to full screen on all OS
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)


        # Use 720 manual setting for the webcam. Static resolution values are used below so we must keep the
        # video feed constant.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Using cvzone library to detect and track the hand
        detector = HandDetector(detectionCon=0.6, minTrackCon=0.6, maxHands=1)

        # Capture continuous video
        while True:

            # Get image frame
            _, image = capture.read()  # read from the video

            # Get interruption key from keyboard
            k = cv2.waitKey(1)  # get input from keyboard

            # Flip image frame
            # cv2image1 = cv2.flip(image, 1)

            # Show FPS on screen
            # _, img = fps_counter.update(cv2image1, pos=(50, 80), color=(0, 255, 0), scale=3, thickness=3)
            

         
            network_msg = camera_track
            previous = await send_msg_if_not_previous(websocket, previous, network_msg)

            
            cv2.imshow(window_name, img)

            if k == 27:  # Press 'Esc' key to exit
                # await websocket.send("Hand Gesture Control connection closed.")
                break  # Exit while loop

        # ========================================

        # Close down window
        cv2.destroyAllWindows()
        # Disable your camera
        capture.release()


async def send_msg_if_not_previous(websocket, previous_msg, msg):
    '''Sends msg to the websocket so long as it is not the same string as 'previous_msg' '''
    if msg != previous_msg:
        if msg != "S":
            await websocket.send("S")
            print("Sent message", "S")
        await websocket.send(msg)
        print("Sent message", msg)
        previous_msg = msg
    return previous_msg


def get_direction_msg(first_finger_index, second_finger_index):
    '''Returns the network message and human readable direction given the first finger and second finger index information based on 720p resolution'''
    if (first_finger_index[3] - first_finger_index[1] > 75) and (second_finger_index[3] - second_finger_index[1] > 75) and (abs(first_finger_index[2] - first_finger_index[0]) < 75) and (abs(second_finger_index[2] - second_finger_index[0]) < 75):
        msg = "F", "Forward"
    elif ((first_finger_index[1] - first_finger_index[3] > 75) and (second_finger_index[1] - second_finger_index[3] > 75) and (abs(first_finger_index[2] - first_finger_index[0]) < 50) and (abs(second_finger_index[2] - second_finger_index[0]) < 50)):
        msg = "B", "Reverse"
    elif (first_finger_index[2] - first_finger_index[0] > 50) and (second_finger_index[2] - second_finger_index[0] > 50):
        msg = "L", "Left"
    elif (first_finger_index[0] - first_finger_index[2] > 50) and (second_finger_index[0] - second_finger_index[2] > 50):
        msg = "R", "Right"
    else:
        msg = "S", "Stop"
    return msg





def camera_track():
    ret, frame = cap.read()
    # if not ret:
    #     print("Failed to grab frame")
    #     break
    
    height, width, _ = frame.shape
    bottom_center = (width // 2, height)  # Center bottom of the frame
    
    # Handle the "Stop" phase (after pressing 'x')
    if stop_display_time:
        elapsed_time = time.time() - stop_display_time
        if elapsed_time < 2:
            # Display the "Stop" message for 2 seconds
            cv2.putText(frame, "S", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            msg = "S", "Stop"
        else:
            # After 2 seconds, enter searching phase
            stop_display_time = None
            searching_phase = True
            currently_tracking = None  # Reset tracking

    # Handle the searching phase
    if searching_phase:
        # Display "Searching Phase" message
        cv2.putText(frame, "R", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        msg = "R", "Right"
        # Wait for a new target to be found
        results = model.predict(frame)
        detected_objects = results[0].boxes
        found_new_target = False
        for obj in detected_objects:
            class_id = int(obj.cls[0])
            label = model.names[class_id]
            if label in targets and label not in collected_objects:
                found_new_target = True
                currently_tracking = label
                break
        
        if found_new_target:
            searching_phase = False  # Exit searching phase immediately
            searching_start_time = None  # Reset the start time
        else:
            # Turn right if no new target is detected
            cv2.putText(frame, "R", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            msg = "R", "Right"
            cv2.imshow("Object Detection with Path Tracking", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue

    # Perform object detection in tracking phase
    results = model.predict(frame)
    detected_objects = results[0].boxes
    
    # Gather target centers, excluding collected ones
    target_centers = []
    for obj in detected_objects:
        class_id = int(obj.cls[0])
        label = model.names[class_id]
        
        if label in collected_objects:
            continue  # Skip collected targets
        
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw bounding box and label
        color = (0, 0, 255) if label in obstacles else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if label in targets:
            target_centers.append((center_x, center_y, label))

    # If there are targets, find the nearest to bottom center
    if target_centers:
        nearest_target = min(target_centers, key=lambda p: np.linalg.norm(np.array(bottom_center) - np.array(p[:2])))
        target_center, target_label = nearest_target[:2], nearest_target[2]

        # Draw line to the nearest target
        cv2.line(frame, bottom_center, target_center, (255, 0, 0), 2)

        # Track the current target
        if currently_tracking != target_label:
            currently_tracking = target_label

        # Display currently tracking object
        cv2.putText(frame, f"Tracking: {currently_tracking}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Calculate direction angle
        vector_to_target = np.array(target_center) - np.array(bottom_center)
        vertical_vector = np.array([0, -1])
        angle = calculate_angle(vector_to_target, vertical_vector)

        if angle > angle_threshold:
            if vector_to_target[0] > 0:
                direction = "R"  # Turn right
                msg = "R", "Right"
            else:
                direction = "L"  # Turn left
                msg = "L", "Left"
        else:
            direction = "F"  # Go forward
            msg = "F", "Forward"
        
        # Display turn direction
        cv2.putText(frame, direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if angle > angle_threshold else (0, 255, 0), 2)
        
    # Display the list of collected planets
    y_offset = 200  # Starting vertical position for the collected list
    cv2.putText(frame, "Collected Planets:", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    for i, obj in enumerate(collected_objects, start=1):
        cv2.putText(frame, f"{i}. {obj}", (50, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Handle 'x' key to mark current target as collected and enter searching phase
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') and currently_tracking is not None:
        collected_objects.append(currently_tracking)
        cv2.putText(frame, f"Collected: {currently_tracking}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        currently_tracking = None  # Reset tracking
        stop_display_time = time.time()  # Start displaying "Stop"

    return msg

# Run the Hand Gesture Recognition ascynchronously after the connection works
asyncio.get_event_loop().run_until_complete(repl())
