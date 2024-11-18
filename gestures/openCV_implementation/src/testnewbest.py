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
import os
import logging
from ultralytics import YOLO
# Load the YOLO model
model = YOLO("best2.pt")

# Define obstacle and target categories
obstacles = {"Sun", "Galaxy", "Black Hole"}
targets = {"Asteroid", "Earth", "Saturn"}

# Angle threshold for deciding if the car needs to turn
angle_threshold = 20
currently_tracking = None
collected_objects = []
searching_phase = False
searching_start_time = None
stop_display_time = None  # To track when to stop displaying "Stop"

# # Set the logging level to ERROR or CRITICAL to suppress YOLO logs
# logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# # Suppress print statements and stderr if needed
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = sys.stdout  # Optional


# Function to calculate angle between vectors in degrees
def calculate_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))



# Add global flag for locked state
locked_in = False

# Timestamp to track when the target was last in "Getting Close" stage
getting_close_timestamp = None

# Timestamp variables for delays
forward_start_time = None

# Timestamp variables for delays
in_front_of_start_time = None
forward_start_time = None
stop_after_forward_start_time = None


# Timestamp variables for delays
in_front_of_start_time = None
forward_start_time = None
stop_after_forward_start_time = None

def get_navigation_direction(frame):
    global currently_tracking, collected_objects, searching_phase, stop_display_time, locked_in, getting_close_timestamp
    global in_front_of_start_time, forward_start_time, stop_after_forward_start_time

    height, width, _ = frame.shape
    bottom_center = (width // 2, height)  # Center bottom of the frame

    # Handle the "Stop" phase (after pressing 'x')
    if stop_display_time:
        elapsed_time = time.time() - stop_display_time
        if elapsed_time < 2:
            msg = "S", "Stop"
        else:
            stop_display_time = None
            searching_phase = True
            currently_tracking = None

    # Handle "In Front Of" behavior
    if in_front_of_start_time:
        elapsed_time = time.time() - in_front_of_start_time
        if elapsed_time < 2:  # Stop for 2 seconds
            msg = "S", "Stop" # in front of
        elif forward_start_time is None:  # Start moving forward
            forward_start_time = time.time()
        else:
            forward_elapsed = time.time() - forward_start_time
            if forward_elapsed < 3:  # Move forward for 1 second
                msg = "F", "Forward" #Moving Forward
            elif stop_after_forward_start_time is None:  # Start stopping for 4 seconds
                stop_after_forward_start_time = time.time()
            else:
                stop_after_forward_elapsed = time.time() - stop_after_forward_start_time
                if stop_after_forward_elapsed < 4:  # Stop for 4 seconds
                    msg = "S", "Stop" #Stopping After Forward
                else:  # Reset and exit locked-in phase
                    in_front_of_start_time = None
                    forward_start_time = None
                    stop_after_forward_start_time = None
                    locked_in = False
                    searching_phase = True  # Enter search phase if no target is found

    # Handle the searching phase
    if searching_phase:
        results = model.predict(frame)
        detected_objects = results[0].boxes #Add suppressions before and after this line
        found_new_target = False
        for obj in detected_objects:
            class_id = int(obj.cls[0])
            label = model.names[class_id]
            if label in targets and label not in collected_objects:
                found_new_target = True
                currently_tracking = label
                break

        if found_new_target:
            searching_phase = False
            searching_start_time = None
        else:
            msg = "L", "Left" #Searching - Turning

    # Perform object detection
    results = model.predict(frame)
    detected_objects = results[0].boxes
    target_centers = []
    for obj in detected_objects:
        class_id = int(obj.cls[0])
        label = model.names[class_id]

        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        color = (0, 0, 255) if label in obstacles else (0, 255, 0)

        # Outline all detected objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if label in targets:
            target_centers.append((center_x, center_y, label))

    # Check for target proximity
    if target_centers:
        nearest_target = min(target_centers, key=lambda p: np.linalg.norm(np.array(bottom_center) - np.array(p[:2])))
        target_center, target_label = nearest_target[:2], nearest_target[2]
        vector_to_target = np.array(target_center) - np.array(bottom_center)
        vertical_vector = np.array([0, -1])
        angle = calculate_angle(vector_to_target, vertical_vector)

        # Draw the line dynamically to the target
        line_color = (255, 0, 0)
        line_thickness = 2
        cv2.line(frame, bottom_center, tuple(target_center), line_color, line_thickness)
        label_position = (target_center[0] + 10, target_center[1] + 10)
        cv2.putText(frame, f"{target_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # "Getting Close" stage based on y-coordinate
        if target_center[1] > height * 0.8:  # Close to the bottom (80% of frame height)
            cv2.putText(frame, "Getting Close!", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            getting_close_timestamp = time.time()  # Update the timestamp

        if locked_in:
            if target_label == currently_tracking:
                msg = "F", "Forward"

        if angle > angle_threshold:
            if vector_to_target[0] > 0:
                msg = "R", "Right"
            else:
                msg = "L", "Left"
        else:
            msg = "F", "Forward"
            locked_in = True
            currently_tracking = target_label
    else:
        # Check if the target was "Getting Close" but is now undetected
        if getting_close_timestamp and time.time() - getting_close_timestamp > .5:
            getting_close_timestamp = None  # Reset the timestamp
            in_front_of_start_time = time.time()  # Trigger "In Front Of"
            return "S", "Stop" #In Front Of

        msg = "S", "Stop"

    return msg







def main():
    global locked_in, searching_phase

    cap = cv2.VideoCapture(0)
    # stream_url = "http://192.168.0.115:8081/stream.mjpg"  # Replace with your stream URL
    # cap = cv2.VideoCapture("http://192.168.0.115:8081/stream.mjpg")

    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        direction, message = get_navigation_direction(frame)
        cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Message: {message}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Self-Driving Car Navigation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):  # Exit lock-in mode and enter searching phase
            locked_in = False
            searching_phase = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
