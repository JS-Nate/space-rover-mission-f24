


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

# Initialize the camera
# cap = cv2.VideoCapture(0)


# Initialize the camera stream from the Raspberry Pi
# stream_url = "http://192.168.0.115:8081/stream.mjpg"  # Replace with your stream URL
# cap = cv2.VideoCapture(stream_url)

# Load the YOLO model
model = YOLO("best.pt")

# Define obstacle and target categories
obstacles = {"Sun", "Asteroids", "Black Hole"}
targets = {"Venus", "Jupiter", "Earth", "Saturn"}

# Angle threshold for deciding if the car needs to turn
angle_threshold = 2
currently_tracking = None
collected_objects = []
searching_phase = False
searching_start_time = None
stop_display_time = None  # To track when to stop displaying "Stop"








capture = cv2.VideoCapture(0)




image = capture.read()



cv2.imshow("Object Detection with Path Tracking", image)


def get_navigation_direction(frame):
    global currently_tracking, collected_objects, searching_phase, stop_display_time

    height, width, _ = frame.shape
    bottom_center = (width // 2, height)  # Center bottom of the frame
    
    # # Handle the "Stop" phase (after pressing 'x')
    if stop_display_time:
        elapsed_time = time.time() - stop_display_time
        if elapsed_time < 2:
            # Display the "Stop" message for 2 seconds
            print("Returning Stop phase")  # Debug log
            # return ("S", "Stop")  # Return as tuple
            msg = "S", "Stop"
        else:
            # After 2 seconds, enter searching phase
            stop_display_time = None
            searching_phase = True
            currently_tracking = None  # Reset tracking

    # # Handle the searching phase
    if searching_phase:
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
            print("Searching phase - Returning Right turn")  # Debug log
            # return ("R", "Searching - Turn Right")  # Return as tuple
            msg = "R", "Right"

    # Perform object detection in tracking phase







    # Set the logging level to ERROR or CRITICAL to suppress YOLO logs
    logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

    # Suppress print statements and stderr if needed
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = sys.stdout  # Optional

    # YOLO prediction logic
    results = model.predict(frame)

    # Re-enable printing
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__  # Re-enable stderr if it was redirected






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

        # Calculate direction angle
        vector_to_target = np.array(target_center) - np.array(bottom_center)
        vertical_vector = np.array([0, -1])
        angle = calculate_angle(vector_to_target, vertical_vector)

        # If the angle exceeds the threshold, decide whether to turn right or left based on the x-component of the vector.
        if angle > angle_threshold:
            if vector_to_target[0] > 0:
                msg = "R", "Right"
            else:
                msg = "L", "Left"
        else:
            msg = "F", "Forward"

        # Track the current target
        if currently_tracking != target_label:
            currently_tracking = target_label

        

        # print(f"Returning direction: {direction}, message: {human_msg}")  # Debug log
        # return (direction, human_msg)  # Return as tuple
    
    # If no targets are detected, return 'S' for stop
    else:
        print("Returning Stop phase")  # Debug log
        msg = "S", "Stop"
    
    return msg

