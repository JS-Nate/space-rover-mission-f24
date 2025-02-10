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
import os
import logging
import keyboard
from ultralytics import YOLO

print("Starting to connect")
#uri = "ws://192.168.0.101:9070/roversocket"
URI = "ws://kind-control-plane:32085/roversocket"


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



# Load the YOLO model
model = YOLO("best2.pt")

# Define obstacle and target categories
obstacles = {"Sun", "Galaxy", "Black Hole"}
targets = {"Asteroid", "Earth", "Saturn"}

# Angle threshold for deciding if the car needs to turn
angle_threshold = 30
currently_tracking = None
collected_objects = []
searching_phase = False
searching_start_time = None
stop_display_time = None  # To track when to stop displaying "Stop"



async def repl():
    async with websockets.connect(URI) as websocket:

        # Send successful connection message to the server
        print("Connected")

        # Speed Control parameters for the Rover
        previous = "S"

        # Font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Set up for fps counter
        fps_counter = cvzone.FPS()
        window_name = "Hand Gesture Recognition Live Capture"


        # Initialize the camera stream from the Raspberry Pi
        stream_url = "http://192.168.0.115:8081/stream.mjpg"  # Replace with your stream URL
        # capture = cv2.VideoCapture(stream_url)
        
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

            # Show FPS on screen
            _, img = fps_counter.update(image, pos=(50, 80), color=(0, 255, 0), scale=3, thickness=3)


            # Get the network message and human-readable message for navigation
            network_msg, human_msg = get_navigation_direction(image)
            # print(human_msg)

            # Now you can use both network_msg and human_msg as needed
            cv2.putText(img, f'Gesture Detected: {human_msg}', (465, 140), font, 1.2, (255, 100, 0), 2, cv2.LINE_AA)

            # Send the network message to the server, ensuring we don't send the same message twice
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


# Function to calculate angle between vectors in degrees
def calculate_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# Add global flag for locked state
locked_in = False

# Timestamp to track when the target was last in "Getting Close" stage
getting_close_timestamp = None

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
            return "S", "Stop"
        else:
            stop_display_time = None
            searching_phase = True
            currently_tracking = None

    # Handle "In Front Of" behavior
    if in_front_of_start_time:
        elapsed_time = time.time() - in_front_of_start_time
        if elapsed_time < 2:  # Stop for 2 seconds
            return "S", "Stop"  # In Front Of
        elif forward_start_time is None:  # Start moving forward
            forward_start_time = time.time()
        else:
            forward_elapsed = time.time() - forward_start_time
            if forward_elapsed < 2:  # Move forward for 1 second
                return "F", "Forward"  # Moving Forward
            elif stop_after_forward_start_time is None:  # Start stopping for 4 seconds
                stop_after_forward_start_time = time.time()
            else:
                stop_after_forward_elapsed = time.time() - stop_after_forward_start_time
                if stop_after_forward_elapsed < 4:  # Stop for 4 seconds
                    return "S", "Stop"  # Stopping After Forward
                else:  # Reset and exit locked-in phase
                    in_front_of_start_time = None
                    forward_start_time = None
                    stop_after_forward_start_time = None
                    locked_in = False
                    searching_phase = True  # Enter search phase if no target is found

    # Handle the searching phase
    if searching_phase:
        results = model.predict(frame)
        detected_objects = results[0].boxes
        for obj in detected_objects:
            class_id = int(obj.cls[0])
            label = model.names[class_id]
            if label in targets and label not in collected_objects:
                currently_tracking = label
                searching_phase = False
                locked_in = False
                break
        else:
            return "L", "Left"  # Searching, turning left

    # # YOLO prediction logic
    # results = model.predict(frame)

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
    target_centers = []
    for obj in detected_objects:
        class_id = int(obj.cls[0])
        label = model.names[class_id]
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
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
        cv2.line(frame, bottom_center, tuple(target_center), (255, 0, 0), 2)
        cv2.putText(frame, f"{target_label}", (target_center[0] + 10, target_center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Handle "Getting Close" stage
        if target_center[1] > height * 0.8:  # Close to the bottom (80% of frame height)
            cv2.putText(frame, "Getting Close!", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            getting_close_timestamp = time.time()  # Update the timestamp

        if locked_in:
            if target_label == currently_tracking:
                return "F", "Forward"
        else:
            if angle > angle_threshold:
                return "R" if vector_to_target[0] > 0 else "L", "Turning"
            else:
                locked_in = True
                currently_tracking = target_label
                return "F", "Forward"
    else:
        # Check if the target was "Getting Close" but is now undetected
        if getting_close_timestamp and time.time() - getting_close_timestamp > 3:
            getting_close_timestamp = None  # Reset the timestamp
            in_front_of_start_time = time.time()  # Trigger "In Front Of"
            return "S", "In Front Of"
        return "S", "Stop"

    return "S", "Stop"  # Default safety stop









# Debugging one

# def get_navigation_direction(frame):
#     global currently_tracking, collected_objects, searching_phase, stop_display_time, locked_in, getting_close_timestamp
#     global in_front_of_start_time, forward_start_time, stop_after_forward_start_time, last_command_time

#     height, width, _ = frame.shape
#     bottom_center = (width // 2, height)

#     # YOLO prediction logic
#     results = model.predict(frame)
#     detected_objects = results[0].boxes
#     target_centers = []

#     for obj in detected_objects:
#         class_id = int(obj.cls[0])
#         label = model.names[class_id]

#         x1, y1, x2, y2 = map(int, obj.xyxy[0])
#         center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#         color = (0, 0, 255) if label in obstacles else (0, 255, 0)

#         # Draw bounding boxes and labels
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         if label in targets:
#             target_centers.append((center_x, center_y, label))

#     # Check for target proximity
#     if target_centers:
#         nearest_target = min(target_centers, key=lambda p: np.linalg.norm(np.array(bottom_center) - np.array(p[:2])))
#         target_center, target_label = nearest_target[:2], nearest_target[2]
#         vector_to_target = np.array(target_center) - np.array(bottom_center)
#         vertical_vector = np.array([0, -1])
#         angle = calculate_angle(vector_to_target, vertical_vector)

#         # Overlay debugging info
#         cv2.line(frame, bottom_center, tuple(target_center), (255, 0, 0), 2)
#         print(f"Target Label: {target_label}")
#         print(f"Angle: {angle:.2f}")

#         # if keyboard.is_pressed('x'):  # Checks if 'x' is pressed
#         #     print("Turned camera IRL to face the object, should give F, forward now")



#         # Direction logic
#         horizontal_distance = vector_to_target[0]
#         if abs(horizontal_distance) < 20 and angle < angle_threshold:
#             return "F", "Forward"
#         elif horizontal_distance > 0:
#             return "R", "Right"
#         else:
#             return "L", "Left"

#     return "S", "Stop"







# Run the Hand Gesture Recognition ascynchronously after the connection works
asyncio.get_event_loop().run_until_complete(repl())
