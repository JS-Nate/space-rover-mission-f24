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
model = YOLO("overheadbest.pt")

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
        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Using cvzone library to detect and track the hand
        # detector = HandDetector(detectionCon=0.6, minTrackCon=0.6, maxHands=1)



        # Capture continuous video
        while True:

            # Get image frame
            _, image = capture.read()  # read from the video
            # Reverse the image feed horizontally and vertically
            # image = cv2.flip(image, 1)

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









def get_navigation_direction(frame):

    # Set the logging level to ERROR or CRITICAL to suppress YOLO logs
    logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

    # YOLO prediction logic
    results = model.predict(frame, verbose=False)

    # Define classifications
    targets = ["earth", "saturn"]

    # Annotate frame with detection results
    # annotated_frame = frame.copy()

    # Access the results (boxes, labels, etc.)
    if len(results) > 0:
        boxes = results[0].boxes  # Accessing the first result
        labels = results[0].names  # Accessing the labels
    else:
        boxes = []
        labels = []

    # List of target centers, bottom, top, and center objects
    bottom_object = None
    top_object = None
    center_object = None
    target_centers = []

    # Initialize bottom_object as None
    bottom_object = None

    for i, box in enumerate(boxes):
        label = labels[int(box.cls)]  # Get the class label
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # Get the center of the bounding box
        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if label.lower() == "bottom":
            if bottom_object is None or box_center[1] < bottom_object[1]:  
                bottom_object = box_center  # Choose the topmost (smallest y-value) "bottom" object

        elif label == "top":
            top_object = box_center
        elif label == "center":
            center_object = box_center
        elif label in targets:
            target_centers.append((box_center, label))
        # print(f"Detected label: {label}, Center: {box_center}")
        # print(f"Final bottom_object: {bottom_object}")  # Debugging print



    # If bottom or top is missing, return None
    if bottom_object is None or top_object is None:
        return "S", "Stop"

    # Draw a line from the 'bottom' object to the 'top' object
    cv2.line(frame, bottom_object, top_object, (0, 255, 255), 2)
    print(bottom_object, top_object)  # Now this should print correctly on every frame update

    return "S", "Stop"













# def get_navigation_direction(frame):

#     # Set the logging level to ERROR or CRITICAL to suppress YOLO logs
#     logging.getLogger('ultralytics').setLevel(logging.CRITICAL)



#     # Dictionary to track target confidence over time
#     target_confidence = {}
#     locked_targets = {}  # Dictionary to store locked target positions
#     # Dictionary to track nearest target over time
#     nearest_target_start_time = None
#     saved_target_position = None  # Stores the saved position of the nearest target


#     # while True:
#     # Capture frame-by-frame
#     # ret, frame = cap.read()

#     # if not ret:
#         # print("Error: Could not read frame from webcam.")
#         # break

#     # Suppress print statements and stderr if needed
#     sys.stdout = open(os.devnull, 'w')
#     sys.stderr = sys.stdout  # Optional

#     # YOLO prediction logic
#     results = model.predict(frame, verbose=False)
#     # Re-enable printing
#     sys.stdout = sys.__stdout__
#     sys.stderr = sys.__stderr__  # Re-enable stderr if it was redirected


#     # Define classifications
#     targets = ["earth", "saturn"]

#     # Annotate frame with detection results
#     annotated_frame = frame.copy()

#     # Access the results (boxes, labels, etc.)
#     if len(results) > 0:
#         boxes = results[0].boxes  # Accessing the first result
#         labels = results[0].names  # Accessing the labels
#     else:
#         boxes = []
#         labels = []

#     # List of target centers, bottom, top, and center objects
#     bottom_object = None
#     top_object = None
#     center_object = None
#     target_centers = []

#     # Find the "bottom", "top", and "center" objects and target centers
#     for i, box in enumerate(boxes):
#         label = labels[int(box.cls)]  # Get the class label
#         x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
#         # Get the center of the bounding box
#         box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

#         if label == "bottom":
#             bottom_object = box_center
#         elif label == "top":
#             top_object = box_center
#         elif label == "center":
#             center_object = box_center
#         elif label in targets:
#             target_centers.append((box_center, label))

#     # If there is no object labeled "bottom" or "top", display the frame without annotations
#     if not bottom_object or not top_object:
#         cv2.imshow("YOLO Detection", frame)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#         # continue

#     # Draw a line from the 'bottom' object to the 'top' object
#     cv2.line(annotated_frame, bottom_object, top_object, (0, 255, 255), 2)
#     # print(bottom_object, top_object)


#     # Calculate direction based on relative positions of bottom and top objects
#     dx = top_object[0] - bottom_object[0]
#     dy = top_object[1] - bottom_object[1]

#     # if abs(dy) > abs(dx):
#     #     direction = "North" if dy < 0 else "South"
#     # else:
#     #     direction = "East" if dx > 0 else "West"

#     # direction_text = f"Direction: {direction}"
#     # cv2.putText(annotated_frame, direction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

#     # Calculate the angle (in degrees) of the line between 'bottom' and 'top'
#     angle_rad = np.arctan2(dy, dx)
#     angle_deg = np.degrees(angle_rad)



#     # print("womp")
#     cv2.circle(annotated_frame, saved_target_position, 5, (0, 0, 255), -1)
#     print(saved_target_position)


#     # Find the nearest target
#     nearest_target = None
#     nearest_target_label = None
#     min_distance = float('inf')
#     focused = False
#     for center, label in target_centers:
#         distance = np.linalg.norm(np.array(center) - np.array(bottom_object))
#         if distance < min_distance:
#             min_distance = distance
#             nearest_target = center
#             nearest_target_label = label
#             # print (nearest_target_label)

#             # If we have a nearest target, track it over time
#             if nearest_target:
#                 if nearest_target_start_time is None:
#                     nearest_target_start_time = time.time()  # Start timer
#                 elif time.time() - nearest_target_start_time >= 2:
#                     if saved_target_position is None:
#                         saved_target_position = nearest_target  # Save position after 2 sec
#             else:
#                 nearest_target_start_time = None  # Reset timer if no target found
            
#             # Draw the saved red dot (if captured)
#             if saved_target_position:
#                 # cv2.circle(annotated_frame, saved_target_position, 5, (0, 0, 255), -1)
#                 # nearest_target = None  # Stop drawing the nearest target once saved_target_position is set


#                 target_dx = saved_target_position[0] - bottom_object[0]
#                 target_dy = saved_target_position[1] - bottom_object[1]
#                 target_angle_rad = np.arctan2(target_dy, target_dx)
#                 target_angle_deg = np.degrees(target_angle_rad)

#                 angle_diff = target_angle_deg - angle_deg
#                 if angle_diff > 180:
#                     angle_diff -= 360
#                 elif angle_diff < -180:
#                     angle_diff += 360

#                 # print("Angle Diff:", angle_diff)

#                 nearest_target_text = f"Nearest Target: {nearest_target_label.capitalize()}"
#                 cv2.putText(annotated_frame, nearest_target_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

#                 # Draw a line from 'center' to the nearest target
#                 # if center_object:
#                 cv2.line(annotated_frame, center_object, saved_target_position, (255, 0, 0), 2)
#                 line_length = np.linalg.norm(np.array(center_object) - np.array(saved_target_position))
#                 length_text = f"Line Length: {line_length:.2f} pixels"
#                 cv2.putText(annotated_frame, length_text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

#                 # Display the angle difference on the screen
#                 angle_diff_text = f"Angle Diff: {angle_diff:.2f} degrees"
#                 cv2.putText(annotated_frame, angle_diff_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


                
#                 # Initialize a flag to remember if F has already been printed
#                 f_printed = False

#                 # Indicate on screen when the line is pointing at the nearest target
#                 if abs(angle_diff) <= 10:
#                     # If we're in focus mode and "Sent Message F" has not been printed yet
#                     if line_length >= 60 and not f_printed:
#                         print("Sent Message F")
#                         cv2.putText(annotated_frame, f"Forward", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

#                         f_printed = True  # Set flag to indicate F has been printed
#                         return "F", "Forward"


#                     # Ensure the focus mode is activated
#                     focused = True

#                     # Update the focused text on screen
#                     focused_text = f"Focused: {focused}"
#                     cv2.putText(annotated_frame, focused_text, (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

#                     if line_length < 60:  # Threshold for "very small" distance
#                         cv2.putText(annotated_frame, "Got it, Stop", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


#                         # Start a timer for 5 seconds to remove the current target
#                         if nearest_target_label not in locked_targets:
#                             locked_targets[nearest_target_label] = time.time()
#                             print(locked_targets)

#                         # Check if 5 seconds have passed
#                         if time.time() - locked_targets[nearest_target_label] >= 5:
#                             # Remove the current target from detection
#                             target_centers = [tc for tc in target_centers if tc[1] != nearest_target_label]
#                             del locked_targets[nearest_target_label]
#                             saved_target_position = None  # Reset saved target position
#                             focused = False
#                             f_printed = False  # Reset flag when focus mode ends

#                         else:
#                             # Display the distance to the target on the screen
#                             print("Sent Message S")
#                             return "S", "Stop"

#                 else:
#                     if focused == False:
#                         if angle_diff > 0:
#                             rotation_direction = "Right"
#                             print("Sent Message R")
#                             return "R", "Right"
#                         else:
#                             rotation_direction = "Left"
#                             print("Sent Message L")
#                             return "L", "Left"

#                         cv2.putText(annotated_frame, f"Rotate {rotation_direction}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)




#         # Draw bounding boxes and labels
#         for i, box in enumerate(boxes):
#             label_in_box = labels[int(box.cls)]
#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
#             color = (0, 255, 0) if label_in_box in targets else (0, 255, 255) if label_in_box == "bottom" else (255, 0, 255) if label_in_box == "top" else (255, 255, 255)
#             text = f"{label_in_box.capitalize()}"
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
#             cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

#         # Resize and display the frame
#         cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("YOLO Detection", 1280, 720)
#         cv2.imshow("YOLO Detection", annotated_frame)
#         cv2.waitKey(1)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # print("Stop")












# Run the Hand Gesture Recognition ascynchronously after the connection works
asyncio.get_event_loop().run_until_complete(repl())