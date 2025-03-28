# BEFORE IMPLEMENTING LAG REDUCTION 02/27 10:23


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
model = YOLO("overheadbest3.pt")



async def repl():
    async with websockets.connect(URI) as websocket:

        # Send successful connection message to the server
        print("Connected")

        # Speed Control parameters for the Rover
        previous = "S"


        # Initialize the camera stream from the Raspberry Pi
        stream_url = "http://192.168.0.115:8081/stream.mjpg"  # Replace with your stream URL
        # capture = cv2.VideoCapture(stream_url)
        
        capture = cv2.VideoCapture(0)

        # Dictionary to track target confidence over time
        target_confidence = {}
        locked_targets = {}  # Dictionary to store locked target positions
        smoothed_boxes = {}
        alpha = 0.2  
        locked_target_position = False
        # Dictionary to track nearest target over time
        nearest_target_start_time = None
        saved_target_position = None  # Stores the saved position of the nearest target
        network_msg = None

        # Capture continuous video
        while True:

            # Get image frame
            ret, frame = capture.read()
            if not ret:
                continue

            results = model.predict(frame, verbose=False)
            targets = ["Earth", "Saturn"]


            # Access the results (boxes, labels, etc.)
            if len(results) > 0:
                boxes = results[0].boxes  # Accessing the first result
                labels = results[0].names  # Accessing the labels
            else:
                boxes = []
                labels = []

            bottom_object = None
            top_object = None
            center_object = None
            target_centers = []

            for i, box in enumerate(boxes):
                label = labels[int(box.cls)]  # Get the class label
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if label == "Bottom":
                    bottom_object = box_center
                elif label == "Top":
                    top_object = box_center
                elif label == "Center":
                    center_object = box_center
                elif label in targets:
                    target_centers.append((box_center, label))
            
            if top_object is None or bottom_object is None:
                print("Error: top_object or bottom_object is None")
                continue  # Skip the rest of the loop iteration

            cv2.line(frame, bottom_object, top_object, (0, 255, 255), 2)
            # Calculate direction based on relative positions of bottom and top objects
            dx = top_object[0] - bottom_object[0]
            dy = top_object[1] - bottom_object[1]
            # print(dx, dy)

            if abs(dy) > abs(dx):
                direction = "North" if dy < 0 else "South"
            else:
                direction = "East" if dx > 0 else "West"
            

            # Calculate the angle (in degrees) of the line between 'bottom' and 'top'
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)


            # Find the nearest target
            nearest_target = None
            nearest_target_label = None
            min_distance = float('inf')
            focused = False
            for center, label in target_centers:
                distance = np.linalg.norm(np.array(center) - np.array(bottom_object))
                if distance < min_distance:
                    min_distance = distance
                    nearest_target = center
                    nearest_target_label = label


                    # If we have a nearest target, track it over time
                    if nearest_target:
                        if nearest_target_start_time is None:
                            nearest_target_start_time = time.time()  # Start timer
                        elif time.time() - nearest_target_start_time >= 2:
                            if saved_target_position is None:
                                saved_target_position = nearest_target  # Save position after 2 sec
                    else:
                        nearest_target_start_time = None  # Reset timer if no target found
                    
                    # Draw the saved red dot (if captured)
                    if saved_target_position:
                        # nearest_target = None  # Stop drawing the nearest target once saved_target_position is set

                        


                        target_dx = saved_target_position[0] - bottom_object[0]
                        target_dy = saved_target_position[1] - bottom_object[1]
                        target_angle_rad = np.arctan2(target_dy, target_dx)
                        target_angle_deg = np.degrees(target_angle_rad)

                        angle_diff = target_angle_deg - angle_deg
                        if angle_diff > 180:
                            angle_diff -= 360
                        elif angle_diff < -180:
                            angle_diff += 360

                        # nearest_target_text = f"Nearest Target: {nearest_target_label.capitalize()}"

                        if center_object is not None and saved_target_position is not None:
                            line_length = np.linalg.norm(np.array(center_object) - np.array(saved_target_position))
                            length_text = f"Line Length: {line_length:.2f} pixels"
                        else:
                            line_length = 0  # or some default value

                        
                        # Initialize a flag to remember if F has already been printed
                        f_printed = False

                        # Indicate on screen when the line is pointing at the nearest target
                        if abs(angle_diff) <= 10:
                            # If we're in focus mode and "Sent Message F" has not been printed yet
                            if line_length >= 60 and not f_printed:
                                # print("Sent Message F")
                                network_msg = "F"

                                f_printed = True  # Set flag to indicate F has been printed

                            # Ensure the focus mode is activated
                            focused = True

                            # Update the focused text on screen
                            # focused_text = f"Focused: {focused}"

                            if line_length < 76:  # Threshold for "very small" distance

                                # Display the distance to the target on the screen
                                # print("Sent Message S")
                                network_msg = "S"

                                # Start a timer for 5 seconds to remove the current target
                                if nearest_target_label not in locked_targets:
                                    locked_targets[nearest_target_label] = time.time()
                                    # print(locked_targets)

                                # Check if 5 seconds have passed
                                if time.time() - locked_targets[nearest_target_label] >= 5:
                                    # Remove the current target from detection
                                    target_centers = [tc for tc in target_centers if tc[1] != nearest_target_label]
                                    del locked_targets[nearest_target_label]
                                    saved_target_position = None  # Reset saved target position
                                    focused = False
                                    f_printed = False  # Reset flag when focus mode ends
                                else:
                                    # Ensure network_msg remains "S" during the 5 seconds
                                    network_msg = "S"

                        else:
                            if focused == False:
                                if angle_diff > 0:
                                    rotation_direction = "Right"
                                    # print("Sent Message R")
                                    network_msg = "R"
                                else:
                                    rotation_direction = "Left"
                                    # print("Sent Message L")
                                    network_msg = "L"

                                # cv2.putText(annotated_frame, f"Rotate {rotation_direction}", (20, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)



            if network_msg is None:
                network_msg = "S"

    
            previous = await send_msg_if_not_previous(websocket, previous, network_msg)


            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Get the network message and human-readable message for navigation
            # network_msg, human_msg = get_navigation_direction(image)        

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
            print("Sents message", "S")
        await websocket.send(msg)
        print("Sents message", msg)
        previous_msg = msg
    return previous_msg




# Run the Hand Gesture Recognition ascynchronously after the connection works
asyncio.get_event_loop().run_until_complete(repl())