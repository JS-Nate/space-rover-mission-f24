import cv2
from ultralytics import YOLO
import os
import numpy as np
import time
import sys

# Ensure we are working with the correct directory
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get directory of the script
model_path = os.path.join(script_dir, "overheadbest3.pt")

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()

# Load the YOLO model
model = YOLO(model_path)

# Access the webcam
cap = cv2.VideoCapture(0)  # Try changing the index if the webcam does not open

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Dictionary to track target confidence over time
target_confidence = {}
locked_targets = {}  # Dictionary to store locked target positions
smoothed_boxes = {}
alpha = 0.2  
locked_target_position = False
# Dictionary to track nearest target over time
nearest_target_start_time = None
saved_target_position = None  # Stores the saved position of the nearest target



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Suppress print statements and stderr if needed
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = sys.stdout  # Optional

    # YOLO prediction logic
    results = model.predict(frame, verbose=False)
    # Re-enable printing
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__  # Re-enable stderr if it was redirected


    # Define classifications
    targets = ["Earth", "Saturn"]

    # Annotate frame with detection results
    annotated_frame = frame.copy()

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


    # Find the "bottom", "top", and "center" objects and target centers
    for i, box in enumerate(boxes):
        label = labels[int(box.cls)]  # Get the class label
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Get the center of the bounding box
        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if label == "Bottom":
            bottom_object = box_center
        elif label == "Top":
            top_object = box_center
        elif label == "Center":
            center_object = box_center
        elif label in targets:
            target_centers.append((box_center, label))
        # Print the positions of top_object, center_object, and bottom_object
        if top_object:
            print(f"Top Object Position: {top_object}")
        if center_object:
            print(f"Center Object Position: {center_object}")
        if bottom_object:
            print(f"Bottom Object Position: {bottom_object}")

    # If there is no object labeled "bottom" or "top", display the frame without annotations
    if not bottom_object or not top_object:
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Draw a line from the 'bottom' object to the 'top' object
    cv2.line(annotated_frame, bottom_object, top_object, (0, 255, 255), 2)
    # print(bottom_object, top_object)

    # Calculate direction based on relative positions of bottom and top objects
    dx = top_object[0] - bottom_object[0]
    dy = top_object[1] - bottom_object[1]
    print(dx, dy)

    if abs(dy) > abs(dx):
        direction = "North" if dy < 0 else "South"
    else:
        direction = "East" if dx > 0 else "West"

    # direction_text = f"Direction: {direction}"
    # cv2.putText(annotated_frame, direction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate the angle (in degrees) of the line between 'bottom' and 'top'
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)



    # print("womp")
    cv2.circle(annotated_frame, saved_target_position, 5, (0, 0, 255), -1)
    # print(saved_target_position)


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
                # cv2.circle(annotated_frame, saved_target_position, 5, (0, 0, 255), -1)
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

                nearest_target_text = f"Nearest Target: {nearest_target_label.capitalize()}"
                cv2.putText(annotated_frame, nearest_target_text, (20, annotated_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw a line from 'center' to the nearest target
                # if center_object:
                cv2.line(annotated_frame, center_object, saved_target_position, (255, 0, 0), 2)


                if center_object is not None and saved_target_position is not None:
                    line_length = np.linalg.norm(np.array(center_object) - np.array(saved_target_position))
                    length_text = f"Line Length: {line_length:.2f} pixels"
                    # cv2.putText(annotated_frame, length_text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    line_length = 0  # or some default value



                length_text = f"Line Length: {line_length:.2f} pixels"
                # cv2.putText(annotated_frame, length_text, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                # Display the angle difference on the screen
                angle_diff_text = f"Angle Diff: {angle_diff:.2f} degrees"
                # cv2.putText(annotated_frame, angle_diff_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


                
                # Initialize a flag to remember if F has already been printed
                f_printed = False

                # Indicate on screen when the line is pointing at the nearest target
                if abs(angle_diff) <= 10:
                    # If we're in focus mode and "Sent Message F" has not been printed yet
                    if line_length >= 60 and not f_printed:
                        print("Sent Message F")
                        cv2.putText(annotated_frame, f"Forward", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

                        f_printed = True  # Set flag to indicate F has been printed

                    # Ensure the focus mode is activated
                    focused = True

                    # Update the focused text on screen
                    focused_text = f"Focused: {focused}"
                    # cv2.putText(annotated_frame, focused_text, (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                    if line_length < 60:  # Threshold for "very small" distance
                        cv2.putText(annotated_frame, "Stop", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                        # Display the distance to the target on the screen
                        print("Sent Message S")

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
                    if focused == False:
                        if angle_diff > 0:
                            rotation_direction = "Right"
                            print("Sent Message R")
                        else:
                            rotation_direction = "Left"
                            print("Sent Message L")

                        cv2.putText(annotated_frame, f"Rotate {rotation_direction}", (20, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)






    #         # Calculate the angle to the nearest target
    #         if nearest_target:
    #             target_dx = nearest_target[0] - bottom_object[0]
    #             target_dy = nearest_target[1] - bottom_object[1]
    #             target_angle_rad = np.arctan2(target_dy, target_dx)
    #             target_angle_deg = np.degrees(target_angle_rad)

    #             angle_diff = target_angle_deg - angle_deg
    #             if angle_diff > 180:
    #                 angle_diff -= 360
    #             elif angle_diff < -180:
    #                 angle_diff += 360

    #             nearest_target_text = f"Nearest Target: {nearest_target_label.capitalize()}"
    #             cv2.putText(annotated_frame, nearest_target_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #             # Draw a line from 'center' to the nearest target
    #             if center_object:
    #                 cv2.line(annotated_frame, center_object, nearest_target, (255, 0, 0), 2)

    #             # Display the angle difference on the screen
    #             angle_diff_text = f"Angle Diff: {angle_diff:.2f} degrees"
    #             cv2.putText(annotated_frame, angle_diff_text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
    #             # Indicate on screen when the line is pointing at the nearest target
    #             if abs(angle_diff) <= 10:
    #                 cv2.putText(annotated_frame, "Aligned with Target", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #                 distance_to_target = np.linalg.norm(np.array(center_object) - np.array(nearest_target))
    #                 print("Sent Message F")
                    
    #                 if distance_to_target < 50:  # Threshold for "very small" distance
    #                     cv2.putText(annotated_frame, "Stop", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
    #             else:
    #                 if nearest_target_label not in locked_targets:
    #                     if angle_diff > 0:
    #                         rotation_direction = "Clockwise"
    #                         print("Sent Message R")
    #                     else:
    #                         rotation_direction = "Counterclockwise"
    #                         print("Sent Message L")

    #                     cv2.putText(annotated_frame, f"Rotate {rotation_direction}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)






 






















    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        label_in_box = labels[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        color = (0, 255, 0) if label_in_box in targets else (0, 255, 255) if label_in_box == "bottom" else (255, 0, 255) if label_in_box == "top" else (255, 255, 255)
        text = f"{label_in_box.capitalize()}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Resize and display the frame
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 1280, 720)
    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print("Stop")












# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()