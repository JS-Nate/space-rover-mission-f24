import cv2
from ultralytics import YOLO
import os
import numpy as np
import math

# Ensure we are working with the correct directory
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get directory of the script
model_path = os.path.join(script_dir, "best.pt")
image_path = os.path.join(script_dir, "IMG_1190.jpg")

# Check if model and image files exist
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    exit()
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    exit()

# Load the YOLO model
model = YOLO(model_path)

# Read the image file
image = cv2.imread(image_path)

# Check if the image file is read correctly
if image is None:
    print("Error: Could not read image file.")
    exit()

# Perform detection on the image
results = model.predict(image)

# Define classifications
targets = ["earth", "saturn"]

# Annotate image with detection results
annotated_image = image.copy()

# Access the results (boxes, labels, etc.)
boxes = results[0].boxes  # Accessing the first result
labels = results[0].names  # Accessing the labels

# Print detected labels to verify the output
detected_labels = [labels[int(box.cls)] for box in boxes]
print("Detected labels:", detected_labels)

# List of target centers, bottom, and top objects
bottom_object = None
top_object = None
target_centers = []

# Find the "bottom" and "top" objects and target centers
for i, box in enumerate(boxes):
    label = labels[int(box.cls)]  # Get the class label
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

    # Get the center of the bounding box
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    if label == "bottom":  # Look for the object labeled "bottom"
        bottom_object = box_center
    elif label == "top":  # Look for the object labeled "top"
        top_object = box_center
    elif label in targets:
        target_centers.append((box_center, label))

# If there is no object labeled "bottom" or "top", exit
if not bottom_object:
    print("Error: 'bottom' object not found.")
    exit()

if not top_object:
    print("Error: 'top' object not found.")
    exit()

# Draw a line from the 'bottom' object to the 'top' object
cv2.line(annotated_image, bottom_object, top_object, (0, 255, 255), 2)  # Yellow line

# Calculate direction based on relative positions of bottom and top objects
dx = top_object[0] - bottom_object[0]  # Difference in x
dy = top_object[1] - bottom_object[1]  # Difference in y

# Determine the direction based on the position of 'top' relative to 'bottom'
if abs(dy) > abs(dx):  # North or South direction
    if dy < 0:
        direction = "North"  # Top is above Bottom
    else:
        direction = "South"  # Bottom is below Top
else:  # East or West direction
    if dx > 0:
        direction = "East"  # Top is to the right of Bottom
    else:
        direction = "West"  # Top is to the left of Bottom

# Prepare the text to display the direction on the image
direction_text = f"Direction: {direction}"

# Display the direction on the image
cv2.putText(annotated_image, direction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text for direction

# Calculate the angle (in degrees) of the line between 'bottom' and 'top'
angle_rad = np.arctan2(dy, dx)  # Angle in radians
angle_deg = np.degrees(angle_rad)  # Convert angle to degrees

# Prepare the text to display the angle on the image
angle_text = f"Angle: {angle_deg:.2f}°"

# Display the angle on the image
cv2.putText(annotated_image, angle_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text for angle

# Find the nearest target
nearest_target = None
min_distance = float('inf')
for center, label in target_centers:
    distance = np.linalg.norm(np.array(center) - np.array(bottom_object))
    if distance < min_distance:
        min_distance = distance
        nearest_target = center

# Calculate the angle to the nearest target
if nearest_target:
    target_dx = nearest_target[0] - bottom_object[0]
    target_dy = nearest_target[1] - bottom_object[1]
    target_angle_rad = np.arctan2(target_dy, target_dx)
    target_angle_deg = np.degrees(target_angle_rad)

    # Determine the rotation direction
    angle_diff = target_angle_deg - angle_deg
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360

    if angle_diff > 0:
        rotation_direction = "Clockwise"
    else:
        rotation_direction = "Counterclockwise"

    # Prepare the text to display the rotation direction on the image
    rotation_text = f"Rotate: {rotation_direction}"

    # Display the rotation direction on the image
    cv2.putText(annotated_image, rotation_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)  # White text for rotation

# Draw bounding boxes and labels for the 'bottom', 'top', and targets
for i, box in enumerate(boxes):
    label_in_box = labels[int(box.cls)]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

    if label_in_box == "bottom":
        color = (0, 255, 255)  # Yellow for bottom
        text = "Bottom"
    elif label_in_box == "top":
        color = (255, 0, 255)  # Magenta for top
        text = "Top"
    elif label_in_box in targets:
        color = (0, 255, 0)  # Green for targets
        text = label_in_box.capitalize()
    else:
        continue

    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(annotated_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Resize the window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)

# Display the image with detections
cv2.imshow("YOLO Detection", annotated_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()