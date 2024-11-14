import cv2
import numpy as np
import time
from ultralytics import YOLO

# Initialize the camera
# cap = cv2.VideoCapture(0)


# Initialize the camera stream from the Raspberry Pi
stream_url = "http://192.168.0.115:8081/stream.mjpg"  # Replace with your stream URL
cap = cv2.VideoCapture(stream_url)

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

# Function to calculate angle between vectors in degrees
def calculate_angle(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    height, width, _ = frame.shape
    bottom_center = (width // 2, height)  # Center bottom of the frame
    
    # Handle the "Stop" phase (after pressing 'x')
    if stop_display_time:
        elapsed_time = time.time() - stop_display_time
        if elapsed_time < 2:
            # Display the "Stop" message for 2 seconds
            cv2.putText(frame, "S", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # After 2 seconds, enter searching phase
            stop_display_time = None
            searching_phase = True
            currently_tracking = None  # Reset tracking

    # Handle the searching phase
    if searching_phase:
        # Display "Searching Phase" message
        cv2.putText(frame, "R", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

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
            cv2.imshow("Object Detection with Path Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

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

        # If the angle exceeds the threshold, decide whether to turn right or left based on the x-component of the vector.
        if angle > angle_threshold:
            if vector_to_target[0] > 0:
                direction = "R"  # Turn right
            else:
                direction = "L"  # Turn left
        else:
            direction = "F"  # Go forward

        
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

    # Display the frame with annotations
    cv2.imshow("Object Detection with Path Tracking", frame)

    # Press 'q' to quit the application
    if key == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
