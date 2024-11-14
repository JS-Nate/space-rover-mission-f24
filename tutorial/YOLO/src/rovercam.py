import cv2

# Replace with your Raspberry Pi camera feed URL
stream_url = "http://192.168.0.115:8081/stream.mjpg"

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Unable to open the video stream")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('Raspberry Pi Camera Feed', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
