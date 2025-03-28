import cv2

def open_webcam():

    # 2 = laptop cam
    # 0 = usb cam
    source = 0

    # Open a connection to the webcam (usually /dev/video0 on Linux)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Reduce exposure (if supported by the camera)
    # The value for exposure depends on the camera driver; you may need to experiment.
    # Negative values often mean auto-exposure is disabled.
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow(f'Webcam (Source {source})', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_webcam()