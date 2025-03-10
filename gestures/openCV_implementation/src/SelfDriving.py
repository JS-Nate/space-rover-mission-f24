import asyncio
import cv2
import numpy as np
import torch
import websockets
import time
import threading
import os
import sys
from ultralytics import YOLO

print("Starting to connect to WebSocket...")
URI = "ws://kind-control-plane:32085/roversocket"

# ✅ **Load YOLO Model with GPU Acceleration (No ONNX)**
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = YOLO("overheadbest3.pt").to(device)
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    model = YOLO("overheadbest3.pt").to("cpu")  # Fallback to CPU

# ✅ **Threaded Video Capture (Zero Buffering)**
class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # 🔥 High FPS for real-time
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 🔥 Minimize buffering delay
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(0.0001)  # 🔥 Minimize CPU load while keeping FPS high

    def read(self):
        with self.lock:
            if self.frame is None:
                print("🚨 Camera feed error: Frame is None. Restarting camera...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)  # Restart camera
                return False, None
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# ✅ **Initialize Threaded Video Capture**
cap = VideoCapture(2)

# ✅ **Use OpenCV Optimized Settings**
cv2.setUseOptimized(True)

async def send_msg_if_not_previous(websocket, previous_msg, msg):
    """ Sends a message to the WebSocket only if it's different from the last one. """
    if msg != previous_msg:
        await websocket.send(msg)
        print("Sent message:", msg)
        previous_msg = msg
    return previous_msg

async def process_yolo(websocket):
    """ 🚀 Asynchronous YOLO Detection Task with Latency Logging. """
    global cap
    previous_msg = "S"
    processing = False  # 🔥 Prevent YOLO from slowing down frame updates

    # ✅ **Enable OpenCV CUDA if Available**
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"OpenCV CUDA Enabled: {use_cuda}")

    nearest_target_start_time = None
    saved_target_position = None
    locked_targets = {}

    while True:
        if processing:
            await asyncio.sleep(0.0001)  # 🔥 Skip frame if YOLO is still processing
            continue

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.002)
            continue

        try:
            processing = True  # 🔥 Prevent multiple frame processing at once
            start_time = time.time()  # 🔥 Start timing for latency measurement

            # ✅ **Use OpenCV CUDA for Faster Processing**
            if use_cuda:
                frame_gpu = cv2.cuda_GpuMat()
                frame_gpu.upload(frame)
                frame = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB).download()

            # ✅ **Run YOLO Inference on GPU (Frame Skipping Enabled)**
            results = model.predict(
                frame, verbose=False, device=device, conf=0.2, iou=0.4, agnostic_nms=True
            )

            # ✅ **Measure Latency**
            latency = time.time() - start_time  # 🔥 Time difference
            fps = 1 / latency if latency > 0 else 0  # 🔥 FPS Calculation
            print(f"🔥 YOLO Processing Latency: {latency:.4f} sec ({fps:.2f} FPS)")

            if results is None or len(results) == 0 or results[0].boxes is None:
                print("🚨 No detections found! Skipping frame...")
                processing = False
                continue  # Skip this frame

            boxes = results[0].boxes
            labels = results[0].names

            network_msg = "S"  # Default to Stop
            annotated_frame = frame.copy()
            bottom_object, top_object, center_object = None, None, None
            target_centers = []
            targets = ["Earth", "Saturn"]

            # ✅ **Process Detections & Draw Bounding Boxes**
            for i, box in enumerate(boxes):
                label = labels[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                color = (0, 255, 0) if label in targets else (255, 255, 255)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

                if label == "Bottom":
                    bottom_object = box_center
                elif label == "Top":
                    top_object = box_center
                elif label == "Center":
                    center_object = box_center
                elif label in targets:
                    target_centers.append((box_center, label))

            # ✅ **Draw Key Connections**
            if bottom_object and top_object:
                cv2.line(annotated_frame, bottom_object, top_object, (255, 0, 0), 2)

            # ✅ **Display FPS & Latency on Screen**
            cv2.putText(annotated_frame, f"Latency: {latency:.4f}s ({fps:.2f} FPS)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLO GPU Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            previous_msg = await send_msg_if_not_previous(websocket, previous_msg, network_msg)

        except Exception as e:
            print(f"❌ YOLO Processing Error: {e}")

        finally:
            processing = False  # 🔥 Release processing flag

async def main():
    async with websockets.connect(URI) as websocket:
        print("Connected to WebSocket")
        asyncio.create_task(process_yolo(websocket))
        while True:
            await asyncio.sleep(0.5)  # 🔥 Keep connection alive without lag

threading.Thread(target=cap.read, daemon=True).start()
asyncio.run(main())

cap.release()
cv2.destroyAllWindows()
