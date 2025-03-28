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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = YOLO("best.pt").to(device)
except Exception as e:
    print(f"model failed {e}")
    model = YOLO("best.pt").to("cpu")  

class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)  
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
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
            time.sleep(0.0001) 

    def read(self):
        with self.lock:
            if self.frame is None:
                print("camera feed error")
                self.cap.release()
                self.cap = cv2.VideoCapture(0) 
                return False, None
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cap = VideoCapture(0)

cv2.setUseOptimized(True)

async def send_msg_if_not_previous(websocket, previous_msg, msg):
    """ Sends a message to the WebSocket only if it's different from the last one. """
    if msg != previous_msg:
        if msg != "S":
            await websocket.send("S")
            print("Sents message", "S")
        await websocket.send(msg)
        print("Sents message", msg)
        previous_msg = msg
    return previous_msg








async def process_yolo(websocket):
    global cap
    previous_msg = "S"
    processing = False  

    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"OpenCV CUDA Enabled: {use_cuda}")

    target_confidence = {}
    locked_targets = {}  
    
    smoothed_boxes = {}
    alpha = 0.2  
    locked_target_position = False
    nearest_target_start_time = None
    saved_target_position = None  
    saved_target_label = None
    network_msg = None
    planets_gotten = 0
    targets = ["Earth", "Rose", "Saturn", "Black Hole"]


    while True:
        if processing:
            await asyncio.sleep(0.0001)  
            continue

        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.002)
            continue


        try:
            processing = True  
            start_time = time.time()  

            
            if use_cuda:
                frame_gpu = cv2.cuda_GpuMat()
                frame_gpu.upload(frame)
                frame = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB).download()

            results = model.predict(
                frame, verbose=False, device=device, conf=0.2, iou=0.4, agnostic_nms=True
            )

            latency = time.time() - start_time  
            fps = 1 / latency if latency > 0 else 0  

            if results is None or len(results) == 0 or results[0].boxes is None:
                print("🚨 No detections found! Skipping frame...")
                processing = False
                continue  

            boxes = results[0].boxes
            labels = results[0].names

            network_msg = "S"  
            annotated_frame = frame.copy()
            bottom_object, top_object, center_object = None, None, None
            target_centers = []

            for i, box in enumerate(boxes):
                label = labels[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                color = (0, 255, 0) if label in targets else (255, 255, 255)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if label == "Bottom":
                    bottom_object = box_center
                elif label == "Top":
                    top_object = box_center
                elif label == "Center":
                    center_object = box_center
                elif label in targets:
                    target_centers.append((box_center, label))
                              
                
            if bottom_object and top_object:
                cv2.line(annotated_frame, bottom_object, top_object, (255, 0, 0), 2)
            dx = top_object[0] - bottom_object[0]
            dy = top_object[1] - bottom_object[1]
            
        
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            
            nearest_target = None
            nearest_target_label = None
            min_distance = float('inf')
            focused = False
       
            
            for center, label in target_centers:
                distance = np.linalg.norm(np.array(center) - np.array(center_object))
                # distance = np.linalg.norm(np.array(center) - np.array(center_object))
                cv2.putText(annotated_frame, f"Dist: {distance:.2f}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if distance < min_distance:
                    min_distance = distance
                    # print(min_distance)
                    nearest_target = center
                    nearest_target_label = label
                    if nearest_target_label:
                        if nearest_target_start_time is None:
                            nearest_target_start_time = time.time()  
                        elif time.time() - nearest_target_start_time >= 3:
                            if saved_target_position is None:
                                saved_target_position = nearest_target  
                                saved_target_label = nearest_target_label
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    else:
                        nearest_target_start_time = None
                        saved_target_position = None
                        saved_target_label = None
                    
                    if saved_target_position is not None:
                        cv2.circle(annotated_frame, saved_target_position, 5, (0, 0, 255), -1)
                        
                        cv2.line(annotated_frame, center_object, saved_target_position, (0, 0, 255), 2)
                        
                        distance_to_saved_target = np.linalg.norm(np.array(center_object) - np.array(saved_target_position))
                        cv2.putText(annotated_frame, f"Distance: {distance_to_saved_target:.2f}", 
                                    (annotated_frame.shape[1] // 2 - 100, annotated_frame.shape[0] - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 1)
                        
                        cv2.putText(annotated_frame, f"Nearest: {saved_target_label}", (10, 460),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
                        
                        
                        target_dx = saved_target_position[0] - bottom_object[0]
                        target_dy = saved_target_position[1] - bottom_object[1]
                        target_angle_rad = np.arctan2(target_dy, target_dx)
                        target_angle_deg = np.degrees(target_angle_rad)

                        angle_diff = target_angle_deg - angle_deg
                        if angle_diff > 180:
                            angle_diff -= 360
                        elif angle_diff < -180:
                            angle_diff += 360
                            
                            
                            
                            
                            
                            
                        
                        
                        cv2.putText(annotated_frame, f"Angle: {angle_diff:.2f}°", (annotated_frame.shape[1] - 200, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        if distance_to_saved_target > 50:
                            if abs(angle_diff) <= 10:
                                network_msg = "F"     
                                focused = True
                                                
                            
                            elif abs(angle_diff) > 10 & focused == False:
                                if angle_diff > 0:
                                    # network_msg = "R"
                                    if time.time() % 0.5 < 0.25:
                                        network_msg = "R"
                                    else:
                                        network_msg = "S"
                                    
                                else:
                                    # network_msg = "L"
                                    if time.time() % 0.5 < 0.25:
                                        network_msg = "L"
                                    else:
                                        network_msg = "S"
                        
                        else:
                            network_msg = "S"
                            if nearest_target_start_time is None:
                                nearest_target_start_time = time.time()
                            elif time.time() - nearest_target_start_time >= 3:
                                saved_target_position = None
                                nearest_target_start_time = None
                                # planets_gotten += 1
                                
                                print(f"target collected: {saved_target_label}")
                                targets.remove(saved_target_label)  
                                
                                saved_target_label = None  

                                # planets_gotten += 1  # Increment the counter
                                print(f"planets collected: {planets_gotten}")

                                break  
                            
                            else:
                                remaining_time = 3 - (time.time() - nearest_target_start_time)
                                cv2.putText(annotated_frame, f"Timer: {remaining_time:.1f}s", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            focused = False
                            
                        
                    else:
                        network_msg = "S"
                        focused = False
         
         
         
            
            cv2.putText(annotated_frame, f"Latency: {latency:.4f}s ({fps:.2f} FPS)", 
                        (annotated_frame.shape[1] - 250, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.putText(annotated_frame, f"Command: {network_msg}", 
                        (annotated_frame.shape[1] - 200, annotated_frame.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 1)

            cv2.imshow("YOLO GPU Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            previous_msg = await send_msg_if_not_previous(websocket, previous_msg, network_msg)



        except Exception as e:
            print(f"Error: {e}")

        finally:
            processing = False  




async def main():
    async with websockets.connect(URI) as websocket:
        print("Connected to WebSocket")
        asyncio.create_task(process_yolo(websocket))
        while True:
            await asyncio.sleep(0.5)  

threading.Thread(target=cap.read, daemon=True).start()
asyncio.run(main())

cap.release()
cv2.destroyAllWindows()
