from ultralytics import YOLO
try:
    model = YOLO("overheadbest-v5.pt")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
