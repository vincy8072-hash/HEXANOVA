from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="drishti_dataset/drishti.yaml",
    epochs=5,
    imgsz=416,
    batch=4,
    workers=0
)

print("Training completed successfully!")