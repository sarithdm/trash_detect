from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)

# Load trained model
#model = YOLO('runs/detect/train/weights/best.pt')

# Detect objects in a video
#results = model.predict(source='video.mp4', save=True)


"""
Extend for People and Vehicle Actions
To add people and vehicle actions, you can:

Augment the TACO dataset with other datasets (e.g., COCO for people and vehicles).
Merge the datasets, ensuring unique class IDs.
Annotate throwing actions using the same tool (e.g., LabelImg).
"""