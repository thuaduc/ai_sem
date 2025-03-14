from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(
    data="config.yaml", epochs=100, imgsz=640, device="cuda", batch=16, patience=10
)
