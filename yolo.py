import torch
from ultralytics import YOLO

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("CUDA is available. Running on GPU.")

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model.to(device)

    # Train the model
    results = model.train(
        data="config.yaml", epochs=100, imgsz=640, device=device, batch=16, patience=10
    )
else:
    print("CUDA is not available. Exiting.")
