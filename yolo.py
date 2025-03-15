import torch
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Check if CUDA is available
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

"""

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

"""

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model
model.to(device)

# Train the model
results = model.train(
    data="config.yaml",
    epochs=10,
    imgsz=640,
    device=device,
    batch=16,
    patience=10,
    dropout=0.3,
    lr0=0.001,
    lrf=0.1,
)
