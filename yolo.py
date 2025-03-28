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

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model
model.to(device)

# Train the model
results = model.train(
    data="config.yaml",
    epochs=150,
    imgsz=640,
    device=device,
    batch=64,
    optimizer="AdamW",
    patience=15,
    dropout=0.5,
    lr0=0.00389,
    lrf=0.01148,
    momentum=0.60707,
    weight_decay=0.00054,
    warmup_epochs=3.27627,
    warmup_momentum=0.58502,
    box=0.19096,
    cls=0.46863,
)
