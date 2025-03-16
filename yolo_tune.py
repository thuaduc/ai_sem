from ultralytics import YOLO
import torch

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

# Initialize the YOLO model
model = YOLO("yolo11n.pt")
model.to(device)

# Define search space
search_space = {
    # Training hyperparameters
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),  # Warmup epochs
    "warmup_momentum": (0.0, 0.95),  # Warmup momentum
    # Loss function weights
    "box": (0.02, 0.2),  # Box loss weight
    "cls": (0.2, 4.0),  # Classification loss weight
}

# Tune hyperparameters on COCO8 for 5 epochs (adjust as needed)
result = model.tune(
    data="config.yaml",
    epochs=10,
    iterations=100,
    optimizer="Adam",
    space=search_space,
    plots=False,
    save=False,
    val=False,
)

print(result)
