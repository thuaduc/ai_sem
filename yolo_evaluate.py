from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("colab/best2.pt")

# Run validation
metrics = model.val(data="config.yaml")

# Print results
print(metrics)
