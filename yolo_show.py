import streamlit as st
import os
import glob
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2

# Load model
model = YOLO("colab/runs_21_03_2025/detect/train6/weights/best.pt")

# Define image and label folders
image_folder = "all/test/images"
label_folder = "all/test/labels"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Streamlit UI
st.title("YOLO Object Detection Viewer")
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
selected_image = st.selectbox("Select an image:", image_paths)


def draw_yolo_boxes(image_path, label_path):
    """Draw YOLO format bounding boxes on an image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    img_width, img_height = image.size

    # Read label file
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()

        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                # Convert YOLO format to pixel coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # Draw rectangle
                outline = "red" if class_id == 1 else "yellow"
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min), f"Class {int(class_id)}", fill="red")

    return image


if selected_image:
    # Run YOLO model inference
    results = model(selected_image, conf=conf_threshold)

    # Display detected objects
    for result in results:
        result_image = result.plot()  # Get annotated image
        st.image(result_image, caption="YOLO Detected Objects", use_column_width=True)

    # Get ground truth label path
    image_name = os.path.basename(selected_image).replace(".jpg", ".txt")
    label_path = os.path.join(label_folder, image_name)

    # Draw ground truth boxes
    ground_truth_image = draw_yolo_boxes(selected_image, label_path)

    # Display ground truth image
    st.image(ground_truth_image, caption="Ground Truth Labels", use_column_width=True)
