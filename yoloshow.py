import streamlit as st
import os
import glob
import cv2
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("colab/best.pt")

# Define image folder
image_folder = "all/images/"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Streamlit app
st.title("YOLO Object Detection Viewer")
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

selected_image = st.selectbox("Select an image:", image_paths)

if selected_image:
    results = model(
        selected_image, conf=conf_threshold
    )  # Run inference with confidence threshold
    for result in results:
        image = Image.open(selected_image)
        result_image = result.plot()  # Get annotated image
        st.image(result_image, caption="Detected Objects", use_column_width=True)
