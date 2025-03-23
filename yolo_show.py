import streamlit as st
import os
import glob
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2

# Load YOLO model
model = YOLO("colab/runs_22_03_2025/detect/train2/weights/best.pt")

# Define paths
image_folder = "datasets/test/images"
label_folder = "datasets/test/labels"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# Streamlit UI
st.title("YOLO Object Detection Viewer & Accuracy Evaluation")
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.05)
selected_image = st.selectbox("Select an image:", image_paths)


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union = box1_area + box2_area - intersection

    if union == 0:
        return 0
    return intersection / union


def evaluate_accuracy(image_path, label_path, results):
    """Evaluate detection accuracy using IoU."""
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Load ground truth labels
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    gt_boxes.append((x_min, y_min, x_max, y_max, int(class_id)))

    # Load predicted bounding boxes
    pred_boxes = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max, conf, class_id = box
            if conf >= conf_threshold:
                pred_boxes.append(
                    (
                        int(x_min),
                        int(y_min),
                        int(x_max),
                        int(y_max),
                        int(class_id),
                        conf,
                    )
                )

    # Match predictions with ground truth using IoU
    tp, fp, fn = 0, 0, len(gt_boxes)

    for pred_box in pred_boxes:
        px_min, py_min, px_max, py_max, p_class, p_conf = pred_box
        best_iou = 0
        best_match = None

        for gt_box in gt_boxes:
            gx_min, gy_min, gx_max, gy_max, g_class = gt_box
            iou = compute_iou(
                (px_min, py_min, px_max, py_max), (gx_min, gy_min, gx_max, gy_max)
            )

            if iou > best_iou and p_class == g_class:
                best_iou = iou
                best_match = gt_box

        if best_iou >= iou_threshold:
            tp += 1
            fn -= 1
            gt_boxes.remove(best_match)  # Remove matched ground truth box
        else:
            fp += 1  # False positive

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


if selected_image:
    # Run YOLO inference
    results = model(selected_image, conf=conf_threshold)

    print(results[0])

    # Display detected objects
    for result in results:
        result_image = result.plot()
        st.image(result_image, caption="YOLO Detected Objects", use_column_width=True)

    # Get ground truth label path
    image_name = os.path.basename(selected_image).replace(".jpg", ".txt")
    label_path = os.path.join(label_folder, image_name)

    # Compute accuracy metrics
    precision, recall = evaluate_accuracy(selected_image, label_path, results)

    # Display precision and recall
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    # Compute F1-score
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    st.write(f"**F1-score:** {f1_score:.2f}")
