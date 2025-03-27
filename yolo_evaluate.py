import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def load_yolo_labels(label_path: str):
    """
    Load YOLO format labels from a text file.

    Args:
        label_path (str): Path to the YOLO format label file

    Returns:
        list: List of annotations [class_id, x_center, y_center, width, height]
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found at {label_path}")
        return []

    with open(label_path, "r") as f:
        labels = []
        for line in f:
            parts = list(map(float, line.strip().split()))
            labels.append(parts)
    return labels


def convert_yolo_to_xyxy(image_size, yolo_bbox):
    """
    Convert YOLO format (normalized center x, center y, width, height)
    to XYXY format (x_min, y_min, x_max, y_max)

    Args:
        image_size (tuple): (width, height) of the image
        yolo_bbox (list): [class_id, x_center, y_center, width, height]

    Returns:
        tuple: (class_id, x_min, y_min, x_max, y_max)
    """
    img_width, img_height = image_size
    class_id, x_center, y_center, width, height = yolo_bbox

    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)

    return (int(class_id), x_min, y_min, x_max, y_max)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes

    Args:
        box1 (tuple): First bounding box (x_min, y_min, x_max, y_max)
        box2 (tuple): Second bounding box (x_min, y_min, x_max, y_max)

    Returns:
        float: IoU value
    """
    # Compute coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    return intersection_area / union_area if union_area > 0 else 0


def evaluate_precision(ground_truth, predictions, iou_threshold=0.5):
    """
    Compute precision, recall, and F1 score

    Args:
        ground_truth (list): Ground truth bounding boxes
        predictions (list): Predicted bounding boxes
        iou_threshold (float): IoU threshold for matching

    Returns:
        dict: Precision metrics
    """
    # Flatten ground truth and predictions
    all_ground_truth = [box for img_gt in ground_truth for box in img_gt]
    all_predictions = [box for img_pred in predictions for box in img_pred]

    # Sort predictions by confidence (if available)
    all_predictions.sort(key=lambda x: x[5] if len(x) > 5 else 1.0, reverse=True)

    # Track true positives and false positives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Match predictions to ground truth
    matched_gt = set()
    for pred in all_predictions:
        # For precision calculation, we only care about the box coordinates
        pred_box = pred[1:5]

        # Find best matching ground truth
        best_match = None
        best_iou = 0
        for i, gt in enumerate(all_ground_truth):
            # If class matches (first element is class)
            if gt[0] == pred[0]:
                iou = compute_iou(pred_box, gt[1:5])
                if iou > best_iou and iou >= iou_threshold:
                    best_match = i
                    best_iou = iou

        # If a match is found
        if best_match is not None and best_match not in matched_gt:
            true_positives += 1
            matched_gt.add(best_match)
        else:
            false_positives += 1

    # Count false negatives (ground truth not matched)
    false_negatives = len(all_ground_truth) - len(matched_gt)

    # Compute metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def visualize_predictions(model_path, image_path, label_path, confidence_threshold=0.5):
    """
    Visualize ground truth and model predictions side by side

    Args:
        model_path (str): Path to the trained YOLO model
        image_path (str): Path to the input image
        label_path (str): Path to the ground truth label file
        confidence_threshold (float): Confidence threshold for predictions

    Returns:
        tuple: Ground truth and prediction data for further analysis
    """
    # Validate input paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at {label_path}")

    # Load model
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image.shape

    # Process ground truth labels
    ground_truth_labels = load_yolo_labels(label_path)
    ground_truth_boxes = [
        convert_yolo_to_xyxy((img_width, img_height), label)
        for label in ground_truth_labels
    ]

    # Run inference
    results = model(image_path, conf=confidence_threshold)[0]

    # Prepare ground truth visualization
    gt_image = image.copy()
    for bbox in ground_truth_boxes:
        # Unpack bbox (class_id, x_min, y_min, x_max, y_max)
        _, x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(gt_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Prepare prediction visualization and data
    pred_image = image.copy()
    predictions = []
    for detection in results.boxes:
        # Convert to integer coordinates
        bbox = detection.xyxy[0]
        class_id = int(detection.cls[0])
        conf = float(detection.conf[0])
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Store prediction with class, coordinates, and confidence
        predictions.append((class_id, x_min, y_min, x_max, y_max, conf))

        cv2.rectangle(pred_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Create side-by-side plot
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Model Predictions")
    plt.imshow(pred_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return ground_truth_boxes, predictions


def main():
    # Configuration
    MODEL_PATH = "colab/runs_27_03_2025_structured/detect/train/weights/best.pt"
    IMAGES_DIR = "datasets/test/images"
    LABELS_DIR = "datasets/test/labels"

    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.1
    IOU_THRESHOLD = 0.5

    # Find first image and its corresponding label
    try:
        image_files = [
            f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        if not image_files:
            print("No images found in the specified directory.")
            return

        # Take the first image
        first_image = image_files[0]
        image_path = os.path.join(IMAGES_DIR, first_image)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(first_image)[0] + ".txt")

        # Visualize predictions and get ground truth and prediction data
        ground_truth, predictions = visualize_predictions(
            MODEL_PATH, image_path, label_path, CONFIDENCE_THRESHOLD
        )

        # Evaluate precision
        metrics = evaluate_precision(
            [ground_truth], [predictions], iou_threshold=IOU_THRESHOLD
        )

        # Print detailed metrics
        print("\nPrecision Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
