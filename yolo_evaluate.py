import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from matplotlib.widgets import Button


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


def process_single_image(model, image_path, label_path, confidence_threshold=0.5):
    """
    Process a single image and return ground truth and predictions

    Args:
        model: Loaded YOLO model
        image_path (str): Path to the input image
        label_path (str): Path to the ground truth label file
        confidence_threshold (float): Confidence threshold for predictions

    Returns:
        tuple: (image, ground_truth_boxes, predictions, gt_image, pred_image)
    """
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

    return image, ground_truth_boxes, predictions, gt_image, pred_image


def calculate_average_metrics(
    model, images_dir, labels_dir, confidence_threshold=0.5, iou_threshold=0.5
):
    """
    Calculate average metrics across all test images

    Args:
        model: Loaded YOLO model
        images_dir (str): Directory containing test images
        labels_dir (str): Directory containing test labels
        confidence_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for evaluation

    Returns:
        tuple: (avg_metrics, all_metrics) - Average metrics and per-image metrics
    """
    image_files = [
        f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_files:
        print("No images found in the specified directory.")
        return None, []

    all_ground_truth = []
    all_predictions = []
    per_image_metrics = []

    # Process each image
    for img_file in image_files:
        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_file}, skipping.")
            continue

        _, ground_truth, predictions, _, _ = process_single_image(
            model, image_path, label_path, confidence_threshold
        )

        # Calculate metrics for this image
        img_metrics = evaluate_precision([ground_truth], [predictions], iou_threshold)
        per_image_metrics.append((img_file, img_metrics))

        # Store for overall metrics calculation
        all_ground_truth.append(ground_truth)
        all_predictions.append(predictions)

    # Calculate overall metrics
    overall_metrics = evaluate_precision(
        all_ground_truth, all_predictions, iou_threshold
    )

    # Calculate average metrics
    avg_metrics = {
        "precision": np.mean([m[1]["precision"] for m in per_image_metrics]),
        "recall": np.mean([m[1]["recall"] for m in per_image_metrics]),
        "f1_score": np.mean([m[1]["f1_score"] for m in per_image_metrics]),
        "true_positives": sum(m[1]["true_positives"] for m in per_image_metrics),
        "false_positives": sum(m[1]["false_positives"] for m in per_image_metrics),
        "false_negatives": sum(m[1]["false_negatives"] for m in per_image_metrics),
    }

    return avg_metrics, overall_metrics, per_image_metrics


def main():
    # Configuration
    MODEL_PATH = "colab/runs_28_03_2025_structured/detect/train2/weights/best.pt"
    IMAGES_DIR = "datasets/test/images"
    LABELS_DIR = "datasets/test/labels"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    try:
        # Load the model once
        model = YOLO(MODEL_PATH)

        # Get all image files
        image_files = [
            f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        if not image_files:
            print("No images found in the specified directory.")
            return

        # Calculate average metrics at the beginning
        print("Calculating average metrics across all test images...")
        avg_metrics, overall_metrics, per_image_metrics = calculate_average_metrics(
            model, IMAGES_DIR, LABELS_DIR, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

        # Display average metrics
        print("\nAverage Metrics Across All Images:")
        for metric, value in avg_metrics.items():
            if metric in ["true_positives", "false_positives", "false_negatives"]:
                print(f"{metric.replace('_', ' ').title()}: {int(value)}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        print("\nOverall Metrics (Calculated on all detections at once):")
        for metric, value in overall_metrics.items():
            if metric in ["true_positives", "false_positives", "false_negatives"]:
                print(f"{metric.replace('_', ' ').title()}: {int(value)}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

        # Create a class for handling image navigation
        class ImageNavigator:
            def __init__(
                self,
                model,
                image_files,
                images_dir,
                labels_dir,
                confidence_threshold=0.5,
                iou_threshold=0.5,
            ):
                self.model = model
                self.image_files = image_files
                self.images_dir = images_dir
                self.labels_dir = labels_dir
                self.confidence_threshold = confidence_threshold
                self.iou_threshold = iou_threshold
                self.current_index = 0

                # Create output directory for visualizations
                self.output_dir = "output_visualizations"
                os.makedirs(self.output_dir, exist_ok=True)

                # Create the figure
                self.fig = plt.figure(figsize=(16, 10))

                # Set up the layout
                self.ax1 = plt.subplot2grid((5, 2), (0, 0), rowspan=4)
                self.ax2 = plt.subplot2grid((5, 2), (0, 1), rowspan=4)
                self.ax_info = plt.subplot2grid((5, 2), (4, 0), colspan=2)

                # Disable axis for info area
                self.ax_info.axis("off")

                # Add buttons
                self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
                self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
                self.ax_save = plt.axes([0.75, 0.05, 0.1, 0.075])

                self.btn_prev = Button(self.ax_prev, "Previous")
                self.btn_next = Button(self.ax_next, "Next")
                self.btn_save = Button(self.ax_save, "Save")

                self.btn_prev.on_clicked(self.on_prev)
                self.btn_next.on_clicked(self.on_next)
                self.btn_save.on_clicked(self.on_save)

                # Initialize with the first image
                self.update_display()

                # Set window title
                plt.suptitle("Object Detection Evaluation", fontsize=16)

            def update_display(self):
                # Clear axes
                self.ax1.clear()
                self.ax2.clear()

                # Get current image
                img_file = self.image_files[self.current_index]
                image_path = os.path.join(self.images_dir, img_file)
                label_path = os.path.join(
                    self.labels_dir, os.path.splitext(img_file)[0] + ".txt"
                )

                # Process the image
                _, ground_truth, predictions, gt_image, pred_image = (
                    process_single_image(
                        self.model, image_path, label_path, self.confidence_threshold
                    )
                )

                # Calculate metrics
                metrics = evaluate_precision(
                    [ground_truth], [predictions], self.iou_threshold
                )

                # Display images
                self.ax1.imshow(gt_image)
                self.ax1.set_title("Ground Truth")
                self.ax1.axis("off")

                self.ax2.imshow(pred_image)
                self.ax2.set_title("Model Predictions")
                self.ax2.axis("off")

                # Display metrics and image info
                metrics_text = f"Image: {img_file} ({self.current_index+1}/{len(self.image_files)})\n"
                metrics_text += f"Precision: {metrics['precision']:.4f}, "
                metrics_text += f"Recall: {metrics['recall']:.4f}, "
                metrics_text += f"F1 Score: {metrics['f1_score']:.4f}\n"
                metrics_text += f"True Positives: {metrics['true_positives']}, "
                metrics_text += f"False Positives: {metrics['false_positives']}, "
                metrics_text += f"False Negatives: {metrics['false_negatives']}"

                # Display metrics info
                self.ax_info.clear()
                self.ax_info.text(
                    0.5,
                    0.5,
                    metrics_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                )
                self.ax_info.axis("off")

                plt.draw()

            def on_prev(self, event):
                self.current_index = (self.current_index - 1) % len(self.image_files)
                self.update_display()

            def on_next(self, event):
                self.current_index = (self.current_index + 1) % len(self.image_files)
                self.update_display()

            def on_save(self, event):
                img_file = self.image_files[self.current_index]
                output_path = os.path.join(
                    self.output_dir,
                    f"visualization_{os.path.splitext(img_file)[0]}.png",
                )

                # Save current figure
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Visualization saved to {output_path}")

        # Create navigator and show
        navigator = ImageNavigator(
            model,
            image_files,
            IMAGES_DIR,
            LABELS_DIR,
            CONFIDENCE_THRESHOLD,
            IOU_THRESHOLD,
        )

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
