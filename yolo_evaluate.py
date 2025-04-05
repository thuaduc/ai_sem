import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob


def load_yolo_labels(label_path: str):
    """
    Load YOLO format labels from a text file.

    Args:
        label_path (str): Path to the YOLO format label file

    Returns:
        list: List of annotations [class_id, x_center, y_center, width, height]
    """
    if not os.path.exists(label_path):
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
    # Track true positives and false positives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Match predictions to ground truth
    matched_gt = set()
    for pred in predictions:
        # For precision calculation, we only care about the box coordinates
        pred_box = pred[1:5]

        # Find best matching ground truth
        best_match = None
        best_iou = 0
        for i, gt in enumerate(ground_truth):
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
    false_negatives = len(ground_truth) - len(matched_gt)

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
        "gt_count": len(ground_truth),
        "pred_count": len(predictions),
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
        class_id, x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(gt_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Remove class label text

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
        # Remove class and confidence text

    return image, ground_truth_boxes, predictions, gt_image, pred_image


def get_label_object_count(label_path):
    """
    Count the number of objects (lines) in a label file

    Args:
        label_path (str): Path to the YOLO format label file

    Returns:
        int: Number of objects in the label file
    """
    if not os.path.exists(label_path):
        return 0

    with open(label_path, "r") as f:
        return len(f.readlines())


def process_and_save_visualization(
    model,
    image_path,
    label_path,
    confidence_threshold=0.5,
    iou_threshold=0.5,
    output_img_path=None,
    output_txt_path=None,
):
    """
    Process a single image, save visualization and metrics to files

    Args:
        model: Loaded YOLO model
        image_path (str): Path to the input image
        label_path (str): Path to the ground truth label file
        confidence_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for evaluation
        output_img_path (str): Path to save visualization
        output_txt_path (str): Path to save metrics
    """
    # Get object count from label file
    object_count = get_label_object_count(label_path)

    # Process the image with labels
    _, ground_truth, predictions, gt_image, pred_image = process_single_image(
        model, image_path, label_path, confidence_threshold
    )

    # Calculate metrics
    metrics = evaluate_precision(ground_truth, predictions, iou_threshold)

    # Add object count
    metrics["object_count"] = object_count

    # Generate metrics text
    metrics_text = (
        f"Image: {os.path.basename(image_path)}\n"
        f"Object Count: {object_count}\n"
        f"Ground Truth Boxes: {metrics['gt_count']}\n"
        f"Predicted Boxes: {metrics['pred_count']}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall: {metrics['recall']:.4f}\n"
        f"F1: {metrics['f1_score']:.4f}\n"
    )

    # Save metrics to text file if path provided
    if output_txt_path:
        with open(output_txt_path, "w") as f:
            f.write(metrics_text)

    # Set up figure for visualization
    if output_img_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Display images
        ax1.imshow(gt_image)
        ax1.axis("off")

        ax2.imshow(pred_image)
        ax2.axis("off")

        # Add metrics text to figure
        # plt.figtext(
        #     0.5,
        #     0.01,
        #     metrics_text,
        #     ha="center",
        #     fontsize=14,
        #     bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        # )

        plt.tight_layout(rect=[0, 0.08, 1, 1])  # Adjust layout for text
        plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory

    # Print metrics to console for monitoring progress
    # print(
    #     f"Processed {os.path.basename(image_path)}: Obj={object_count}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
    # )

    return metrics


def batch_process_dataset(
    model, img_dir, label_dir, output_dir, confidence_threshold=0.5, iou_threshold=0.5
):
    """
    Process all images in the dataset and save visualizations and metrics

    Args:
        model: Loaded YOLO model
        img_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        output_dir (str): Directory to save outputs
        confidence_threshold (float): Confidence threshold for predictions
        iou_threshold (float): IoU threshold for evaluation
    """
    # Create output directories
    vis_dir = os.path.join(output_dir, "visualizations")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Get all image files
    image_files = (
        glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.jpeg"))
        + glob.glob(os.path.join(img_dir, "*.png"))
    )

    # Summary statistics
    all_metrics = []
    total_images = len(image_files)
    processed_images = 0
    total_objects = 0

    print(f"Found {total_images} images to process")

    # Process each image
    for image_path in image_files:
        # Get corresponding label path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        # Skip if label doesn't exist
        if not os.path.exists(label_path):
            print(f"WARNING: No label found for {base_name}, skipping")
            continue

        # Define output paths
        vis_path = os.path.join(vis_dir, f"{base_name}_vis.png")
        metrics_path = os.path.join(metrics_dir, f"{base_name}_metrics.txt")

        # Process image and save outputs
        metrics = process_and_save_visualization(
            model,
            image_path,
            label_path,
            confidence_threshold,
            iou_threshold,
            vis_path,
            metrics_path,
        )

        # Store metrics for summary
        metrics["image"] = base_name
        all_metrics.append(metrics)

        # Update totals
        total_objects += metrics["object_count"]

        # Update progress
        processed_images += 1
        # print(f"Progress: {processed_images}/{total_images} images processed")

    # Calculate and save overall metrics
    if all_metrics:
        avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m["f1_score"] for m in all_metrics) / len(all_metrics)
        total_tp = sum(m["true_positives"] for m in all_metrics)
        total_fp = sum(m["false_positives"] for m in all_metrics)
        total_fn = sum(m["false_negatives"] for m in all_metrics)
        total_gt = sum(m["gt_count"] for m in all_metrics)
        total_pred = sum(m["pred_count"] for m in all_metrics)

        summary_text = (
            f"=== EVALUATION SUMMARY ===\n"
            f"Total images processed: {len(all_metrics)}\n"
            f"Total objects in labels: {total_objects}\n"
            f"Total ground truth boxes: {total_gt}\n"
            f"Total predicted boxes: {total_pred}\n"
            f"Average Precision: {avg_precision:.4f}\n"
            f"Average Recall: {avg_recall:.4f}\n"
            f"Average F1 Score: {avg_f1:.4f}\n"
            f"Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}\n"
            f"Confidence_threshold: {confidence_threshold}\n"
            f"==========================="
        )

        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(summary_text)

    # print(f"Processing complete. Results saved to {output_dir}")


def main():
    # Configuration
    MODEL_PATH = "colab/runs_30_03_2025_structured/detect/train2/weights/best.pt"
    IMG_DIR = "datasets/test/images"
    LABEL_DIR = "datasets/test/labels"
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.1
    OUTPUT_DIR = "output_evaluations"

    try:
        # Load the model
        print(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)

        # Batch process all images
        print(f"Starting batch processing of images in {IMG_DIR}")
        batch_process_dataset(
            model, IMG_DIR, LABEL_DIR, OUTPUT_DIR, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
        )

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
