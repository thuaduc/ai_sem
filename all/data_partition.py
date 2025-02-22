import os
import cv2
import numpy as np

# Define paths
input_image_dir = "images"
input_label_dir = "labels"
output_image_dir = "output/images"
output_label_dir = "output/labels"

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Image parameters
orig_width, orig_height = 2080, 1544  # Original size of images
crop_size = 512  # Sliding window size
step_size = 256  # Overlapping step


# Helper function to adjust labels
def adjust_yolo_labels(labels, crop_x, crop_y, crop_size):
    new_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = map(float, label.split())

        # Convert from normalized coordinates to absolute pixel values in original image
        abs_x_center = x_center * orig_width
        abs_y_center = y_center * orig_height
        abs_width = width * orig_width
        abs_height = height * orig_height

        # Check if the bounding box is within the crop window
        if (
            crop_x <= abs_x_center <= crop_x + crop_size
            and crop_y <= abs_y_center <= crop_y + crop_size
        ):
            # Adjust the bounding box relative to the crop
            new_x_center = (abs_x_center - crop_x) / crop_size
            new_y_center = (abs_y_center - crop_y) / crop_size

            new_width = abs_width / crop_size
            new_height = abs_height / crop_size

            # box overfloat to the left
            if new_x_center < new_width:
                # print(f"old x center {new_x_center} old width {new_width}")
                new_x_center = (new_x_center + (new_width / 2)) / 2
                new_width = new_x_center * 2
                # print(f"new x center {new_x_center} new width {new_width}\n")

            if new_y_center < new_height:
                new_y_center = (new_y_center + (new_height / 2)) / 2
                new_height = new_y_center * 2

            if (new_x_center + (new_width / 2)) > 1:
                print(f"old x center {new_x_center} old width {new_width}")
                new_x_center -= (new_x_center + (new_width / 2) - 1) / 2
                new_width = (1 - new_x_center) * 2
                print(f"new x center {new_x_center} new width {new_width}\n")

            if new_y_center + new_height / 2 > 1:
                new_y_center -= (new_y_center + (new_height / 2) - 1) / 2
                new_height = (1 - new_y_center) * 2

            # Add to new labels
            new_labels.append(
                f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"
            )

    return new_labels


# Process images in alphabetical order
image_files = sorted([f for f in os.listdir(input_image_dir) if f.endswith(".jpg")])

counter = 1  # Counter for naming new images

for image_file in image_files:
    img_path = os.path.join(input_image_dir, image_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image: {image_file}")
        continue

    label_file = os.path.join(input_label_dir, image_file.replace(".jpg", ".txt"))
    labels = []

    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            labels = f.read().strip().split("\n")

    # Sliding window
    for y in range(0, orig_height - crop_size + 1, step_size):
        for x in range(0, orig_width - crop_size + 1, step_size):
            cropped_img = img[y : y + crop_size, x : x + crop_size]
            new_image_name = f"{counter}.jpg"
            new_label_name = f"{counter}.txt"

            # Adjust labels
            new_labels = adjust_yolo_labels(labels, x, y, crop_size)

            # Save cropped image and labels
            cv2.imwrite(os.path.join(output_image_dir, new_image_name), cropped_img)

            if new_labels:
                with open(os.path.join(output_label_dir, new_label_name), "w") as f:
                    f.write("\n".join(new_labels))

            counter += 1


# Copy classes.txt to output directory
classes_file = os.path.join(input_label_dir, "classes.txt")
if os.path.exists(classes_file):
    os.system(f"cp {classes_file} {output_label_dir}/classes.txt")

print(
    "Processing complete. Cropped images and updated labels saved in 'output/' folder."
)
