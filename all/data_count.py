import os

# Paths to the directories
images_dir_train = "datasets/train/images"
labels_dir_train = "datasets/train/labels"
images_dir_val = "datasets/val/images"
labels_dir_val = "datasets/val/labels"


def count_labels_and_images(images_dir, labels_dir):
    class_counts = {0: 0, 1: 0, 2: 0}  # To count class 0, 1, and 2
    total_images = 0

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)

        # Only process .txt files
        if label_file.endswith(".txt"):
            # Read each label file
            with open(label_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(
                        parts[0]
                    )  # Class id is the first element in YOLO format
                    if class_id in class_counts:
                        class_counts[class_id] += 1

            # Count the corresponding image if the label file exists
            image_file = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                total_images += 1

    return class_counts, total_images


# Count for training and validation sets
train_class_counts, train_total_images = count_labels_and_images(
    images_dir_train, labels_dir_train
)
val_class_counts, val_total_images = count_labels_and_images(
    images_dir_val, labels_dir_val
)

# Display the results
print(f"Training set:")
print(
    f"Class 0: {train_class_counts[0]}, Class 1: {train_class_counts[1]}, Class 2: {train_class_counts[2]}"
)
print(f"Total images: {train_total_images}\n")

print(f"Validation set:")
print(
    f"Class 0: {val_class_counts[0]}, Class 1: {val_class_counts[1]}, Class 2: {val_class_counts[2]}"
)
print(f"Total images: {val_total_images}")
