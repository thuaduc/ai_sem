import os
import shutil
import random
from collections import defaultdict


def load_labels(label_path):
    """Loads labels and groups images by class ID."""
    class_to_images = defaultdict(list)

    for label_file in os.listdir(label_path):
        if not label_file.endswith(".txt"):
            continue

        image_name = label_file.replace(".txt", ".jpg")
        image_path = os.path.join("output/images", image_name)
        label_file_path = os.path.join(label_path, label_file)

        if os.path.exists(image_path):
            with open(label_file_path, "r") as f:
                lines = f.readlines()
                if lines:
                    class_id = int(lines[0].split()[0])
                    class_to_images[class_id].append((image_path, label_file_path))

    return class_to_images


def split_and_move(class_to_images, train_ratio=0.9):
    """Splits the data and moves it into the appropriate directories."""
    base_dir = "datasets"
    train_img_dir = os.path.join(base_dir, "train/images")
    train_lbl_dir = os.path.join(base_dir, "train/labels")
    val_img_dir = os.path.join(base_dir, "val/images")
    val_lbl_dir = os.path.join(base_dir, "val/labels")

    # Create directories if they don't exist
    for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(folder, exist_ok=True)

    for class_id, images in class_to_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_set = images[:split_idx]
        val_set = images[split_idx:]

        for img_path, lbl_path in train_set:
            shutil.copy(img_path, train_img_dir)
            shutil.copy(lbl_path, train_lbl_dir)

        for img_path, lbl_path in val_set:
            shutil.copy(img_path, val_img_dir)
            shutil.copy(lbl_path, val_lbl_dir)


if __name__ == "__main__":
    label_folder = "output/labels"
    class_images = load_labels(label_folder)
    split_and_move(class_images)
