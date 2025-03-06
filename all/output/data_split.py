import os
import shutil
import random

# Define paths
source_images = "images"
source_labels = "labels"
dest_root = "data"
subsets = {"train": 0.7, "val": 0.15, "test": 0.15}

# Ensure destination directories exist
for subset in subsets:
    os.makedirs(os.path.join(dest_root, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, subset, "labels"), exist_ok=True)

# Get all image filenames
image_files = sorted([f for f in os.listdir(source_images) if f.endswith(".jpg")])
random.shuffle(image_files)

total_images = len(image_files)
split_indices = {
    "train": int(total_images * subsets["train"]),
    "val": int(total_images * (subsets["train"] + subsets["val"])),
}

# Distribute files
for i, img_file in enumerate(image_files):
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(source_labels, label_file)

    subset = (
        "train"
        if i < split_indices["train"]
        else "val" if i < split_indices["val"] else "test"
    )

    shutil.copy(
        os.path.join(source_images, img_file),
        os.path.join(dest_root, subset, "images", img_file),
    )

    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(dest_root, subset, "labels", label_file))

# Copy classes.txt to each label folder
for subset in subsets:
    shutil.copy(
        os.path.join(source_labels, "classes.txt"),
        os.path.join(dest_root, subset, "labels", "classes.txt"),
    )

print("Dataset successfully split!")
