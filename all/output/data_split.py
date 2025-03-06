import os
import shutil
import random

# Define paths
source_images = "images"
source_labels = "labels"
dest_root = "data"
subsets = {"train": 0.8, "val": 0.2}

# Ensure destination directories exist
for subset in subsets:
    os.makedirs(os.path.join(dest_root, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, subset, "labels"), exist_ok=True)

# Get all image filenames and shuffle
image_files = [f for f in os.listdir(source_images) if f.endswith(".jpg")]
random.shuffle(image_files)

# Split dataset
split_index = int(len(image_files) * subsets["train"])
train_files = image_files[:split_index]
val_files = image_files[split_index:]


# Function to copy files
def copy_files(file_list, subset):
    for img_file in file_list:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.copy(
            os.path.join(source_images, img_file),
            os.path.join(dest_root, subset, "images", img_file),
        )
        label_path = os.path.join(source_labels, label_file)
        if os.path.exists(label_path):
            shutil.copy(
                label_path, os.path.join(dest_root, subset, "labels", label_file)
            )


# Copy files to respective folders
copy_files(train_files, "train")
copy_files(val_files, "val")

print("Dataset successfully split!")
