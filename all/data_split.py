import os
import shutil
import random
import time

# Seed randomness for different splits each run
random.seed(time.time())

# Define paths
source_images = "output/images"
source_labels = "output/labels"
dest_root = "datasets"
subsets = {"train": 0.85, "val": 0.15}

# Ensure destination directories exist
for subset in subsets:
    os.makedirs(os.path.join(dest_root, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, subset, "labels"), exist_ok=True)

# Get all image filenames, but only keep ones that have a corresponding label
image_files = [
    f
    for f in os.listdir(source_images)
    if f.endswith(".jpg")
    and os.path.exists(os.path.join(source_labels, os.path.splitext(f)[0] + ".txt"))
]

# Shuffle the filtered list
random.shuffle(image_files)

# Split dataset randomly
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
        shutil.copy(
            os.path.join(source_labels, label_file),
            os.path.join(dest_root, subset, "labels", label_file),
        )


# Copy files to respective folders
copy_files(train_files, "train")
copy_files(val_files, "val")

print("Dataset successfully split! Only images with corresponding labels are included.")
