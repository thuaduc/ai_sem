import cv2
import os
import tkinter as tk
from tkinter import filedialog, Checkbutton, BooleanVar
from PIL import Image, ImageTk
import numpy as np

dataset_path = "output"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

# Get sorted image list
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[0]),
)

# Read class names
classes_file = os.path.join(label_folder, "classes.txt")
if os.path.exists(classes_file):
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = []

# Assign fixed colors per class
class_colors = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (0, 0, 255),  # Blue
    3: (255, 255, 0),  # Yellow
}

current_index = 1

# Initialize Tkinter
root = tk.Tk()
root.title("YOLO Image Viewer")
draw_boxes = BooleanVar(value=True)

# Create UI elements
canvas_width, canvas_height = 600, 600
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Image number label
image_number_label = tk.Label(
    root, text=f"Image: {current_index + 1}/{len(image_files)}"
)
image_number_label.pack()


def show_image():
    global current_index

    if current_index < 0:
        current_index = 0
    if current_index >= len(image_files):
        current_index = len(image_files) - 1

    image_path = os.path.join(image_folder, image_files[current_index])
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    label_file = os.path.join(
        label_folder, image_files[current_index].replace(".jpg", ".txt")
    )

    if draw_boxes.get() and os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)
                class_id = int(class_id)
                if class_id not in class_colors:
                    continue

                color = class_colors[class_id]
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    str(class_id),
                    (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    # Convert OpenCV image to PIL format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Center the image on the canvas
    x_offset = (canvas_width - width) // 2
    y_offset = (canvas_height - height) // 2

    img_tk = ImageTk.PhotoImage(image)
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk

    # Update the image number label
    image_number_label.config(text=f"Image: {current_index}/{len(image_files)}")


def next_image():
    global current_index
    current_index += 1
    show_image()


def prev_image():
    global current_index
    current_index -= 1
    show_image()


def toggle_boxes():
    show_image()


# Button frame for centering buttons
button_frame = tk.Frame(root)
button_frame.pack()

btn_prev = tk.Button(button_frame, text="Previous", command=prev_image)
btn_prev.pack(side=tk.LEFT, padx=10, pady=10)

btn_next = tk.Button(button_frame, text="Next", command=next_image)
btn_next.pack(side=tk.LEFT, padx=10, pady=10)

chk_boxes = Checkbutton(
    button_frame, text="Show Bounding Boxes", variable=draw_boxes, command=toggle_boxes
)
chk_boxes.pack(side=tk.LEFT, padx=10, pady=10)

show_image()
root.mainloop()
