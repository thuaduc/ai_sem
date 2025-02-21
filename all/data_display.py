import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Paths to the output dataset
IMAGE_DIR = "output/images"
LABEL_DIR = "output/labels"
IMG_SIZE = 512  # Display image size


class ImageViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Viewer")
        self.root.geometry("600x600")

        # Load image list
        self.image_files = sorted(
            [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
        )
        self.current_index = 0
        self.show_boxes = tk.BooleanVar(value=True)  # Toggle bounding box display

        if not self.image_files:
            print("No images found in directory:", IMAGE_DIR)
            return

        # Create UI Elements
        self.canvas = tk.Canvas(root, width=IMG_SIZE, height=IMG_SIZE)
        self.canvas.pack(pady=10)

        self.checkbox = ttk.Checkbutton(
            root,
            text="Show Bounding Boxes",
            variable=self.show_boxes,
            command=self.update_image,
        )
        self.checkbox.pack()

        # Navigation Buttons
        self.prev_button = ttk.Button(root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = ttk.Button(root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # Load First Image
        self.update_image()

    def load_image(self, image_path):
        """Loads an image and converts it to a Tkinter-compatible format."""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def get_label_color(self, class_idx):
        """Returns a unique color for each label."""
        # Use a predefined set of colors for each label (class_idx)
        color_map = {
            0: (255, 0, 0),  # Red
            1: (0, 255, 0),  # Green
            2: (0, 0, 255),  # Blue
            3: (255, 255, 0),  # Yellow
            4: (255, 0, 255),  # Magenta
            5: (0, 255, 255),  # Cyan
            # Add more colors as needed
        }

        # If class_idx is not in the color_map, return a default color
        return color_map.get(int(class_idx), (0, 0, 0))  # Default to black

    def adjust_bounding_boxes(self, img, label_path, patch_x, patch_y):
        """Adjusts bounding boxes for the 512x512 patches using sliding window partitioning."""
        if img is None:
            print(f"Invalid image passed to adjust_bounding_boxes.")
            return Image.new(
                "RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)
            )  # Return blank image

        if not os.path.exists(label_path):
            return Image.fromarray(img)  # No labels for this image

        img_array = img.copy()
        height, width, _ = img_array.shape
        boxes = open(label_path, "r").read().strip().split("\n")
        drawn_boxes = []  # Track boxes that have already been drawn

        for box in boxes:
            parts = box.split()
            if len(parts) != 5:
                continue  # Skip invalid labels
            class_idx, x, y, w, h = map(float, parts)
            print(parts)

            # Get the unique color for this label
            color = self.get_label_color(class_idx)

            # Convert YOLO format to pixel coordinates in the full image
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            # Check if the bounding box is within the patch
            if (
                x1 >= patch_x + IMG_SIZE
                or x2 <= patch_x
                or y1 >= patch_y + IMG_SIZE
                or y2 <= patch_y
            ):
                continue  # Skip bounding boxes outside the current patch

            # Adjust the bounding box to be relative to the top-left corner of the patch
            x1 = max(x1 - patch_x, 0)
            y1 = max(y1 - patch_y, 0)
            x2 = min(x2 - patch_x, IMG_SIZE)
            y2 = min(y2 - patch_y, IMG_SIZE)

            # Prevent overlap: If a box already exists, skip drawing it
            overlap = False
            for prev_x1, prev_y1, prev_x2, prev_y2 in drawn_boxes:
                # Check for overlap
                if not (
                    x2 <= prev_x1 or x1 >= prev_x2 or y2 <= prev_y1 or y1 >= prev_y2
                ):
                    overlap = True
                    break  # Skip if overlap is detected

            if not overlap:
                # Draw rectangle with the unique color
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                drawn_boxes.append((x1, y1, x2, y2))  # Add the box to drawn boxes

        return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

    def update_image(self):
        """Updates the displayed image with or without bounding boxes."""
        if not self.image_files:
            return

        img_name = self.image_files[self.current_index]
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

        img = self.load_image(img_path)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            return  # Stop if image is invalid

        # Define the top-left corner of the current patch
        patch_x = 0
        patch_y = 0

        if self.show_boxes.get():
            img_array = cv2.imread(img_path)
            if img_array is None:
                print(f"Error loading image for bounding boxes: {img_path}")
                return  # Prevents passing None

            img = self.adjust_bounding_boxes(img_array, label_path, patch_x, patch_y)

        img = img.resize((IMG_SIZE, IMG_SIZE))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def next_image(self):
        """Shows the next image."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_image()

    def prev_image(self):
        """Shows the previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
