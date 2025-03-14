import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# S = grids
# B = number of predicted bounding boxes
# C = number of classes
class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=1, C=4, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(self.image_dir))
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        image = pil_to_tensor(Image.open(img_path).convert("RGB"))

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(
                        float, line.split()
                    )
                    class_id = int(class_id)
                    boxes.append(
                        [
                            class_id,
                            x_center,
                            y_center,
                            width,
                            height,
                        ]
                    )
        else:
            boxes.append([-1, 0, 0, 0, 0])

        labels_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            if box[0] != -1:
                class_id, x_center, y_center, width, height = box
                i, j = int(self.S * y_center), int(self.S * x_center)
                x_cell, y_cell = self.S * x_center - j, self.S * y_center - i
                width_cell, height_celll = width * self.S, height * self.S

                if labels_matrix[i, j, 4] == 0:
                    labels_matrix[i, j, 4] = 1
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_celll]
                    )
                    labels_matrix[i, j, 5:9] = box_coordinates
                    labels_matrix[class_id] = 1
                    print(i, j)

        return image, labels_matrix


def plot_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    img_width, img_height = image.size

    for box in boxes:

        for i in range(7):
            for j in range(7):
                if box[i, j, 4] == 1:
                    x_center, y_center, width, height = box[0, 2, 5:]
                    x_min = x_center - width / 2
                    y_min = (1024 / 7) + y_center - (height / 2)

                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width=width,
                        height=height,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x_min,
                        y_min - 5,
                        f"Class: ",
                        color="red",
                        fontsize=12,
                        weight="bold",
                    )

    plt.show()


if __name__ == "__main__":
    # Create Datasets with transformations
    train_dataset = YOLODataset("data/example/images", "data/example/labels")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example Usage
    for images, labels_matrix in train_loader:
        print(images.shape, labels_matrix.shape)

        image = Image.open("data/example/images/347.jpg").convert("RGB")
        plot_image_with_boxes(image, labels_matrix)

        break
