import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Custom Dataset for Object Detection
class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(
            self.label_dir, self.image_files[idx].replace(".jpg", ".txt")
        )

        # Load image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype("float32") / 255).permute(2, 0, 1)

        # Load labels (YOLO format)
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_label = int(parts[0])
                    if 0 <= class_label <= 3:  # Ensure only valid labels
                        class_labels.append(class_label)
                        bboxes.append(parts[1:])  # (x, y, w, h) normalized

        if self.transform:
            image = self.transform(image)

        return image, bboxes, class_labels


# DataLoader Setup
def get_dataloaders(batch_size=16):
    train_dataset = ObjectDetectionDataset("data/train/images", "data/train/labels")
    val_dataset = ObjectDetectionDataset("data/val/images", "data/val/labels")
    test_dataset = ObjectDetectionDataset("data/test/images", "data/test/labels")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


# Collate function to handle variable number of objects per image
def collate_fn(batch):
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes)
    labels = torch.stack(labels)

    return images, bboxes, labels


# Loss Functions
class ObjectDetectionLoss(nn.Module):
    def __init__(self):
        super(ObjectDetectionLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, pred_bboxes, pred_classes, target_bboxes, target_classes):
        # Flatten the tensors to handle batch processing
        pred_bboxes = pred_bboxes.view(-1, 4)
        target_bboxes = target_bboxes.view(-1, 4)
        pred_classes = pred_classes.view(-1, pred_classes.size(-1))
        target_classes = target_classes.view(-1)

        # Filter out padded elements (-1 for classes)
        valid_indices = (target_classes != -1).nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return torch.tensor(0.0, requires_grad=True)

        valid_pred_bboxes = pred_bboxes[valid_indices]
        valid_target_bboxes = target_bboxes[valid_indices]
        valid_pred_classes = pred_classes[valid_indices]
        valid_target_classes = target_classes[valid_indices]

        bbox_loss = self.bbox_loss(valid_pred_bboxes, valid_target_bboxes)
        class_loss = self.class_loss(valid_pred_classes, valid_target_classes)

        return bbox_loss + class_loss


# Model Definition
class CustomObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(CustomObjectDetector, self).__init__()

        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        bboxes = self.bbox_head(features)
        class_logits = self.class_head(features)
        return bboxes, class_logits


def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for images, target_bboxes, target_classes in dataloader:
            images, target_bboxes, target_classes = (
                images.to(device),
                target_bboxes.to(device),
                target_classes.to(device),
            )
            optimizer.zero_grad()
            pred_bboxes, pred_classes = model(images)

            loss = criterion(pred_bboxes, pred_classes, target_bboxes, target_classes)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

num_classes = 4
model = CustomObjectDetector(num_classes).to(device)
criterion = ObjectDetectionLoss()

train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_loader, optimizer, criterion, device, epochs=10)
