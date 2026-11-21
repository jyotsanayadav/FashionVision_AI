import struct
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "fashion-mnist"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")


# Fashion-MNIST dataset loader
class FashionDataset(Dataset):

    def __init__(self, images_path, labels_path):

        with open(images_path, "rb") as f:
            magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051

            images = np.frombuffer(
                f.read(), dtype=np.uint8
            ).reshape(count, rows, cols).copy()

        with open(labels_path, "rb") as f:
            magic, label_count = struct.unpack(">II", f.read(8))
            assert magic == 2049

            labels = np.frombuffer(
                f.read(), dtype=np.uint8
            ).copy()

        assert count == label_count

        self.images = torch.from_numpy(images).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# Load training and test data
train_dataset = FashionDataset(
    DATA_DIR / "train-images-idx3-ubyte",
    DATA_DIR / "train-labels-idx1-ubyte"
)

test_dataset = FashionDataset(
    DATA_DIR / "t10k-images-idx3-ubyte",
    DATA_DIR / "t10k-labels-idx1-ubyte"
)

# Split training data into training and validation
train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)


# Create ResNet-18
model = resnet18(weights=None)

# Adapt ResNet for 28x28 grayscale images
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=3, stride=1,
    padding=1, bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE
)


# Evaluation function
def evaluate(loader):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# Training
print("FASHIONVISION AI - RESNET-18")
print("--------------------------------")
print("Device:", DEVICE)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))
print()

best_val_accuracy = 0.0
history = []

for epoch in range(EPOCHS):

    start_time = time.time()

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(val_loader)

    elapsed = time.time() - start_time

    history.append([
        epoch + 1,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        elapsed
    ])

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_DIR / "resnet18_fashion_mnist.pth"
        )


# Final test evaluation using best checkpoint
model.load_state_dict(
    torch.load(
        MODEL_DIR / "resnet18_fashion_mnist.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(test_loader)

print("\nFINAL TEST RESULTS")
print("------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save training history
with open(
    RESULT_DIR / "resnet18_history.csv",
    "w", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Epoch", "Train Loss", "Train Accuracy",
        "Val Loss", "Val Accuracy", "Time"
    ])

    writer.writerows(history)

print("\nModel saved in:", MODEL_DIR)
print("Training history saved in:", RESULT_DIR)