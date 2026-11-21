import struct
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "fashion-mnist"
MODEL_PATH = ROOT / "models" / "resnet18_fashion_mnist.pth"
RESULT_DIR = ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32

DEVICE = torch.device(
    "xpu" if torch.xpu.is_available() else "cpu"
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle boot"
]


class FashionDataset(Dataset):

    def __init__(self, images_path, labels_path):

        with open(images_path, "rb") as f:
            magic, count, rows, cols = struct.unpack(
                ">IIII", f.read(16)
            )

            assert magic == 2051

            images = np.frombuffer(
                f.read(), dtype=np.uint8
            ).reshape(count, rows, cols).copy()

        with open(labels_path, "rb") as f:
            magic, label_count = struct.unpack(
                ">II", f.read(8)
            )

            assert magic == 2049

            labels = np.frombuffer(
                f.read(), dtype=np.uint8
            ).copy()

        assert count == label_count

        self.images = (
            torch.from_numpy(images).float().unsqueeze(1) / 255.0
        )

        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# Load official test dataset
test_dataset = FashionDataset(
    DATA_DIR / "t10k-images-idx3-ubyte",
    DATA_DIR / "t10k-labels-idx1-ubyte"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Recreate ResNet-18 architecture
model = resnet18(weights=None)

model.conv1 = nn.Conv2d(
    1, 64, kernel_size=3,
    stride=1, padding=1, bias=False
)

model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model = model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()

confusion = torch.zeros(10, 10, dtype=torch.int64)

total_loss = 0.0
correct = 0
total = 0


# Evaluation
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Build confusion matrix
        indices = labels.cpu() * 10 + predictions.cpu()

        confusion += torch.bincount(
            indices, minlength=100
        ).reshape(10, 10)


test_loss = total_loss / total
test_accuracy = correct / total

# Calculate precision, recall and F1
macro_f1 = 0.0

print("\nFASHIONVISION AI - RESNET-18")
print("--------------------------------")
print("Device:", DEVICE)
print("Test images:", total)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print("--------------------------------")

report = []

for i, name in enumerate(CLASS_NAMES):

    tp = confusion[i, i].item()
    fp = confusion[:, i].sum().item() - tp
    fn = confusion[i, :].sum().item() - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    macro_f1 += f1

    report.append([
        name, precision, recall, f1
    ])

    print(
        f"{name:15s} "
        f"Precision: {precision:.4f} "
        f"Recall: {recall:.4f} "
        f"F1: {f1:.4f}"
    )

macro_f1 /= len(CLASS_NAMES)

print(f"\nMacro F1 Score: {macro_f1:.4f}")

# Save confusion matrix
with open(
    RESULT_DIR / "resnet18_confusion_matrix.csv",
    "w", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(["Actual / Predicted"] + CLASS_NAMES)

    for i, name in enumerate(CLASS_NAMES):
        writer.writerow([name] + confusion[i].tolist())

# Save classification report
with open(
    RESULT_DIR / "resnet18_classification_report.csv",
    "w", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Class", "Precision", "Recall", "F1 Score"
    ])

    writer.writerows(report)

print("\nResults saved in:", RESULT_DIR)