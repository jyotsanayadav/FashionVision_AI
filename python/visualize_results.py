import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Fashion-MNIST classes
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# 1. Load training history
history = []

with open(RESULTS_DIR / "resnet18_history.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        history.append(row)

epochs = [int(row["Epoch"]) for row in history]
train_loss = [float(row["Train Loss"]) for row in history]
val_loss = [float(row["Val Loss"]) for row in history]
train_acc = [float(row["Train Accuracy"]) * 100 for row in history]
val_acc = [float(row["Val Accuracy"]) * 100 for row in history]


# 2. Training and validation loss graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 Training vs Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "resnet18_loss_curve.png", dpi=300)
plt.close()


# 3. Training and validation accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet-18 Training vs Validation Accuracy")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "resnet18_accuracy_curve.png", dpi=300)
plt.close()


# 4. Load confusion matrix
with open(RESULTS_DIR / "resnet18_confusion_matrix.csv", "r") as file:
    reader = csv.reader(file)
    rows = list(reader)

matrix = np.array(
    [[int(value) for value in row[1:]] for row in rows[1:]],
    dtype=int
)

plt.figure(figsize=(10, 8))
plt.imshow(matrix, interpolation="nearest")
plt.title("ResNet-18 Confusion Matrix")
plt.colorbar(label="Number of Images")

plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
plt.yticks(range(len(CLASSES)), CLASSES)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "resnet18_confusion_matrix.png", dpi=300)
plt.close()

print("Visualizations generated successfully!")
print("Saved in:", RESULTS_DIR)
print("1. resnet18_loss_curve.png")
print("2. resnet18_accuracy_curve.png")
print("3. resnet18_confusion_matrix.png")