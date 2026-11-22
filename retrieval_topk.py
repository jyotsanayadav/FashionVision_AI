"""
Task 10: image retrieval top-k with the trained ResNet-18 (no training, loads models/resnet18_fashion_mnist.pth).
Each test image is a query; its 512-D embedding (output just before the classifier) is compared with the
other 9,999 test images by cosine similarity. A retrieved image is "correct" if it has the same class.

Run from the project root:  python python/retrieval_topk.py
Outputs (results/):
  retrieval_topk_metrics.csv   Precision@k, Hit@k and per-class Precision@5
  retrieval_examples.png       example queries with their 5 nearest matches (green = same class, red = different)
"""
import csv
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "fashion-mnist"
MODEL_PATH = ROOT / "models" / "resnet18_fashion_mnist.pth"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
KS = [1, 5, 10]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_idx():
    with open(DATA_DIR / "t10k-images-idx3-ubyte", "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols).copy()
    with open(DATA_DIR / "t10k-labels-idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return images, labels


def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.fc = nn.Identity()          # output is now the 512-D embedding
    return model.to(DEVICE).eval()


def main():
    images, labels = load_idx()
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    model = build_model()
    print("ResNet-18 loaded. Extracting embeddings for", len(x), "images...")

    feats = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            feats.append(model(x[i:i + 256].to(DEVICE)).cpu())
    feats = torch.cat(feats)
    feats = feats / feats.norm(dim=1, keepdim=True)          # cosine similarity
    y = torch.from_numpy(labels).long()
    n = len(y)

    # ---- nearest neighbours (self excluded), processed in chunks ----
    kmax = max(KS)
    nn_idx = torch.zeros(n, kmax, dtype=torch.long)
    for s in range(0, n, 1000):
        sim = feats[s:s + 1000] @ feats.T
        sim[torch.arange(sim.size(0)), torch.arange(s, s + sim.size(0))] = -2.0   # drop the query itself
        nn_idx[s:s + 1000] = sim.topk(kmax, dim=1).indices
    same = (y[nn_idx] == y[:, None])                          # n x kmax boolean

    rows = [["Metric", "Value"]]
    print("\nRETRIEVAL RESULTS (10,000 queries, cosine similarity)")
    for k in KS:
        p = same[:, :k].float().mean().item()                 # precision@k
        h = same[:, :k].any(dim=1).float().mean().item()      # hit@k
        print(f"k={k:2d}  Precision@{k}: {p*100:.2f}%   Hit@{k}: {h*100:.2f}%")
        rows += [[f"Precision@{k}", f"{p*100:.2f}%"], [f"Hit@{k}", f"{h*100:.2f}%"]]
    print("\nPer-class Precision@5:")
    for c in range(10):
        pc = same[y == c, :5].float().mean().item()
        print(f"  {CLASS_NAMES[c]:12s} {pc*100:.2f}%")
        rows.append([f"Precision@5 {CLASS_NAMES[c]}", f"{pc*100:.2f}%"])
    with open(RESULT_DIR / "retrieval_topk_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ---- example figure: one query for 6 classes, with 5 matches each ----
    show = [1, 3, 6, 7, 8, 9]                                 # Trouser, Dress, Shirt, Sneaker, Bag, Ankle boot
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(show), 6, figsize=(8, 1.4 * len(show)))
    for r, c in enumerate(show):
        q = int(rng.choice(np.where(labels == c)[0]))
        axes[r, 0].imshow(images[q], cmap="gray"); axes[r, 0].set_title("Query", fontsize=8)
        for j in range(5):
            m = int(nn_idx[q, j])
            ax = axes[r, j + 1]
            ax.imshow(images[m], cmap="gray")
            ax.set_title(CLASS_NAMES[labels[m]], fontsize=6,
                         color="green" if labels[m] == c else "red")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Retrieval: query and 5 nearest garments (green = same class, red = different)", fontsize=9)
    fig.tight_layout(); fig.savefig(RESULT_DIR / "retrieval_examples.png", dpi=150); plt.close(fig)
    print("\nSaved results/retrieval_topk_metrics.csv and results/retrieval_examples.png")


if __name__ == "__main__":
    main()