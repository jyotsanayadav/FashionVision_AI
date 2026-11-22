"""
Task 9: latent-space organization of the trained VAE (no training, loads models/fashion_vae.pth).
Run from the project root:  python python/visualize_latent_space.py
Outputs (results/):
  vae_latent_pca.png      2D PCA of the 20-D latent means, coloured by class
  vae_latent_tsne.png     t-SNE view (only if scikit-learn is installed)
  vae_interpolation.png   smooth morphs between pairs of garments (latent style variants)
"""
import struct
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "fashion-mnist"
MODEL_PATH = ROOT / "models" / "fashion_vae.pth"
OUTPUT_DIR = ROOT / "results"
LATENT_DIM = 20
HIDDEN_DIM = 400
DEVICE = torch.device("cpu")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_images():
    with open(DATA_DIR / "t10k-images-idx3-ubyte", "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols).copy()
    return torch.from_numpy(images).float().unsqueeze(1) / 255.0


def load_labels():
    with open(DATA_DIR / "t10k-labels-idx1-ubyte", "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


class FashionVAE(nn.Module):          # identical to the class used in training
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, HIDDEN_DIM), nn.ReLU())
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(),
                                     nn.Linear(HIDDEN_DIM, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)


def scatter(points, labels, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        m = labels == c
        ax.scatter(points[m, 0], points[m, 1], s=6, color=cmap(c), label=CLASSES[c], alpha=0.7)
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("Saved", path)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    images, labels = load_images(), load_labels()

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model = FashionVAE().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("VAE loaded. Best validation loss:", ckpt.get("best_val_loss"))

    with torch.no_grad():
        mu, _ = model.encode(images)
    mu = mu.numpy()

    # ---- PCA (pure numpy) ----
    centered = mu - mu.mean(axis=0)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca2 = centered @ Vt[:2].T
    var = (S ** 2) / np.sum(S ** 2)
    scatter(pca2, labels,
            f"VAE latent space (PCA, first 2 components explain {100*var[:2].sum():.1f}% of variance)",
            OUTPUT_DIR / "vae_latent_pca.png")

    # ---- t-SNE (optional) ----
    try:
        from sklearn.manifold import TSNE
        n = 3000
        tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(mu[:n])
        scatter(tsne, labels[:n], "VAE latent space (t-SNE, 3000 test images)",
                OUTPUT_DIR / "vae_latent_tsne.png")
    except ImportError:
        print("scikit-learn not installed: skipped t-SNE (PCA plot is enough).")

    # ---- Latent interpolation: style variants between garments ----
    pairs = [(1, 3), (0, 4), (5, 9), (7, 8)]          # class pairs
    steps = 9
    fig, axes = plt.subplots(len(pairs), steps, figsize=(steps * 1.3, len(pairs) * 1.5))
    rng = np.random.default_rng(42)
    for r, (ca, cb) in enumerate(pairs):
        ia = rng.choice(np.where(labels == ca)[0]); ib = rng.choice(np.where(labels == cb)[0])
        za, zb = mu[ia], mu[ib]
        for s in range(steps):
            t = s / (steps - 1)
            z = torch.tensor((1 - t) * za + t * zb, dtype=torch.float32)[None]
            with torch.no_grad():
                img = model.decode(z).reshape(28, 28).numpy()
            axes[r, s].imshow(img, cmap="gray"); axes[r, s].axis("off")
        axes[r, 0].set_title(CLASSES[ca], fontsize=8); axes[r, -1].set_title(CLASSES[cb], fontsize=8)
    fig.suptitle("Latent interpolation: left garment morphs into right garment", fontsize=11)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "vae_interpolation.png", dpi=150); plt.close(fig)
    print("Saved", OUTPUT_DIR / "vae_interpolation.png")
    print("LATENT SPACE VISUALIZATION COMPLETED!")


if __name__ == "__main__":
    main()