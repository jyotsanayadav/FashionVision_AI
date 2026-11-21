import struct
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "fashion-mnist"
MODEL_PATH = ROOT / "models" / "fashion_vae.pth"
OUTPUT_DIR = ROOT / "results"

LATENT_DIM = 20
HIDDEN_DIM = 400

DEVICE = torch.device("cpu")


# ---------------- DATASET ----------------

def load_images():

    path = DATA_DIR / "t10k-images-idx3-ubyte"

    with open(path, "rb") as f:

        magic, count, rows, cols = struct.unpack(
            ">IIII", f.read(16)
        )

        assert magic == 2051

        images = np.frombuffer(
            f.read(), dtype=np.uint8
        ).reshape(count, rows, cols).copy()

    return torch.from_numpy(images).float().unsqueeze(1) / 255.0


# ---------------- VAE MODEL ----------------

class FashionVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, HIDDEN_DIM),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 784),
            nn.Sigmoid()
        )

    def encode(self, x):

        x = x.view(x.size(0), -1)

        hidden = self.encoder(x)

        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ---------------- MAIN ----------------

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("FASHIONVISION AI - VAE VISUALIZATION")
    print("------------------------------------")

    images = load_images()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = FashionVAE().to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("Model loaded successfully!")
    print("Best validation loss:", checkpoint["best_val_loss"])

    # -------- IMAGE RECONSTRUCTION --------

    original = images[:10].to(DEVICE)

    with torch.no_grad():

        reconstruction, _, _ = model(original)

    reconstruction = reconstruction.reshape(-1, 1, 28, 28)

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))

    for i in range(10):

        axes[0, i].imshow(
            original[i].squeeze().numpy(),
            cmap="gray"
        )

        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstruction[i].squeeze().cpu().numpy(),
            cmap="gray"
        )

        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()

    reconstruction_path = OUTPUT_DIR / "vae_reconstruction.png"

    plt.savefig(reconstruction_path, dpi=150)
    plt.close()

    print("Reconstruction saved:", reconstruction_path)

    # -------- NEW IMAGE GENERATION --------

    with torch.no_grad():

        random_latent = torch.randn(25, LATENT_DIM).to(DEVICE)

        generated = model.decode(random_latent)

    generated = generated.reshape(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(5, 5, figsize=(7, 7))

    for i, ax in enumerate(axes.flat):

        ax.imshow(generated[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()

    generated_path = OUTPUT_DIR / "vae_generated_images.png"

    plt.savefig(generated_path, dpi=150)
    plt.close()

    print("Generated images saved:", generated_path)

    print("\nVAE VISUALIZATION COMPLETED!")


if __name__ == "__main__":
    main()