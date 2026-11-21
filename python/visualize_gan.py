import csv
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.utils import save_image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(ROOT, "models", "fashion_gan.pth")
HISTORY_PATH = os.path.join(ROOT, "results", "gan_history.csv")
OUTPUT_DIR = os.path.join(ROOT, "results")

LATENT_DIM = 100
DEVICE = torch.device("cpu")


# ---------------- GENERATOR ----------------

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.ConvTranspose2d(
                LATENT_DIM, 128,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


# ---------------- LOAD TRAINING HISTORY ----------------

def load_history():

    epochs = []
    d_losses = []
    g_losses = []

    with open(HISTORY_PATH, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:

            epochs.append(int(row["Epoch"]))
            d_losses.append(float(row["Discriminator Loss"]))
            g_losses.append(float(row["Generator Loss"]))

    return epochs, d_losses, g_losses


# ---------------- MAIN ----------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("FASHIONVISION AI - GAN VISUALIZATION")
    print("------------------------------------")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    generator = Generator().to(DEVICE)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()

    print("Generator loaded successfully!")
    print("Training epochs:", checkpoint["epoch"])

    # -------- GENERATE 100 NEW IMAGES --------

    with torch.no_grad():

        noise = torch.randn(
            100, LATENT_DIM, 1, 1,
            device=DEVICE
        )

        generated_images = generator(noise)

    generated_path = os.path.join(
        OUTPUT_DIR,
        "gan_generated_100.png"
    )

    save_image(
        generated_images,
        generated_path,
        nrow=10,
        normalize=True,
        value_range=(-1, 1)
    )

    print("100 generated images saved:", generated_path)

    # -------- PLOT TRAINING LOSSES --------

    epochs, d_losses, g_losses = load_history()

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        d_losses,
        marker="o",
        label="Discriminator Loss"
    )

    plt.plot(
        epochs,
        g_losses,
        marker="o",
        label="Generator Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = os.path.join(
        OUTPUT_DIR,
        "gan_loss_curve.png"
    )

    plt.savefig(loss_path, dpi=150)
    plt.close()

    print("Loss curve saved:", loss_path)

    print("\nGAN VISUALIZATION COMPLETED!")


if __name__ == "__main__":
    main()