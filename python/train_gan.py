import struct
import csv
import time
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


# ---------------- CONFIGURATION ----------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "dataset", "fashion-mnist")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "fashion_gan.pth")
HISTORY_PATH = os.path.join(RESULT_DIR, "gan_history.csv")

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0002

LATENT_DIM = 100
SEED = 42

DEVICE = torch.device(
    "xpu" if torch.xpu.is_available() else "cpu"
)


# ---------------- DATASET ----------------

class FashionDataset(Dataset):

    def __init__(self, images_path):

        with open(images_path, "rb") as f:

            magic, count, rows, cols = struct.unpack(
                ">IIII", f.read(16)
            )

            assert magic == 2051

            images = np.frombuffer(
                f.read(), dtype=np.uint8
            ).reshape(count, rows, cols).copy()

        self.images = (
            torch.from_numpy(images)
            .float()
            .unsqueeze(1) / 127.5 - 1.0
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


# ---------------- GENERATOR ----------------

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # 1 x 1 -> 7 x 7
            nn.ConvTranspose2d(
                LATENT_DIM, 128,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 7 x 7 -> 14 x 14
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 -> 28 x 28
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


# ---------------- DISCRIMINATOR ----------------

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # 28 x 28 -> 14 x 14
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2, inplace=True),

            # 14 x 14 -> 7 x 7
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, image):
        return self.network(image)


# ---------------- WEIGHT INITIALIZATION ----------------

def initialize_weights(model):

    for layer in model.modules():

        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):

            nn.init.normal_(layer.weight, 0.0, 0.02)

        elif isinstance(layer, nn.BatchNorm2d):

            nn.init.normal_(layer.weight, 1.0, 0.02)
            nn.init.constant_(layer.bias, 0)


# ---------------- TRAINING ----------------

def main():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("FASHIONVISION AI - DEEP CONVOLUTIONAL GAN")
    print("------------------------------------------")
    print("Device:", DEVICE)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    if DEVICE.type == "xpu":
        torch.xpu.manual_seed_all(SEED)

    print("\nLoading Fashion-MNIST...")

    dataset = FashionDataset(
        os.path.join(DATA_DIR, "train-images-idx3-ubyte")
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    print("Training images:", len(dataset))

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(
        25, LATENT_DIM, 1, 1,
        device=DEVICE
    )

    history = []

    print("\nStarting GAN training...")

    for epoch in range(EPOCHS):

        start_time = time.time()

        generator.train()
        discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        for real_images in loader:

            real_images = real_images.to(DEVICE)

            batch_size = real_images.size(0)

            real_labels = torch.ones(
                batch_size, 1, device=DEVICE
            )

            fake_labels = torch.zeros(
                batch_size, 1, device=DEVICE
            )

            # -------- TRAIN DISCRIMINATOR --------

            optimizer_d.zero_grad()

            real_output = discriminator(real_images)

            real_loss = criterion(
                real_output, real_labels
            )

            noise = torch.randn(
                batch_size, LATENT_DIM, 1, 1,
                device=DEVICE
            )

            fake_images = generator(noise)

            fake_output = discriminator(
                fake_images.detach()
            )

            fake_loss = criterion(
                fake_output, fake_labels
            )

            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_d.step()

            # -------- TRAIN GENERATOR --------

            optimizer_g.zero_grad()

            generated_output = discriminator(fake_images)

            g_loss = criterion(
                generated_output, real_labels
            )

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        avg_d_loss = total_d_loss / len(loader)
        avg_g_loss = total_g_loss / len(loader)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Discriminator Loss: {avg_d_loss:.4f}")
        print(f"Generator Loss: {avg_g_loss:.4f}")
        print(f"Time: {elapsed:.2f} seconds")

        history.append([
            epoch + 1,
            avg_d_loss,
            avg_g_loss,
            elapsed
        ])

        # Save generated samples after every epoch

        generator.eval()

        with torch.no_grad():

            generated = generator(fixed_noise)

            sample_path = os.path.join(
                RESULT_DIR,
                f"gan_epoch_{epoch + 1:02d}.png"
            )

            save_image(
                generated,
                sample_path,
                nrow=5,
                normalize=True,
                value_range=(-1, 1)
            )

        # Save checkpoint after every epoch

        torch.save({
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "epoch": epoch + 1,
            "latent_dim": LATENT_DIM
        }, MODEL_PATH)

        print("Generated samples saved!")
        print("Checkpoint saved!")

        # Update history after every epoch

        with open(HISTORY_PATH, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Epoch",
                "Discriminator Loss",
                "Generator Loss",
                "Time"
            ])

            writer.writerows(history)

    print("\nGAN TRAINING COMPLETED")
    print("------------------------------------------")
    print("Generator saved:", MODEL_PATH)
    print("History saved:", HISTORY_PATH)
    print("Generated images saved in:", RESULT_DIR)


if __name__ == "__main__":
    main()