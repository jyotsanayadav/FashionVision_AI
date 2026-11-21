import struct
import csv
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------- CONFIGURATION ----------------

DATA_DIR = "dataset/fashion-mnist"

MODEL_PATH = "models/fashion_vae.pth"
HISTORY_PATH = "results/vae_history.csv"

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

INPUT_DIM = 784
HIDDEN_DIM = 400
LATENT_DIM = 20

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
            .unsqueeze(1) / 255.0
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


# ---------------- VAE MODEL ----------------

class FashionVAE(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU()
        )

        # Mean and log variance of latent distribution
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
            nn.Sigmoid()
        )

    def encode(self, x):

        x = x.view(x.size(0), -1)

        hidden = self.encoder(x)

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ---------------- VAE LOSS ----------------

def vae_loss(reconstruction, original, mu, logvar):

    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstruction,
        original.view(original.size(0), -1),
        reduction="sum"
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_divergence

    return total_loss, reconstruction_loss, kl_divergence


# ---------------- TRAINING ----------------

def run_epoch(model, loader, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_reconstruction = 0
    total_kl = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:

        for images in loader:

            images = images.to(DEVICE)

            if training:
                optimizer.zero_grad()

            reconstruction, mu, logvar = model(images)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()

    dataset_size = len(loader.dataset)

    return (
        total_loss / dataset_size,
        total_reconstruction / dataset_size,
        total_kl / dataset_size
    )


# ---------------- MAIN ----------------

def main():

    print("FASHIONVISION AI - VARIATIONAL AUTOENCODER")
    print("------------------------------------------")
    print("Device:", DEVICE)

    torch.manual_seed(SEED)

    if DEVICE.type == "xpu":
        torch.xpu.manual_seed_all(SEED)

    print("\nLoading Fashion-MNIST...")

    dataset = FashionDataset(
        f"{DATA_DIR}/train-images-idx3-ubyte"
    )

    train_size = 50000
    val_size = 10000

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    model = FashionVAE().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_val_loss = float("inf")

    with open(HISTORY_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Train Reconstruction Loss",
            "Train KL Loss",
            "Validation Loss",
            "Validation Reconstruction Loss",
            "Validation KL Loss",
            "Time"
        ])

        for epoch in range(EPOCHS):

            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            train_loss, train_recon, train_kl = run_epoch(
                model,
                train_loader,
                optimizer
            )

            val_loss, val_recon, val_kl = run_epoch(
                model,
                val_loader
            )

            elapsed = time.time() - start_time

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Reconstruction Loss: {train_recon:.4f}")
            print(f"Train KL Loss: {train_kl:.4f}")

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Reconstruction Loss: {val_recon:.4f}")
            print(f"Validation KL Loss: {val_kl:.4f}")

            print(f"Time: {elapsed:.2f} seconds")

            writer.writerow([
                epoch + 1,
                train_loss,
                train_recon,
                train_kl,
                val_loss,
                val_recon,
                val_kl,
                elapsed
            ])

            f.flush()

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "latent_dim": LATENT_DIM,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch + 1
                }, MODEL_PATH)

                print("Best VAE model saved!")

    print("\nVAE TRAINING COMPLETED")
    print("------------------------------------------")
    print("Best validation loss:", best_val_loss)
    print("Model saved:", MODEL_PATH)
    print("History saved:", HISTORY_PATH)


if __name__ == "__main__":
    main()