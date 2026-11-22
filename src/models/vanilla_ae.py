"""Vanilla fully connected autoencoder from Model 01."""
import torch.nn as nn


class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, images):
        return self.encoder(images)

    def decode(self, latent):
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images):
        latent = self.encode(images)
        reconstruction = self.decode(latent)
        return reconstruction, latent
