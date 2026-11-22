"""Sparse fully connected autoencoder from Model 02.

The architecture is intentionally the same as Model 01; sparsity is imposed by
the L1 latent-activation penalty in src.training.reconstruction.
"""
import torch.nn as nn


class SparseAutoencoder(nn.Module):
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
        return self.decode(latent), latent
