"""Variational Autoencoder from Model 04."""
import torch
import torch.nn as nn


def clamp_logvar(logvar: torch.Tensor, minimum: float = -20.0, maximum: float = 10.0):
    return torch.clamp(logvar, minimum, maximum)


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64,
                 logvar_min: float = -20.0, logvar_max: float = 10.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def encode(self, images):
        h = self.encoder(images)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, images, eps=None):
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar, eps=eps)
        return self.decode(z), mu, logvar, z
