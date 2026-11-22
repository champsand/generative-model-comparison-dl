"""Fully connected Vanilla GAN from Model 03."""
import torch.nn as nn


class VanillaGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, z):
        return self.net(z)


class VanillaGANDiscriminator(nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(negative_slope),
            nn.Linear(512, 256), nn.LeakyReLU(negative_slope),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, images):
        return self.net(images)


def init_vanilla_gan_weights(module, std: float = 0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        nn.init.zeros_(module.bias)
