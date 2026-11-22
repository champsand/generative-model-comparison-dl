"""MLP classifier and separately trained frozen-latent decoder from Model 07."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64,
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def encode(self, images):
        x = images.view(images.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def forward(self, images):
        x = images.view(images.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1))
        logits = self.fc3(z)
        return logits, z

    def forward_with_preactivations(self, images):
        x = images.view(images.size(0), -1)
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        z = F.relu(a2)
        logits = self.fc3(z)
        return logits, z, a1, a2


class FrozenLatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1, 1, 28, 28)
