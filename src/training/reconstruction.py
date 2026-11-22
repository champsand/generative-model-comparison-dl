"""Reconstruction objectives used by Vanilla AE, Sparse AE, and VAE."""
import torch
import torch.nn.functional as F


def autoencoder_loss(reconstruction, target):
    """Model 01: mean pixel-wise MSE."""
    return F.mse_loss(reconstruction, target, reduction="mean")


def sparse_autoencoder_loss(reconstruction, target, latent, lambda_l1: float = 1e-3):
    """Model 02: reconstruction MSE + lambda * mean(abs(latent activations))."""
    mse = F.mse_loss(reconstruction, target, reduction="mean")
    l1_raw = latent.abs().mean()
    total = mse + float(lambda_l1) * l1_raw
    return total, mse, l1_raw


def vae_loss(reconstruction, target, mu, logvar, beta: float = 1.0):
    """Model 04: per-image summed reconstruction SSE + beta * summed KL, batch mean."""
    recon_per_image = (reconstruction - target).pow(2).flatten(1).sum(dim=1)
    kl_per_image = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    recon = recon_per_image.mean()
    kl = kl_per_image.mean()
    total = recon + float(beta) * kl
    return total, recon, kl, recon_per_image, kl_per_image


def train_autoencoder_batch(model, images, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, latent = model(images)
    loss = autoencoder_loss(reconstruction, images)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "reconstruction": reconstruction.detach(),
            "latent": latent.detach()}


def train_sparse_autoencoder_batch(model, images, optimizer, lambda_l1: float = 1e-3):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, latent = model(images)
    total, mse, l1_raw = sparse_autoencoder_loss(reconstruction, images, latent, lambda_l1)
    total.backward()
    optimizer.step()
    return {"total_objective": float(total.detach()), "recon_mse": float(mse.detach()),
            "l1_raw": float(l1_raw.detach()), "reconstruction": reconstruction.detach(),
            "latent": latent.detach()}


def train_vae_batch(model, images, optimizer, beta: float = 1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, mu, logvar, z = model(images)
    total, recon, kl, _, _ = vae_loss(reconstruction, images, mu, logvar, beta)
    total.backward()
    optimizer.step()
    return {"total_objective": float(total.detach()), "recon_sse": float(recon.detach()),
            "kl": float(kl.detach()), "reconstruction": reconstruction.detach(),
            "mu": mu.detach(), "logvar": logvar.detach(), "z": z.detach()}
