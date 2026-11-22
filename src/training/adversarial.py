"""One-batch adversarial updates matching the Vanilla GAN and DCGAN notebooks."""
import torch
import torch.nn as nn


def _sample_noise(batch_size, latent_dim, device, noise_generator=None, convolutional=False):
    shape = (batch_size, latent_dim, 1, 1) if convolutional else (batch_size, latent_dim)
    if noise_generator is None:
        return torch.randn(shape, device=device)
    # The experiment notebooks keep their train-noise generator on CPU.
    return torch.randn(shape, generator=noise_generator, device="cpu").to(device, non_blocking=True)


def train_vanilla_gan_batch(generator, discriminator, real01, optimizer_g, optimizer_d,
                            latent_dim: int = 100, noise_generator=None):
    """Vanilla GAN update: sigmoid discriminator + BCELoss + non-saturating G loss."""
    device = real01.device
    real = real01 * 2.0 - 1.0
    batch = real.size(0)
    ones = torch.ones(batch, 1, device=device)
    zeros = torch.zeros(batch, 1, device=device)
    criterion = nn.BCELoss()

    z_d = _sample_noise(batch, latent_dim, device, noise_generator, convolutional=False)
    fake = generator(z_d)
    optimizer_d.zero_grad(set_to_none=True)
    d_real = discriminator(real)
    d_fake = discriminator(fake.detach())
    loss_d = 0.5 * (criterion(d_real, ones) + criterion(d_fake, zeros))
    loss_d.backward()
    optimizer_d.step()

    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        optimizer_g.zero_grad(set_to_none=True)
        loss_g = criterion(discriminator(fake), ones)
        loss_g.backward()
        optimizer_g.step()
    finally:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

    return {"g_loss": float(loss_g.detach()), "d_loss": float(loss_d.detach()),
            "d_real_mean": float(d_real.detach().mean()), "d_fake_mean": float(d_fake.detach().mean())}


def train_dcgan_batch(generator, discriminator, real01, optimizer_g, optimizer_d,
                      latent_dim: int = 100, noise_generator=None):
    """DCGAN update: raw discriminator logits + BCEWithLogitsLoss."""
    device = real01.device
    real = real01 * 2.0 - 1.0
    batch = real.size(0)
    ones = torch.ones(batch, device=device)
    zeros = torch.zeros(batch, device=device)
    criterion = nn.BCEWithLogitsLoss()

    z_d = _sample_noise(batch, latent_dim, device, noise_generator, convolutional=True)
    with torch.no_grad():
        fake_d = generator(z_d)
    optimizer_d.zero_grad(set_to_none=True)
    logits_real = discriminator(real)
    logits_fake = discriminator(fake_d.detach())
    loss_d_real = criterion(logits_real, ones)
    loss_d_fake = criterion(logits_fake, zeros)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_d.backward()
    optimizer_d.step()

    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        z_g = _sample_noise(batch, latent_dim, device, noise_generator, convolutional=True)
        optimizer_g.zero_grad(set_to_none=True)
        logits_g = discriminator(generator(z_g))
        loss_g = criterion(logits_g, ones)
        loss_g.backward()
        optimizer_g.step()
    finally:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

    return {"g_loss": float(loss_g.detach()), "d_loss": float(loss_d.detach()),
            "d_loss_real": float(loss_d_real.detach()), "d_loss_fake": float(loss_d_fake.detach()),
            "d_real_prob": float(torch.sigmoid(logits_real.detach()).mean()),
            "d_fake_prob": float(torch.sigmoid(logits_fake.detach()).mean())}
