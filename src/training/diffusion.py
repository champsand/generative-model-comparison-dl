"""DDPM epsilon-prediction objective and one-batch optimizer update."""
import torch
import torch.nn.functional as F

from src.models.ddpm import q_sample


def diffusion_epsilon_loss(model, schedule, x0_unit, timesteps=None, noise=None):
    """Return MSE between predicted and true Gaussian noise; DDPM uses x in [-1,1]."""
    x0 = x0_unit * 2.0 - 1.0
    batch = x0.shape[0]
    if timesteps is None:
        timesteps = torch.randint(0, schedule.T, (batch,), device=x0.device)
    if noise is None:
        noise = torch.randn_like(x0)
    x_t = q_sample(schedule, x0, timesteps, noise)
    eps_hat = model(x_t, timesteps)
    loss = F.mse_loss(eps_hat, noise, reduction="mean")
    return loss, {"x_t": x_t, "timesteps": timesteps, "noise": noise, "eps_hat": eps_hat}


def train_diffusion_batch(model, schedule, images_unit, optimizer, grad_clip_norm: float = 1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, details = diffusion_epsilon_loss(model, schedule, images_unit)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()
    return {"epsilon_mse": float(loss.detach()),
            "grad_clip_norm": float(grad_clip_norm)}
