"""DDPM linear noise schedule, time-conditioned U-Net, and ancestral sampler.

Architecture and defaults mirror Model 06: T=1000, linear beta 1e-4..0.02,
base channels 32, channel multipliers (1,2,4), two residual blocks per level,
time embedding 128, GroupNorm(8), dropout 0.1.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """Linear-beta DDPM schedule with forward and posterior coefficients."""
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.T = int(T)
        betas = np.linspace(beta_start, beta_end, self.T, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        coef_x0 = betas * np.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        coef_xt = (1.0 - alpha_bar_prev) * np.sqrt(alphas) / (1.0 - alpha_bar)
        sqrt_recip_ab = np.sqrt(1.0 / alpha_bar)
        sqrt_recipm1_ab = np.sqrt(1.0 / alpha_bar - 1.0)

        def t32(values):
            return torch.tensor(values, dtype=torch.float32, device=device)

        self.betas = t32(betas)
        self.alpha_bar = t32(alpha_bar)
        self.sqrt_alpha_bar = t32(np.sqrt(alpha_bar))
        self.sqrt_one_minus_alpha_bar = t32(np.sqrt(1.0 - alpha_bar))
        self.f_coef_x0 = coef_x0.tolist()
        self.f_coef_xt = coef_xt.tolist()
        self.f_sqrt_recip_ab = sqrt_recip_ab.tolist()
        self.f_sqrt_recipm1_ab = sqrt_recipm1_ab.tolist()
        self.f_post_std = np.sqrt(posterior_variance).tolist()
        self.f_sqrt_recip_alpha = np.sqrt(1.0 / alphas).tolist()
        self.f_eps_coef = (betas / np.sqrt(1.0 - alpha_bar)).tolist()


def q_sample(schedule: DiffusionSchedule, x0, timesteps, noise):
    a = schedule.sqrt_alpha_bar[timesteps].view(-1, 1, 1, 1)
    s = schedule.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1, 1)
    return a * x0 + s * noise


def to_model_space(images_unit):
    return images_unit * 2.0 - 1.0


def to_unit(images_model):
    return ((images_model + 1.0) / 2.0).clamp(0.0, 1.0)


def timestep_embedding(timesteps, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                           device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, groups=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, channel_mult=(1, 2, 4), num_res_blocks=2,
                 time_emb_dim=128, groupnorm_groups=8, dropout=0.1):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        channels = [base_channels * m for m in channel_mult]
        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(current, out_ch, time_emb_dim, groupnorm_groups, dropout))
                current = out_ch
            self.down_blocks.append(blocks)
            self.downsamples.append(Downsample(current) if level < len(channels) - 1 else nn.Identity())
        self.mid = nn.ModuleList([
            ResBlock(current, current, time_emb_dim, groupnorm_groups, dropout),
            ResBlock(current, current, time_emb_dim, groupnorm_groups, dropout),
        ])
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            blocks = nn.ModuleList([ResBlock(current + out_ch, out_ch, time_emb_dim,
                                             groupnorm_groups, dropout)])
            current = out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(current, out_ch, time_emb_dim, groupnorm_groups, dropout))
            self.up_blocks.append(blocks)
            self.upsamples.append(Upsample(current) if level > 0 else nn.Identity())
        self.out_norm = nn.GroupNorm(groupnorm_groups, current)
        self.out_conv = nn.Conv2d(current, in_channels, 3, padding=1)

    def forward(self, x, timesteps):
        temb = self.time_mlp(timestep_embedding(timesteps, self.time_emb_dim))
        h = self.in_conv(x)
        skips = []
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, temb)
            skips.append(h)
            h = downsample(h)
        for block in self.mid:
            h = block(h, temb)
        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, temb)
            h = upsample(h)
        return self.out_conv(F.silu(self.out_norm(h)))


def p_mean_std(model, schedule: DiffusionSchedule, x_t, t_int: int, clip_x0: bool = True):
    timesteps = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps_hat = model(x_t, timesteps)
    x0_hat = schedule.f_sqrt_recip_ab[t_int] * x_t - schedule.f_sqrt_recipm1_ab[t_int] * eps_hat
    if clip_x0:
        x0_hat = x0_hat.clamp(-1.0, 1.0)
    mean = schedule.f_coef_x0[t_int] * x0_hat + schedule.f_coef_xt[t_int] * x_t
    return mean, schedule.f_post_std[t_int], eps_hat, x0_hat


@torch.no_grad()
def ddpm_sample(model, schedule: DiffusionSchedule, n: int, device, generator: torch.Generator,
                clip_x0: bool = True, record_after_steps=None):
    """Run the complete ancestral reverse process, optionally recording selected steps."""
    was_training = model.training
    model.eval()
    record_set = set(record_after_steps or [])
    records = {}
    x = torch.randn((n, 1, 28, 28), generator=generator, device=device)
    if 0 in record_set:
        records[0] = x.detach().float().cpu()
    try:
        for step_index, t_int in enumerate(reversed(range(schedule.T))):
            mean, std, _, _ = p_mean_std(model, schedule, x, t_int, clip_x0)
            if t_int > 0:
                noise = torch.randn(x.shape, generator=generator, device=device)
                x = mean + std * noise
            else:
                x = mean
            if (step_index + 1) in record_set:
                records[step_index + 1] = x.detach().float().cpu()
    finally:
        model.train(was_training)
    return x, records
