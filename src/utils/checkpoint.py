"""Atomic PyTorch checkpoint I/O helpers."""
from pathlib import Path
import os
import torch


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_torch_checkpoint(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # compatibility with older PyTorch versions
        return torch.load(path, map_location=map_location)
