"""Seed helpers used by the experiment notebooks."""
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
