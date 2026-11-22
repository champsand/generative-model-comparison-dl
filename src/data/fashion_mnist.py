"""Canonical Fashion-MNIST pooling and shared stratified split.

The implementation follows the experiment notebooks exactly:
1. torchvision train=True (60,000) followed by train=False (10,000).
2. Labels are concatenated in that order.
3. Per-class shuffle using np.random.default_rng(42).
4. 5,600 / 700 / 700 indices per class for train / validation / test.
5. Concatenate class chunks, then shuffle train, validation, test in that order.

Important: because the official 60k/10k partitions are pooled before splitting,
the project's 7,000-image test split is not the untouched official test partition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import FashionMNIST

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMAGE_SHAPE = (1, 28, 28)
TOTAL_IMAGES = 70_000
DEFAULT_SEED = 42
DEFAULT_COUNTS_PER_CLASS = (5_600, 700, 700)


def load_pooled_fashion_mnist(data_root: str | Path, download: bool = True):
    """Return pooled images ``[70000,1,28,28]`` float32 in [0,1] and int64 labels.

    Dividing the uint8 torchvision image tensors by 255 is numerically equivalent
to applying ``transforms.ToTensor()`` to these 8-bit grayscale images.
    """
    root = str(Path(data_root).expanduser())
    train_ds = FashionMNIST(root=root, train=True, download=download)
    test_ds = FashionMNIST(root=root, train=False, download=download)
    if len(train_ds) != 60_000 or len(test_ds) != 10_000:
        raise RuntimeError("Unexpected Fashion-MNIST partition sizes.")

    images_u8 = torch.cat([train_ds.data, test_ds.data], dim=0)
    images = images_u8.to(torch.float32).div_(255.0).unsqueeze(1).contiguous()
    labels = torch.cat([train_ds.targets, test_ds.targets], dim=0).cpu().numpy().astype(np.int64)

    if tuple(images.shape) != (TOTAL_IMAGES, *IMAGE_SHAPE):
        raise RuntimeError(f"Unexpected pooled image shape: {tuple(images.shape)}")
    if labels.shape != (TOTAL_IMAGES,):
        raise RuntimeError(f"Unexpected pooled label shape: {labels.shape}")
    if not torch.isfinite(images).all() or images.min().item() < 0 or images.max().item() > 1:
        raise RuntimeError("Pooled images must be finite float32 values in [0,1].")
    return images, labels


def make_stratified_split(
    labels: np.ndarray,
    seed: int = DEFAULT_SEED,
    train_per_class: int = DEFAULT_COUNTS_PER_CLASS[0],
    val_per_class: int = DEFAULT_COUNTS_PER_CLASS[1],
    test_per_class: int = DEFAULT_COUNTS_PER_CLASS[2],
) -> Dict[str, np.ndarray]:
    """Create the exact shared stratified split used by all seven notebooks."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (TOTAL_IMAGES,):
        raise ValueError(f"Expected {TOTAL_IMAGES} labels, got {labels.shape}.")
    if (train_per_class, val_per_class, test_per_class) != DEFAULT_COUNTS_PER_CLASS:
        raise ValueError("This project uses exactly 5,600/700/700 samples per class.")

    rng = np.random.default_rng(seed)
    train_parts, val_parts, test_parts = [], [], []
    for class_id in range(10):
        class_indices = np.flatnonzero(labels == class_id).copy()
        if class_indices.size != 7_000:
            raise ValueError(f"Class {class_id} contains {class_indices.size} samples, expected 7,000.")
        rng.shuffle(class_indices)
        train_parts.append(class_indices[:5_600])
        val_parts.append(class_indices[5_600:6_300])
        test_parts.append(class_indices[6_300:])

    train = np.concatenate(train_parts).astype(np.int64)
    validation = np.concatenate(val_parts).astype(np.int64)
    test = np.concatenate(test_parts).astype(np.int64)
    # Keep this order: it is part of the canonical index hashes.
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    return {
        "train": np.ascontiguousarray(train, dtype=np.int64),
        "validation": np.ascontiguousarray(validation, dtype=np.int64),
        "test": np.ascontiguousarray(test, dtype=np.int64),
    }


def sha256_int64(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_split(splits: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    expected_sizes = {"train": 56_000, "validation": 7_000, "test": 7_000}
    expected_per_class = {"train": 5_600, "validation": 700, "test": 700}
    report: Dict[str, object] = {"checks": {}, "counts": {}, "per_class": {}, "sha256": {}}

    if set(splits) != set(expected_sizes):
        raise ValueError(f"Split keys must be {sorted(expected_sizes)}.")
    for name in ("train", "validation", "test"):
        idx = np.ascontiguousarray(splits[name], dtype=np.int64)
        if not np.array_equal(idx, splits[name]):
            raise ValueError(f"{name} indices could not be represented as int64.")
        counts = np.bincount(labels[idx], minlength=10)
        report["counts"][name] = int(len(idx))
        report["per_class"][name] = counts.tolist()
        report["sha256"][name] = sha256_int64(idx)
        report["checks"][f"{name}_size"] = len(idx) == expected_sizes[name]
        report["checks"][f"{name}_unique"] = len(np.unique(idx)) == len(idx)
        report["checks"][f"{name}_per_class"] = bool(np.all(counts == expected_per_class[name]))
        if idx.min() < 0 or idx.max() >= len(labels):
            raise ValueError(f"{name} contains an out-of-range index.")

    names = ("train", "validation", "test")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            report["checks"][f"disjoint_{left}_{right}"] = (
                np.intersect1d(splits[left], splits[right]).size == 0
            )
    union = np.sort(np.concatenate([splits[k] for k in names]))
    report["checks"]["full_coverage"] = np.array_equal(union, np.arange(TOTAL_IMAGES))
    report["all_passed"] = all(report["checks"].values())
    if not report["all_passed"]:
        raise ValueError(f"Invalid split: {report['checks']}")
    return report


def build_shared_datasets(images: torch.Tensor, labels: np.ndarray, seed: int = DEFAULT_SEED):
    """Build TensorDataset subsets plus canonical split indices and verification."""
    if tuple(images.shape) != (TOTAL_IMAGES, *IMAGE_SHAPE):
        raise ValueError(f"Expected images {(TOTAL_IMAGES, *IMAGE_SHAPE)}, got {tuple(images.shape)}")
    labels = np.asarray(labels, dtype=np.int64)
    splits = make_stratified_split(labels, seed=seed)
    report = verify_split(splits, labels)
    dataset = TensorDataset(images, torch.from_numpy(labels.copy()).long())
    subsets = {name: Subset(dataset, idx.tolist()) for name, idx in splits.items()}
    return subsets, splits, report


def build_shared_loaders(
    subsets: Dict[str, Subset],
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Build deterministic train loader and ordered validation/test loaders."""
    if batch_size not in (128, 256):
        raise ValueError("The assignment specifies batch_size 128 or 256.")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loaders = {
        "train": DataLoader(subsets["train"], batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=num_workers, pin_memory=pin_memory),
        "validation": DataLoader(subsets["validation"], batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(subsets["test"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
    }
    return loaders, generator
