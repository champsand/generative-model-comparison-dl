"""Fréchet distance calculation from already extracted Inception features.

The experiment notebooks use pytorch-fid 0.3.0 InceptionV3 pool3 (2,048-D)
features. This module intentionally accepts those features rather than silently
choosing a different feature extractor/preprocessing pipeline.
"""
import numpy as np


def frechet_distance_from_features(real_features, generated_features, eps: float = 1e-6) -> float:
    try:
        from scipy.linalg import sqrtm
    except ImportError as exc:
        raise ImportError("Install scipy to calculate FID.") from exc
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(generated_features, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2 or real.shape[1] != fake.shape[1]:
        raise ValueError("Real and generated features must be 2-D with matching feature dimensions.")
    if len(real) < 2 or len(fake) < 2 or not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Features must contain at least two finite rows per distribution.")
    mu1, mu2 = real.mean(axis=0), fake.mean(axis=0)
    sigma1, sigma2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a significant imaginary component.")
        covmean = covmean.real
    value = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    if not np.isfinite(value):
        raise ValueError(f"FID is non-finite: {value}")
    if value < -1e-3:
        raise ValueError(f"FID is materially negative: {value}")
    return max(0.0, value)
