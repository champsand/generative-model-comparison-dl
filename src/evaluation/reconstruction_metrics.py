"""MSE, PSNR, and SSIM evaluation for image reconstruction experiments."""
import math
import numpy as np


def reconstruction_metrics(originals, reconstructions, ssim_params=None):
    """Compute aggregate test-pixel MSE/PSNR and mean per-image SSIM.

    Inputs should be arrays/tensors shaped [N,1,28,28] in [0,1]. MSE is the
    global squared-error sum divided by the total number of pixels; PSNR is
    computed from that aggregate MSE, matching the experiment notebooks.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("Install scikit-image to compute SSIM.") from exc

    if hasattr(originals, "detach"):
        originals = originals.detach().cpu().numpy()
    if hasattr(reconstructions, "detach"):
        reconstructions = reconstructions.detach().cpu().numpy()
    x = np.asarray(originals, dtype=np.float32)
    y = np.asarray(reconstructions, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected equal [N,1,28,28] arrays, got {x.shape} and {y.shape}.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs contain NaN or Infinity.")
    if x.min() < 0 or x.max() > 1 or y.min() < 0 or y.max() > 1:
        raise ValueError("Inputs must be in [0,1].")

    squared_error_sum = float(np.square(x.astype(np.float64) - y.astype(np.float64)).sum())
    total_pixels = int(x.size)
    mse = squared_error_sum / total_pixels
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    params = ssim_params or {
        "data_range": 1.0, "gaussian_weights": True, "sigma": 1.5,
        "win_size": 11, "use_sample_covariance": False,
    }
    values = [
        float(structural_similarity(a[0], b[0], **params))
        for a, b in zip(x, y)
    ]
    return {
        "num_images": int(len(x)), "num_pixels": total_pixels,
        "mse": float(mse), "psnr_db": float(psnr),
        "ssim_mean": float(np.mean(values)), "ssim_std": float(np.std(values, ddof=1)),
        "ssim_count": int(len(values)), "ssim_params": params,
    }
