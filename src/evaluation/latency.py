"""Synchronized CUDA/CPU batch-latency benchmark helper."""
import statistics
import time
import torch


def benchmark_batch_latency(callable_fn, warmup_runs: int = 20, timed_runs: int = 100, device=None):
    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("warmup_runs must be >=0 and timed_runs >=1")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            callable_fn()
        sync()
        runs_ms = []
        for _ in range(timed_runs):
            sync()
            start = time.perf_counter()
            callable_fn()
            sync()
            runs_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms_per_batch": statistics.mean(runs_ms),
        "std_ms_per_batch": statistics.stdev(runs_ms) if len(runs_ms) > 1 else 0.0,
        "median_ms_per_batch": statistics.median(runs_ms),
        "min_ms_per_batch": min(runs_ms), "max_ms_per_batch": max(runs_ms),
        "warmup_runs": warmup_runs, "timed_runs": timed_runs,
        "device": str(device),
    }
