import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(ref: np.ndarray, result: np.ndarray) -> float:
    return peak_signal_noise_ratio(ref, result, data_range=255)


def compute_ssim(ref: np.ndarray, result: np.ndarray) -> float:
    return structural_similarity(ref, result, data_range=255)


def compute_imbalance_ratio(worker_times: list[float]) -> float:
    arr = np.array(worker_times, dtype=float)
    return float(arr.max() / arr.mean())


def compute_speedup(serial_time: float, parallel_time: float) -> float:
    return serial_time / parallel_time


def compute_efficiency(speedup: float, n_workers: int) -> float:
    return speedup / n_workers
