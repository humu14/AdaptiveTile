import numpy as np
import pytest
from src.metrics import compute_psnr, compute_ssim, compute_imbalance_ratio, compute_speedup, compute_efficiency


def test_psnr_identical():
    img = np.random.randint(0, 256, (512, 768), dtype=np.uint8)
    assert compute_psnr(img, img) == float("inf")


def test_psnr_range():
    ref = np.zeros((64, 64), dtype=np.uint8)
    noisy = np.full((64, 64), 1, dtype=np.uint8)
    psnr = compute_psnr(ref, noisy)
    assert 40 < psnr < 60


def test_ssim_identical():
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    assert abs(compute_ssim(img, img) - 1.0) < 1e-6


def test_imbalance_ratio_equal():
    times = [1.0, 1.0, 1.0, 1.0]
    assert compute_imbalance_ratio(times) == pytest.approx(1.0)


def test_imbalance_ratio_skewed():
    times = [4.0, 1.0, 1.0, 1.0]
    assert compute_imbalance_ratio(times) == pytest.approx(4.0 / 1.75)


def test_speedup():
    assert compute_speedup(10.0, 2.5) == pytest.approx(4.0)


def test_efficiency():
    assert compute_efficiency(4.0, 8) == pytest.approx(0.5)
