import cv2
import numpy as np
from src.config import (
    CANNY_LOW, CANNY_HIGH, BILATERAL_D, BILATERAL_SIGMA, NLM_H,
    MORPH_KERNEL_A, MORPH_KERNEL_B,
)


def process_tile_cpu_pipeline_a(tile_bgr: np.ndarray) -> np.ndarray:
    """Edge and gradient pipeline. Heterogeneous compute time."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    mag_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    combined = cv2.addWeighted(mag_u8, 0.5, edges, 0.5, 0)
    kernel = np.ones((MORPH_KERNEL_A, MORPH_KERNEL_A), np.uint8)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


def process_tile_cpu_pipeline_b(tile_bgr: np.ndarray) -> np.ndarray:
    """Denoise and threshold pipeline. More uniform compute time."""
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    denoised = cv2.fastNlMeansDenoising(bilateral, None, h=NLM_H)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    kernel = np.ones((MORPH_KERNEL_B, MORPH_KERNEL_B), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(opened, kernel, iterations=1)


PIPELINES_CPU = {
    "edge_gradient": process_tile_cpu_pipeline_a,
    "denoise_threshold": process_tile_cpu_pipeline_b,
}
