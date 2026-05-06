import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage.feature import local_binary_pattern
from src.config import CANNY_LOW, CANNY_HIGH

FEATURE_NAMES = [
    "edge_density",
    "gradient_variance",
    "intensity_variance",
    "histogram_entropy",
    "lbp_texture_score",
    "tile_row",
    "tile_col",
]


def extract_features(tile_bgr: np.ndarray, row: int, col: int) -> dict:
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    edges = cv2.Canny(gray.astype(np.uint8), CANNY_LOW, CANNY_HIGH)
    edge_density = float(edges.mean() / 255.0)

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sx**2 + sy**2)
    gradient_variance = float(magnitude.var())

    intensity_variance = float(gray.var())

    hist, _ = np.histogram(gray, bins=16, range=(0, 256))
    hist_prob = hist / hist.sum()
    histogram_entropy = float(scipy_entropy(hist_prob + 1e-10))

    lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method="uniform")
    lbp_texture_score = float(lbp.mean())

    return {
        "edge_density": edge_density,
        "gradient_variance": gradient_variance,
        "intensity_variance": intensity_variance,
        "histogram_entropy": histogram_entropy,
        "lbp_texture_score": lbp_texture_score,
        "tile_row": float(row),
        "tile_col": float(col),
    }


def features_to_array(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
