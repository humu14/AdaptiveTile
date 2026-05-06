from pathlib import Path
from typing import Iterator, Tuple
import cv2
import numpy as np
from src.config import KODAK_DIR, DIV2K_DIR, BSDS500_DIR


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def iter_dataset(dataset: str, max_images: int = None) -> Iterator[Tuple[str, np.ndarray]]:
    dirs = {
        "kodak": KODAK_DIR,
        "div2k": DIV2K_DIR / "valid_HR",
        "bsds500": BSDS500_DIR / "images" / "test",
    }
    if dataset not in dirs:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(dirs)}")
    base = dirs[dataset]
    paths = sorted(base.glob("*.png")) + sorted(base.glob("*.jpg"))
    if max_images:
        paths = paths[:max_images]
    for p in paths:
        yield p.stem, load_image(p)


def count_images(dataset: str) -> int:
    return sum(1 for _ in iter_dataset(dataset))
