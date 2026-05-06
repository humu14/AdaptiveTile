import numpy as np
import pytest
from src.features import extract_features, FEATURE_NAMES


def make_tile(h=64, w=64, fill=None):
    if fill is not None:
        return np.full((h, w, 3), fill, dtype=np.uint8)
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 7


def test_extract_returns_dict_with_all_keys():
    tile = make_tile()
    feats = extract_features(tile, row=0, col=0)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


def test_all_values_finite():
    tile = make_tile()
    feats = extract_features(tile, row=1, col=2)
    for name, val in feats.items():
        assert np.isfinite(val), f"Non-finite value for {name}: {val}"


def test_uniform_tile_has_zero_variance():
    tile = make_tile(fill=128)
    feats = extract_features(tile, row=0, col=0)
    assert feats["intensity_variance"] < 1e-6


def test_edge_dense_tile_has_high_edge_density():
    # Alternating 8-px vertical stripes create many Canny edges
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    for c in range(64):
        tile[:, c] = 255 if (c // 8) % 2 == 0 else 0
    feats = extract_features(tile, row=0, col=0)
    assert feats["edge_density"] > 0.1
