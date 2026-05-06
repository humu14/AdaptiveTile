import numpy as np
import pytest
from src.worker_cpu import process_tile_cpu_pipeline_a, process_tile_cpu_pipeline_b


def make_bgr_tile(h=64, w=64):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_pipeline_a_output_shape():
    tile = make_bgr_tile()
    result = process_tile_cpu_pipeline_a(tile)
    assert result.shape == (64, 64)


def test_pipeline_a_output_dtype():
    tile = make_bgr_tile()
    result = process_tile_cpu_pipeline_a(tile)
    assert result.dtype == np.uint8


def test_pipeline_b_output_shape():
    tile = make_bgr_tile()
    result = process_tile_cpu_pipeline_b(tile)
    assert result.shape == (64, 64)


def test_pipeline_b_output_dtype():
    tile = make_bgr_tile()
    result = process_tile_cpu_pipeline_b(tile)
    assert result.dtype == np.uint8


def test_pipeline_a_non_trivial():
    # Uniform images produce no edges; use inputs with different edge content
    uniform = np.zeros((64, 64, 3), dtype=np.uint8)
    # Half-black half-white: strong vertical edge at centre
    edgy = np.zeros((64, 64, 3), dtype=np.uint8)
    edgy[:, 32:] = 255
    r_uniform = process_tile_cpu_pipeline_a(uniform)
    r_edgy = process_tile_cpu_pipeline_a(edgy)
    assert not np.array_equal(r_uniform, r_edgy)
