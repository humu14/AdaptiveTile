import numpy as np
import pytest
import torch

GPU_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_pipeline_a_output_shape():
    from src.worker_gpu import process_tile_gpu_pipeline_a
    tile = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = process_tile_gpu_pipeline_a(tile)
    assert result.shape == (64, 64)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_pipeline_b_output_shape():
    from src.worker_gpu import process_tile_gpu_pipeline_b
    tile = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = process_tile_gpu_pipeline_b(tile)
    assert result.shape == (64, 64)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA not available")
def test_gpu_pipeline_a_output_dtype():
    from src.worker_gpu import process_tile_gpu_pipeline_a
    tile = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = process_tile_gpu_pipeline_a(tile)
    assert result.dtype == np.uint8
