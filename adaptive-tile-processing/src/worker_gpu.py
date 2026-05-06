"""GPU processing pipelines using PyTorch + Kornia."""
import time
import numpy as np
import torch
import kornia
from src.config import DEVICE, NUM_GPU_STREAMS, TILE_SIZE, HALO


def _to_tensor(tile_bgr: np.ndarray, device: str) -> torch.Tensor:
    """BGR uint8 -> [1, 3, H, W] float32 on device, values in [0,1]."""
    t = torch.from_numpy(tile_bgr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """[1, 1, H, W] or similar float32 -> HxW uint8 numpy."""
    return (tensor.squeeze().clamp(0, 1) * 255).byte().cpu().numpy()


def process_tile_gpu_pipeline_a(tile_bgr: np.ndarray, device: str = DEVICE) -> np.ndarray:
    """Edge and gradient pipeline on GPU using Kornia."""
    t = _to_tensor(tile_bgr, device)
    gray = kornia.color.bgr_to_grayscale(t)
    blurred = kornia.filters.gaussian_blur2d(gray, (5, 5), (1.0, 1.0))
    sobel = kornia.filters.sobel(blurred)
    _, edges = kornia.filters.canny(
        blurred, low_threshold=50.0 / 255.0, high_threshold=150.0 / 255.0
    )
    combined = 0.5 * sobel + 0.5 * edges.float()
    kernel = torch.ones(5, 5, device=device)
    closed = kornia.morphology.closing(combined, kernel)
    return _to_uint8(closed)


def process_tile_gpu_pipeline_b(tile_bgr: np.ndarray, device: str = DEVICE) -> np.ndarray:
    """Denoise and threshold pipeline on GPU (Kornia approximations)."""
    t = _to_tensor(tile_bgr, device)
    gray = kornia.color.bgr_to_grayscale(t)
    bilateral = kornia.filters.bilateral_blur(gray, (9, 9), 0.3, (9, 9))
    denoised = kornia.filters.gaussian_blur2d(bilateral, (7, 7), (2.0, 2.0))
    local_mean = kornia.filters.gaussian_blur2d(denoised, (11, 11), (3.5, 3.5))
    thresh = (denoised > local_mean - 2.0 / 255.0).float()
    kernel = torch.ones(3, 3, device=device)
    opened = kornia.morphology.opening(thresh, kernel)
    dilated = kornia.morphology.dilation(opened, kernel)
    return _to_uint8(dilated)


PIPELINES_GPU = {
    "edge_gradient": process_tile_gpu_pipeline_a,
    "denoise_threshold": process_tile_gpu_pipeline_b,
}


def run_gpu_serial(
    image: np.ndarray,
    pipeline: str = "edge_gradient",
    tile_size: int = TILE_SIZE,
    halo: int = HALO,
    device: str = DEVICE,
) -> tuple:
    """Process image tile-by-tile on GPU. Returns (output, elapsed_s)."""
    from src.tiling import extract_tiles
    from src.merge import merge_tiles
    fn = PIPELINES_GPU[pipeline]
    tiles = extract_tiles(image, tile_size, halo)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results = []
    for tile in tiles:
        out = fn(tile.data, device)
        results.append((tile, out))

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output = merge_tiles(results, image.shape[:2], halo)
    return output, elapsed


def run_gpu_parallel(
    image: np.ndarray,
    pipeline: str = "edge_gradient",
    tile_size: int = TILE_SIZE,
    halo: int = HALO,
    n_streams: int = NUM_GPU_STREAMS,
    device: str = DEVICE,
) -> tuple:
    """Process tiles concurrently using CUDA streams. Returns (output, elapsed_s)."""
    from src.tiling import extract_tiles
    from src.merge import merge_tiles
    fn = PIPELINES_GPU[pipeline]
    tiles = extract_tiles(image, tile_size, halo)
    results = [None] * len(tiles)

    if device != "cuda":
        return run_gpu_serial(image, pipeline, tile_size, halo, device)

    streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i, tile in enumerate(tiles):
        stream = streams[i % n_streams]
        with torch.cuda.stream(stream):
            results[i] = (tile, fn(tile.data, device))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    output = merge_tiles(results, image.shape[:2], halo)
    return output, elapsed
