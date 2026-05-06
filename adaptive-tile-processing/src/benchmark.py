"""Unified benchmark runner for all 4 execution modes."""
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.config import NUM_WORKERS, TILE_SIZE, HALO, DEVICE, NUM_GPU_STREAMS
from src.tiling import extract_tiles, TileInfo
from src.merge import merge_tiles
from src.worker_cpu import PIPELINES_CPU
from src.scheduler import StaticScheduler, DynamicScheduler, PredictiveScheduler, HybridScheduler
from src.metrics import compute_psnr, compute_ssim, compute_imbalance_ratio, compute_speedup, compute_efficiency


@dataclass
class BenchmarkResult:
    mode: str
    pipeline: str
    dataset: str
    image_id: str
    n_workers: int
    tile_size: int
    halo: int
    scheduler: str
    total_time_s: float
    speedup: float = 0.0
    efficiency: float = 0.0
    imbalance_ratio: float = 1.0
    psnr: float = 0.0
    ssim: float = 0.0
    n_tiles: int = 0
    worker_times: list = field(default_factory=list)


def _worker_task(args):
    tile_data, pipeline_name = args
    t0 = time.perf_counter()
    from src.worker_cpu import PIPELINES_CPU
    result = PIPELINES_CPU[pipeline_name](tile_data)
    return result, time.perf_counter() - t0


def run_cpu_serial(
    image: np.ndarray,
    pipeline: str = "edge_gradient",
    tile_size: int = TILE_SIZE,
    halo: int = HALO,
    image_id: str = "",
    dataset: str = "",
) -> tuple:
    """Returns (BenchmarkResult, output_image)."""
    fn = PIPELINES_CPU[pipeline]
    tiles = extract_tiles(image, tile_size, halo)

    t0 = time.perf_counter()
    results = []
    for tile in tiles:
        out = fn(tile.data)
        results.append((tile, out))
    elapsed = time.perf_counter() - t0

    output = merge_tiles(results, image.shape[:2], halo)

    return BenchmarkResult(
        mode="cpu_serial", pipeline=pipeline, dataset=dataset, image_id=image_id,
        n_workers=1, tile_size=tile_size, halo=halo, scheduler="serial",
        total_time_s=elapsed, speedup=1.0, efficiency=1.0,
        imbalance_ratio=1.0, n_tiles=len(tiles),
    ), output


def run_cpu_parallel(
    image: np.ndarray,
    serial_time: float,
    serial_output: np.ndarray,
    pipeline: str = "edge_gradient",
    scheduler_name: str = "dynamic",
    n_workers: int = NUM_WORKERS,
    tile_size: int = TILE_SIZE,
    halo: int = HALO,
    predictor=None,
    image_id: str = "",
    dataset: str = "",
    executor: Optional[ProcessPoolExecutor] = None,
) -> "BenchmarkResult":
    from src.features import extract_features, features_to_array
    tiles = extract_tiles(image, tile_size, halo)

    hybrid_sched = None
    pred_map = {}

    if scheduler_name == "static":
        sched = StaticScheduler()
        batches = sched.assign(tiles, n_workers)
        ordered_tiles = [t for batch in batches for t in batch]
    elif scheduler_name == "dynamic":
        ordered_tiles = list(tiles)
    elif scheduler_name in ("predictive", "hybrid"):
        if predictor is not None:
            feats = np.array([
                features_to_array(extract_features(t.data, t.row, t.col))
                for t in tiles
            ])
            preds = predictor.predict(feats)
            if scheduler_name == "predictive":
                sched = PredictiveScheduler()
                ordered_tiles = sched.get_queue(tiles, preds)
            else:
                hybrid_sched = HybridScheduler()
                ordered_tiles = hybrid_sched.get_initial_queue(tiles, preds)
                pred_map = {t.tile_id: float(preds[i]) for i, t in enumerate(tiles)}
        else:
            ordered_tiles = list(tiles)
    else:
        ordered_tiles = list(tiles)

    tile_results = {}
    worker_times = []
    t0 = time.perf_counter()

    def _run_with_executor(ex):
        futures = {
            ex.submit(_worker_task, (t.data, pipeline)): t
            for t in ordered_tiles
        }
        for fut in as_completed(futures):
            tile = futures[fut]
            out, elapsed = fut.result()
            tile_results[tile.tile_id] = (tile, out)
            worker_times.append(elapsed)
            if hybrid_sched and tile.tile_id in pred_map:
                hybrid_sched.record_completion(
                    tile.tile_id, pred_map[tile.tile_id], elapsed * 1000
                )

    if executor is not None:
        _run_with_executor(executor)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            _run_with_executor(ex)

    total_time = time.perf_counter() - t0

    results_ordered = [tile_results[i] for i in sorted(tile_results.keys())]
    output = merge_tiles(results_ordered, image.shape[:2], halo)

    psnr = compute_psnr(serial_output, output)
    ssim_val = compute_ssim(serial_output, output)
    speedup = compute_speedup(serial_time, total_time)
    efficiency = compute_efficiency(speedup, n_workers)
    imbalance = compute_imbalance_ratio(worker_times) if worker_times else 1.0

    return BenchmarkResult(
        mode="cpu_parallel", pipeline=pipeline, dataset=dataset, image_id=image_id,
        n_workers=n_workers, tile_size=tile_size, halo=halo, scheduler=scheduler_name,
        total_time_s=total_time, speedup=speedup, efficiency=efficiency,
        imbalance_ratio=imbalance, psnr=psnr, ssim=ssim_val,
        n_tiles=len(tiles), worker_times=worker_times,
    )


def run_gpu_benchmark(
    image: np.ndarray,
    mode: str,
    serial_time: float,
    serial_output: np.ndarray,
    pipeline: str = "edge_gradient",
    n_streams: int = NUM_GPU_STREAMS,
    tile_size: int = TILE_SIZE,
    halo: int = HALO,
    image_id: str = "",
    dataset: str = "",
) -> "BenchmarkResult":
    """Run GPU serial or parallel benchmark. mode = 'gpu_serial' or 'gpu_parallel'."""
    from src.worker_gpu import run_gpu_serial as _gpu_serial, run_gpu_parallel as _gpu_parallel

    if mode == "gpu_serial":
        output, elapsed = _gpu_serial(image, pipeline, tile_size, halo)
    elif mode == "gpu_parallel":
        output, elapsed = _gpu_parallel(image, pipeline, tile_size, halo, n_streams)
    else:
        raise ValueError(f"Unknown GPU mode: {mode}")

    n_tiles = len(extract_tiles(image, tile_size, halo))
    speedup = compute_speedup(serial_time, elapsed)
    efficiency = compute_efficiency(speedup, n_streams if mode == "gpu_parallel" else 1)

    psnr = compute_psnr(serial_output, output) if serial_output is not None else 0.0
    ssim_val = compute_ssim(serial_output, output) if serial_output is not None else 0.0

    return BenchmarkResult(
        mode=mode, pipeline=pipeline, dataset=dataset, image_id=image_id,
        n_workers=n_streams if mode == "gpu_parallel" else 1,
        tile_size=tile_size, halo=halo, scheduler="streams",
        total_time_s=elapsed, speedup=speedup, efficiency=efficiency,
        imbalance_ratio=1.0, psnr=psnr, ssim=ssim_val, n_tiles=n_tiles,
    )


if __name__ == "__main__":
    pass
