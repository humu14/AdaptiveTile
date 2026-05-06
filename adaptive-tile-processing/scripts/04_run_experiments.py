"""Run all 8 benchmark experiments. Saves per-run CSV rows to outputs/logs/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor

from src.config import (
    TILE_SIZE, HALO, NUM_WORKERS, LOGS_DIR, MODELS_DIR,
    TILE_SIZES, HALOS, WORKER_COUNTS, DEVICE,
    PIPELINE_A, PIPELINE_B,
)
from src.dataset import iter_dataset
from src.benchmark import run_cpu_serial, run_cpu_parallel, run_gpu_benchmark
from src.predictor import TileComplexityPredictor

# Pipeline B (denoise_threshold: bilateral+NLM) is the primary benchmark pipeline.
# Its per-tile compute time (>10ms) far exceeds IPC overhead (~1ms), enabling
# meaningful parallel speedup. Pipeline A (Canny edges, ~0.3ms/tile) is
# compared directly in Exp E.
BENCH = PIPELINE_B


def load_predictor():
    path = MODELS_DIR / "predictor.pkl"
    if path.exists():
        return TileComplexityPredictor.load(path)
    print("WARNING: predictor.pkl not found. Predictive/hybrid schedulers use unordered tiles.")
    return None


def result_to_row(r, **extra) -> dict:
    return {
        "mode": r.mode, "pipeline": r.pipeline, "dataset": r.dataset,
        "image_id": r.image_id, "n_workers": r.n_workers, "tile_size": r.tile_size,
        "halo": r.halo, "scheduler": r.scheduler, "total_time_s": r.total_time_s,
        "speedup": r.speedup, "efficiency": r.efficiency,
        "imbalance_ratio": r.imbalance_ratio, "psnr": r.psnr, "ssim": r.ssim,
        "n_tiles": r.n_tiles, **extra,
    }


if __name__ == "__main__":
    predictor = load_predictor()
    rows = []

    # Pre-warm pools for each worker count to eliminate spawn overhead from timing
    print("Pre-warming process pools...")
    all_counts = sorted(set(WORKER_COUNTS) | {NUM_WORKERS})
    pools = {nw: ProcessPoolExecutor(max_workers=nw) for nw in all_counts}
    default_pool = pools[NUM_WORKERS]

    # ── Experiment A: Speedup vs worker count ────────────────────────────────
    print("\n=== Exp A: Speedup vs workers ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=6)), desc="ExpA"):
        serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, HALO, img_id, "kodak")
        rows.append(result_to_row(serial_r, experiment="A"))
        for nw in WORKER_COUNTS:
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, BENCH, "dynamic",
                nw, TILE_SIZE, HALO, predictor, img_id, "kodak",
                executor=pools[nw],
            )
            rows.append(result_to_row(r, experiment="A"))

    # ── Experiment B: Scheduler comparison ───────────────────────────────────
    print("\n=== Exp B: Scheduler comparison ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=8)), desc="ExpB"):
        serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, HALO, img_id, "kodak")
        rows.append(result_to_row(serial_r, experiment="B"))
        for sched in ["static", "dynamic", "predictive", "hybrid"]:
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, BENCH, sched,
                NUM_WORKERS, TILE_SIZE, HALO, predictor, img_id, "kodak",
                executor=default_pool,
            )
            rows.append(result_to_row(r, experiment="B"))

    # ── Experiment C: Tile size study ─────────────────────────────────────────
    print("\n=== Exp C: Tile size ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=4)), desc="ExpC"):
        for ts in TILE_SIZES:
            serial_r, serial_out = run_cpu_serial(image, BENCH, ts, HALO, img_id, "kodak")
            rows.append(result_to_row(serial_r, experiment="C"))
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, BENCH, "hybrid",
                NUM_WORKERS, ts, HALO, predictor, img_id, "kodak",
                executor=default_pool,
            )
            rows.append(result_to_row(r, experiment="C"))

    # ── Experiment D: Halo ablation ───────────────────────────────────────────
    print("\n=== Exp D: Halo ablation ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=4)), desc="ExpD"):
        for h in HALOS:
            serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, h, img_id, "kodak")
            rows.append(result_to_row(serial_r, experiment="D"))
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, BENCH, "hybrid",
                NUM_WORKERS, TILE_SIZE, h, predictor, img_id, "kodak",
                executor=default_pool,
            )
            rows.append(result_to_row(r, experiment="D"))

    # ── Experiment E: Pipeline A vs B ─────────────────────────────────────────
    print("\n=== Exp E: Pipeline comparison ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=8)), desc="ExpE"):
        for pipe in [PIPELINE_A, PIPELINE_B]:
            serial_r, serial_out = run_cpu_serial(image, pipe, TILE_SIZE, HALO, img_id, "kodak")
            rows.append(result_to_row(serial_r, experiment="E"))
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, pipe, "dynamic",
                NUM_WORKERS, TILE_SIZE, HALO, predictor, img_id, "kodak",
                executor=default_pool,
            )
            rows.append(result_to_row(r, experiment="E"))

    # ── Experiment F: Dataset generalization ──────────────────────────────────
    print("\n=== Exp F: Dataset generalization ===")
    for ds, max_imgs in [("kodak", 8), ("div2k", 5), ("bsds500", 10)]:
        try:
            for img_id, image in tqdm(list(iter_dataset(ds, max_images=max_imgs)), desc=f"ExpF-{ds}"):
                serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, HALO, img_id, ds)
                rows.append(result_to_row(serial_r, experiment="F"))
                r = run_cpu_parallel(
                    image, serial_r.total_time_s, serial_out, BENCH, "hybrid",
                    NUM_WORKERS, TILE_SIZE, HALO, predictor, img_id, ds,
                    executor=default_pool,
                )
                rows.append(result_to_row(r, experiment="F"))
        except Exception as e:
            print(f"  Skipping {ds}: {e}")

    # ── Experiment G: 4-mode comparison ──────────────────────────────────────
    print("\n=== Exp G: 4-mode comparison ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=8)), desc="ExpG"):
        serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, HALO, img_id, "kodak")
        rows.append(result_to_row(serial_r, experiment="G"))
        r_cpu_par = run_cpu_parallel(
            image, serial_r.total_time_s, serial_out, BENCH, "hybrid",
            NUM_WORKERS, TILE_SIZE, HALO, predictor, img_id, "kodak",
            executor=default_pool,
        )
        rows.append(result_to_row(r_cpu_par, experiment="G"))
        if DEVICE == "cuda":
            for gpu_mode in ["gpu_serial", "gpu_parallel"]:
                r_gpu = run_gpu_benchmark(
                    image, gpu_mode, serial_r.total_time_s, serial_out,
                    BENCH, 4, TILE_SIZE, HALO, img_id, "kodak",
                )
                rows.append(result_to_row(r_gpu, experiment="G"))

    # ── Experiment H: ML predictor vs heuristic ───────────────────────────────
    print("\n=== Exp H: Predictor vs dynamic (imbalance comparison) ===")
    for img_id, image in tqdm(list(iter_dataset("kodak", max_images=8)), desc="ExpH"):
        serial_r, serial_out = run_cpu_serial(image, BENCH, TILE_SIZE, HALO, img_id, "kodak")
        rows.append(result_to_row(serial_r, experiment="H"))
        for sched in ["predictive", "dynamic", "hybrid"]:
            r = run_cpu_parallel(
                image, serial_r.total_time_s, serial_out, BENCH, sched,
                NUM_WORKERS, TILE_SIZE, HALO, predictor, img_id, "kodak",
                executor=default_pool,
            )
            rows.append(result_to_row(r, experiment="H"))

    for nw, pool in pools.items():
        pool.shutdown(wait=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_path = LOGS_DIR / "experiment_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print("\nMean speedup by mode:")
    print(df.groupby(["experiment", "mode"])["speedup"].mean().to_string())
