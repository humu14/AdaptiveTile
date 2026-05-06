"""Collect tile runtime training data for ML predictor."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import TILE_SIZE, HALO, LOGS_DIR, PREDICTOR_TRAIN_REPEATS
from src.dataset import iter_dataset
from src.tiling import extract_tiles
from src.features import extract_features, FEATURE_NAMES
from src.worker_cpu import process_tile_cpu_pipeline_a


def profile_tile(tile_data: np.ndarray, n_repeats: int) -> float:
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        process_tile_cpu_pipeline_a(tile_data)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000  # ms


if __name__ == "__main__":
    records = []
    datasets = [("kodak", None)]  # All 24 Kodak images

    for dataset_name, max_imgs in datasets:
        try:
            for img_name, image in tqdm(
                iter_dataset(dataset_name, max_images=max_imgs),
                desc=f"Profiling {dataset_name}",
            ):
                tiles = extract_tiles(image, TILE_SIZE, HALO)
                for tile in tiles:
                    feats = extract_features(tile.data, tile.row, tile.col)
                    runtime_ms = profile_tile(tile.data, PREDICTOR_TRAIN_REPEATS)
                    record = {
                        "image_id": f"{dataset_name}_{img_name}",
                        "tile_id": tile.tile_id,
                        **feats,
                        "runtime_ms": runtime_ms,
                    }
                    records.append(record)
        except Exception as e:
            print(f"Skipping {dataset_name}: {e}")

    df = pd.DataFrame(records)
    out_path = LOGS_DIR / "tile_profiles.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} tile profiles to {out_path}")
    print(df[["runtime_ms"] + FEATURE_NAMES].describe())
