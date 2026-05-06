"""Train all 11 predictor models, evaluate, save best + comparison CSV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import LOGS_DIR, MODELS_DIR
from src.features import FEATURE_NAMES
from src.predictor import TileComplexityPredictor


if __name__ == "__main__":
    print("Loading tile profiles...")
    df = pd.read_csv(LOGS_DIR / "tile_profiles.csv")
    print(f"  {len(df)} tiles from {df['image_id'].nunique()} images")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["runtime_ms"].values.astype(np.float32)
    groups = df["image_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    predictor = TileComplexityPredictor()

    print("Training tabular models...")
    predictor.fit_tabular(X_train, y_train)
    tabular_metrics = predictor.evaluate_tabular(X_test, y_test)

    print("\n=== TABULAR MODEL COMPARISON ===")
    tab_rows = []
    for model_name, m in tabular_metrics.items():
        row = {"model": model_name, "type": "tabular", **m, "inference_ms": 0.0}
        tab_rows.append(row)
        print(f"  {model_name:20s} MAE={m['mae']:.3f}ms  R2={m['r2']:.3f}  Spearman={m['spearman']:.3f}")
    print(f"\nBest tabular model (Spearman): {predictor.best_tabular_model}")

    # CNN models
    print("\nTraining CNN models (takes several minutes)...")
    from src.dataset import iter_dataset
    from src.tiling import extract_tiles
    from src.config import TILE_SIZE, HALO

    tile_imgs = []
    tile_runtimes = []
    profile_lookup = {}
    for _, row in df.iterrows():
        key = f"{row['image_id']}_{int(row['tile_id'])}"
        profile_lookup[key] = row["runtime_ms"]

    for img_name, image in iter_dataset("kodak"):
        tiles = extract_tiles(image, TILE_SIZE, HALO)
        for tile in tiles:
            key = f"kodak_{img_name}_{tile.tile_id}"
            if key in profile_lookup:
                tile_imgs.append(tile.data)
                tile_runtimes.append(profile_lookup[key])

    cnn_rows = []
    if len(tile_imgs) >= 20:
        n_cnn = min(len(tile_imgs), 400)
        tile_imgs = tile_imgs[:n_cnn]
        tile_runtimes_arr = np.array(tile_runtimes[:n_cnn], dtype=np.float32)

        n_train_cnn = int(len(tile_imgs) * 0.8)
        predictor.fit_cnn(tile_imgs[:n_train_cnn], tile_runtimes_arr[:n_train_cnn])
        cnn_metrics = predictor.evaluate_cnn(tile_imgs[n_train_cnn:], tile_runtimes_arr[n_train_cnn:])
        inference_times = predictor.measure_cnn_inference_time(tile_imgs[0])

        print("\n=== CNN MODEL COMPARISON ===")
        for model_name, m in cnn_metrics.items():
            inf_ms = inference_times.get(model_name, 0.0)
            row = {"model": model_name, "type": "cnn", "inference_ms": inf_ms, **m}
            cnn_rows.append(row)
            print(f"  {model_name:25s} MAE={m['mae']:.3f}ms  Spearman={m['spearman']:.3f}  Inference={inf_ms:.2f}ms")
    else:
        print("  Not enough tiles for CNN training. Run 02_profile_tiles.py first.")

    all_rows = tab_rows + cnn_rows
    metrics_df = pd.DataFrame(all_rows)
    out_path = LOGS_DIR / "predictor_comparison.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\nSaved model comparison to {out_path}")

    predictor.save(MODELS_DIR / "predictor.pkl")
    print(f"Saved predictor to {MODELS_DIR / 'predictor.pkl'}")
    print(f"Best tabular: {predictor.best_tabular_model}")
    print(f"Best CNN: {predictor.best_cnn_model}")
