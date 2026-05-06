"""
Retrain full predictor pipeline on preprocessed tile data.

Compares raw vs preprocessed performance.
Uses GroupShuffleSplit + 5-fold GroupKFold CV.
Saves:
  outputs/models/predictor_preprocessed.pkl
  outputs/logs/predictor_comparison_preprocessed.csv
  outputs/plots/training_*.svg
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.config import LOGS_DIR, MODELS_DIR, PLOTS_DIR
from src.features import FEATURE_NAMES
from src.predictor import TileComplexityPredictor, TABULAR_MODELS

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
N_CV_FOLDS = 5


def savefig(name: str):
    # Save SVG to plots dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = PLOTS_DIR / f"training_{name}.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=150)
    
    # Save PNG to figures dir
    FIGS_DIR = PLOTS_DIR.parent / "figures"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGS_DIR / f"training_{name}.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    
    plt.close()
    return svg_path


def evaluate_split(predictor, X_test, y_test):
    """Evaluate all trained tabular models on test set."""
    return predictor.evaluate_tabular(X_test, y_test)


def cross_validate_model(model, X, y, groups, n_splits=N_CV_FOLDS):
    """GroupKFold CV — keeps images intact across folds."""
    gkf = GroupKFold(n_splits=n_splits)
    maes, rmses, r2s, spearmans = [], [], [], []
    from copy import deepcopy
    for train_idx, val_idx in gkf.split(X, y, groups):
        m = deepcopy(model)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], preds))
        rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))
        r2s.append(r2_score(y[val_idx], preds))
        spearmans.append(spearmanr(y[val_idx], preds).correlation)
    return {
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "spearman_mean": np.mean(spearmans), "spearman_std": np.std(spearmans),
    }


# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("TRAINING ON PREPROCESSED DATA")
print("=" * 65)

df_raw = pd.read_csv(LOGS_DIR / "tile_profiles.csv")
df_pre = pd.read_csv(LOGS_DIR / "tile_profiles_preprocessed.csv")

print(f"Raw:          {len(df_raw):,} tiles from {df_raw['image_id'].nunique()} images")
print(f"Preprocessed: {len(df_pre):,} tiles from {df_pre['image_id'].nunique()} images")

feature_cols = FEATURE_NAMES  # same names, values differ after preprocessing

X_raw = df_raw[feature_cols].values.astype(np.float32)
y_raw = df_raw["runtime_ms"].values.astype(np.float32)
groups_raw = df_raw["image_id"].values

X_pre = df_pre[feature_cols].values.astype(np.float32)
y_pre = df_pre["runtime_ms"].values.astype(np.float32)
groups_pre = df_pre["image_id"].values

# Train/test split — consistent method
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_r, test_r = next(gss.split(X_raw, y_raw, groups_raw))
X_train_r, X_test_r = X_raw[train_r], X_raw[test_r]
y_train_r, y_test_r = y_raw[train_r], y_raw[test_r]

train_p, test_p = next(gss.split(X_pre, y_pre, groups_pre))
X_train_p, X_test_p = X_pre[train_p], X_pre[test_p]
y_train_p, y_test_p = y_pre[train_p], y_pre[test_p]

print(f"\nRaw    — Train: {len(train_r):,} | Test: {len(test_r):,}")
print(f"Prepro — Train: {len(train_p):,} | Test: {len(test_p):,}")


# ── Train models ─────────────────────────────────────────────────────────────
print("\n── Training tabular models on PREPROCESSED data ──")
pred_pre = TileComplexityPredictor()
pred_pre.fit_tabular(X_train_p, y_train_p)
metrics_pre = pred_pre.evaluate_tabular(X_test_p, y_test_p)

print("\n── Training tabular models on RAW data (baseline) ──")
pred_raw = TileComplexityPredictor()
pred_raw.fit_tabular(X_train_r, y_train_r)
metrics_raw = pred_raw.evaluate_tabular(X_test_r, y_test_r)


# ── Cross-validation ─────────────────────────────────────────────────────────
print(f"\n── {N_CV_FOLDS}-fold GroupKFold CV on preprocessed data ──")
cv_results = {}
for name, model in TABULAR_MODELS.items():
    print(f"  CV: {name}...", end=" ", flush=True)
    cv = cross_validate_model(model, X_pre, y_pre, groups_pre, n_splits=N_CV_FOLDS)
    cv_results[name] = cv
    print(f"MAE={cv['mae_mean']:.3f}±{cv['mae_std']:.3f}  "
          f"R²={cv['r2_mean']:.3f}±{cv['r2_std']:.3f}  "
          f"Spearman={cv['spearman_mean']:.3f}±{cv['spearman_std']:.3f}")


# ── Build comparison table ────────────────────────────────────────────────────
print("\n── Comparison: RAW vs. PREPROCESSED (hold-out test set) ──")
print(f"{'Model':<22} {'RAW_MAE':>9} {'PRE_MAE':>9} {'Δ_MAE':>8} "
      f"{'RAW_R2':>8} {'PRE_R2':>8} {'RAW_Sp':>8} {'PRE_Sp':>8}")
print("-" * 90)

rows = []
for name in metrics_raw:
    r = metrics_raw[name]
    p = metrics_pre.get(name, {})
    delta_mae = p.get("mae", np.nan) - r["mae"]
    cv = cv_results.get(name, {})
    row = {
        "model": name,
        "raw_mae": r["mae"], "pre_mae": p.get("mae", np.nan),
        "delta_mae": delta_mae,
        "raw_rmse": r["rmse"], "pre_rmse": p.get("rmse", np.nan),
        "raw_r2": r["r2"], "pre_r2": p.get("r2", np.nan),
        "raw_spearman": r["spearman"], "pre_spearman": p.get("spearman", np.nan),
        "cv_mae_mean": cv.get("mae_mean", np.nan),
        "cv_mae_std": cv.get("mae_std", np.nan),
        "cv_r2_mean": cv.get("r2_mean", np.nan),
        "cv_r2_std": cv.get("r2_std", np.nan),
        "cv_spearman_mean": cv.get("spearman_mean", np.nan),
        "cv_spearman_std": cv.get("spearman_std", np.nan),
    }
    rows.append(row)
    print(f"{name:<22} {r['mae']:>9.4f} {p.get('mae', np.nan):>9.4f} {delta_mae:>+8.4f} "
          f"{r['r2']:>8.3f} {p.get('r2', np.nan):>8.3f} "
          f"{r['spearman']:>8.3f} {p.get('spearman', np.nan):>8.3f}")

results_df = pd.DataFrame(rows)
out_csv = LOGS_DIR / "predictor_comparison_preprocessed.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved comparison CSV: {out_csv}")


# ── Visualizations ────────────────────────────────────────────────────────────

# 1. MAE comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x = np.arange(len(results_df))
width = 0.35

axes[0].bar(x - width/2, results_df["raw_mae"], width, label="Raw", color="#4472C4")
axes[0].bar(x + width/2, results_df["pre_mae"], width, label="Preprocessed", color="#70AD47")
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("MAE (ms)")
axes[0].set_title("MAE: Raw vs. Preprocessed")
axes[0].legend()

axes[1].bar(x - width/2, results_df["raw_r2"], width, label="Raw", color="#4472C4")
axes[1].bar(x + width/2, results_df["pre_r2"], width, label="Preprocessed", color="#70AD47")
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("R²")
axes[1].set_title("R²: Raw vs. Preprocessed")
axes[1].legend()

axes[2].bar(x - width/2, results_df["raw_spearman"], width, label="Raw", color="#4472C4")
axes[2].bar(x + width/2, results_df["pre_spearman"], width, label="Preprocessed", color="#70AD47")
axes[2].set_xticks(x)
axes[2].set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
axes[2].set_ylabel("Spearman ρ")
axes[2].set_title("Spearman: Raw vs. Preprocessed")
axes[2].legend()

plt.suptitle("Model Performance: Raw vs. Preprocessed Data", fontsize=13)
plt.tight_layout()
savefig("01_raw_vs_preprocessed")
print("Saved: training_01_raw_vs_preprocessed.svg")


# 2. CV results (mean ± std)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_cv = [("cv_mae_mean", "cv_mae_std", "MAE (ms)"),
              ("cv_r2_mean", "cv_r2_std", "R²"),
              ("cv_spearman_mean", "cv_spearman_std", "Spearman ρ")]

for ax, (mean_col, std_col, ylabel) in zip(axes, metrics_cv):
    ax.bar(x, results_df[mean_col], color="#4472C4",
           yerr=results_df[std_col], capsize=4, ecolor="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"CV {ylabel} (±std, {N_CV_FOLDS}-fold)")

plt.suptitle(f"{N_CV_FOLDS}-fold GroupKFold Cross-Validation on Preprocessed Data", fontsize=13)
plt.tight_layout()
savefig("02_cv_results")
print("Saved: training_02_cv_results.svg")


# 3. Predicted vs. actual scatter — best model
best_model_name = results_df.loc[results_df["pre_spearman"].idxmax(), "model"]
print(f"\nBest preprocessed model (Spearman): {best_model_name}")
best_preds = pred_pre._tabular_models[best_model_name].predict(X_test_p)

# Predicted vs. actual scatter (2x1 grid)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].scatter(y_test_p, best_preds, alpha=0.5, s=15, color="#4472C4")
lim = [min(y_test_p.min(), best_preds.min()), max(y_test_p.max(), best_preds.max())]
axes[0].plot(lim, lim, "r--", lw=1.5, label="Perfect")
axes[0].set_title(f"{best_model_name}: Predicted vs. Actual")
residuals = y_test_p - best_preds
axes[1].scatter(best_preds, residuals, alpha=0.5, s=15, color="#ED7D31")
axes[1].axhline(0, color="k", lw=1.5, ls="--")
axes[1].set_title(f"{best_model_name}: Residuals")
plt.tight_layout()
savefig("03_predicted_vs_actual")
print("Saved: training_03_predicted_vs_actual.svg")


# 4. Residual distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(residuals, bins=30, edgecolor="k", color="#4472C4")
axes[0].set_xlabel("Residual (ms)")
axes[0].set_title(f"{best_model_name}: Residual Distribution")

import scipy.stats as sc_stats
sc_stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Residual Q-Q Plot")
plt.tight_layout()
savefig("04_residuals")
print("Saved: training_04_residuals.svg")


# 5. Feature importance (tree models)
# Feature importance (2x1 grid)
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
for ax, model_name in zip(axes, ["random_forest", "xgboost"]):
    m = pred_pre._tabular_models.get(model_name)
    if m is not None:
        sub = m.named_steps.get("model", None) if hasattr(m, "named_steps") else m
        imp = getattr(sub, "feature_importances_", None)
        if imp is not None:
            sorted_idx = np.argsort(imp)[::-1]
            ax.bar(range(len(feature_cols)), imp[sorted_idx], color="#4472C4")
            ax.set_xticks(range(len(feature_cols)))
            ax.set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{model_name} Feature Importance")
plt.tight_layout()
savefig("05_feature_importance")
print("Saved: training_05_feature_importance.svg")


# ── Save best model ───────────────────────────────────────────────────────────
pred_pre.save(MODELS_DIR / "predictor_preprocessed.pkl")
print(f"\nSaved model: {MODELS_DIR / 'predictor_preprocessed.pkl'}")
print(f"Best tabular model (preprocessed): {pred_pre.best_tabular_model}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
best_row = results_df.loc[results_df["pre_spearman"].idxmax()]
print(f"Best model on preprocessed data: {best_row['model']}")
print(f"  MAE:      {best_row['pre_mae']:.4f} ms  (raw: {best_row['raw_mae']:.4f} ms)")
print(f"  RMSE:     {best_row['pre_rmse']:.4f} ms  (raw: {best_row['raw_rmse']:.4f} ms)")
print(f"  R²:       {best_row['pre_r2']:.4f}     (raw: {best_row['raw_r2']:.4f})")
print(f"  Spearman: {best_row['pre_spearman']:.4f}     (raw: {best_row['raw_spearman']:.4f})")
cv_best = cv_results.get(best_row["model"], {})
if cv_best:
    print(f"  CV MAE:   {cv_best['mae_mean']:.4f}±{cv_best['mae_std']:.4f} ms")
    print(f"  CV R²:    {cv_best['r2_mean']:.4f}±{cv_best['r2_std']:.4f}")
print("=" * 65)



