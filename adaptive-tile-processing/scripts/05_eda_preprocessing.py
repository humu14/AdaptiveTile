"""
EDA and data preprocessing for tile_profiles.csv.

Statistical decisions made via:
  - Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling (normality)
  - IQR, Modified Z-score (MAD), 3-sigma (outlier detection; majority vote)
  - Skewness / kurtosis (transformation decision)
  - Pearson / Spearman / Kendall (correlation)
  - Box-Cox / Yeo-Johnson (feature transformation)
  - Population Stability Index (PSI), Jensen-Shannon Divergence (JSD)

Outputs:
  outputs/plots/eda_*.svg
  outputs/logs/tile_profiles_preprocessed.csv
  outputs/logs/eda_report.txt
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, anderson,
    pearsonr, spearmanr, kendalltau,
    iqr, entropy as scipy_entropy,
)
from scipy.special import rel_entr
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import LOGS_DIR, PLOTS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "edge_density", "gradient_variance", "intensity_variance",
    "histogram_entropy", "lbp_texture_score", "tile_row", "tile_col",
]
TARGET = "runtime_ms"
ALPHA = 0.05
SKEW_THRESH = 0.5      # |skew| > threshold → consider transform
IQR_FENCE = 1.5        # IQR multiplier for outlier detection
MAD_FENCE = 3.5        # MAD-based modified z-score threshold
ZSCORE_FENCE = 3.0     # 3-sigma threshold
N_BINS_PSI = 10        # PSI bins

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LOGS_DIR / "eda_report.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def savefig(name: str):
    # Save SVG to plots dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = PLOTS_DIR / f"eda_{name}.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=150)
    
    # Save PNG to figures dir
    FIGS_DIR = PLOTS_DIR.parent / "figures"
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGS_DIR / f"eda_{name}.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    
    plt.close()
    return svg_path


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS_PSI) -> float:
    """Population Stability Index between two arrays."""
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=bins)
    e_pct = (e_counts / e_counts.sum()) + 1e-10
    a_pct = (a_counts / a_counts.sum()) + 1e-10
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def jsd(p: np.ndarray, q: np.ndarray, n_bins: int = 30) -> float:
    """Jensen-Shannon Divergence between two arrays."""
    bins = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    return float(0.5 * (scipy_entropy(p_hist, m) + scipy_entropy(q_hist, m)))


def iqr_outliers(s: pd.Series) -> np.ndarray:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr_val = q3 - q1
    return (s < q1 - IQR_FENCE * iqr_val) | (s > q3 + IQR_FENCE * iqr_val)


def mad_outliers(s: pd.Series) -> np.ndarray:
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return pd.Series([False] * len(s), index=s.index)
    modified_z = 0.6745 * (s - med) / mad
    return modified_z.abs() > MAD_FENCE


def zscore_outliers(s: pd.Series) -> np.ndarray:
    return (np.abs(stats.zscore(s)) > ZSCORE_FENCE)


def majority_outliers(s: pd.Series) -> np.ndarray:
    """Outlier if ≥ 2/3 methods agree."""
    a = iqr_outliers(s).astype(int)
    b = mad_outliers(s).astype(int)
    c = pd.Series(zscore_outliers(s).astype(int), index=s.index)
    return (a + b + c) >= 2


def normality_tests(s: pd.Series, name: str) -> dict:
    vals = s.dropna().values
    if len(vals) > 5000:
        vals = vals[:5000]  # Shapiro-Wilk max

    sw_stat, sw_p = shapiro(vals) if len(vals) <= 5000 else (np.nan, np.nan)
    dp_stat, dp_p = normaltest(vals)
    ad_result = anderson(vals, dist="norm")
    ad_critical = ad_result.critical_values[2]  # 5% significance
    ad_normal = ad_result.statistic < ad_critical

    return {
        "shapiro_stat": sw_stat, "shapiro_p": sw_p,
        "shapiro_normal": sw_p > ALPHA if not np.isnan(sw_p) else None,
        "dagostino_stat": dp_stat, "dagostino_p": dp_p,
        "dagostino_normal": dp_p > ALPHA,
        "anderson_stat": ad_result.statistic,
        "anderson_normal": ad_normal,
        "skewness": float(skew(vals)),
        "kurtosis": float(kurtosis(vals)),
        "majority_normal": int(sw_p > ALPHA if not np.isnan(sw_p) else 0)
                         + int(dp_p > ALPHA)
                         + int(ad_normal) >= 2,
    }


# ── Report writer ─────────────────────────────────────────────────────────────
report_lines = []

def report(msg: str = ""):
    report_lines.append(msg)
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Load & basic info
# ═══════════════════════════════════════════════════════════════════════════════
report("=" * 70)
report("EDA REPORT — tile_profiles.csv")
report("=" * 70)

df = pd.read_csv(LOGS_DIR / "tile_profiles.csv")
report(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} cols")
report(f"Images: {df['image_id'].nunique()} unique")
report(f"Tiles per image (mean): {df.groupby('image_id').size().mean():.1f}")

report("\n── Missing values ──")
missing = df.isnull().sum()
report(missing[missing > 0].to_string() if missing.any() else "None")

report("\n── Descriptive statistics ──")
report(df[FEATURE_COLS + [TARGET]].describe().round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Distribution plots (histogram + KDE + Q-Q)
# ═══════════════════════════════════════════════════════════════════════════════
# Histogram + KDE (3x3 grid)
all_cols = FEATURE_COLS + [TARGET]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(all_cols):
    vals = df[col].dropna().values
    axes[i].hist(vals, bins=25, color="#4472C4", alpha=0.7)
    sns.kdeplot(vals, ax=axes[i], color="#ED7D31")
    axes[i].set_title(col, fontsize=10)
plt.tight_layout()
savefig("01_distributions")
report("Saved: eda_01_distributions.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Normality tests
# ═══════════════════════════════════════════════════════════════════════════════
report("\n── Normality Tests (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling) ──")
report(f"{'Feature':<22} {'SW_p':>7} {'DP_p':>7} {'AD_normal':>10} "
       f"{'Skew':>7} {'Kurt':>7} {'Normal?':>8}")
report("-" * 78)

normality_results = {}
for col in all_cols:
    r = normality_tests(df[col], col)
    normality_results[col] = r
    sw_p_str = f"{r['shapiro_p']:.4f}" if r['shapiro_p'] is not None else "N/A  "
    report(f"{col:<22} {sw_p_str:>7} {r['dagostino_p']:>7.4f} {str(r['anderson_normal']):>10} "
           f"{r['skewness']:>7.3f} {r['kurtosis']:>7.3f} {str(r['majority_normal']):>8}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Outlier detection (majority vote)
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n── Outlier Detection (IQR | MAD | 3σ — majority vote ≥2/3) ──")
report(f"{'Feature':<22} {'IQR':>6} {'MAD':>6} {'Zscore':>8} {'Majority':>10} {'%':>6}")
report("-" * 65)

outlier_masks = {}
for col in all_cols:
    iqr_mask = iqr_outliers(df[col])
    mad_mask = mad_outliers(df[col])
    z_mask = pd.Series(zscore_outliers(df[col]).astype(bool), index=df.index)
    maj_mask = majority_outliers(df[col])
    outlier_masks[col] = maj_mask
    pct = 100 * maj_mask.sum() / len(df)
    report(f"{col:<22} {iqr_mask.sum():>6} {mad_mask.sum():>6} {z_mask.sum():>8} "
           f"{maj_mask.sum():>10} {pct:>5.1f}%")

# Visualize outliers for target
# Visualize outliers (2x4 grid)
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS + [TARGET]):
    is_out = outlier_masks[col]
    axes[i].scatter(df[col][~is_out], df[TARGET][~is_out], alpha=0.5, s=10, color="#4472C4")
    axes[i].scatter(df[col][is_out], df[TARGET][is_out], alpha=0.9, s=30, color="#FF0000", marker="x")
    axes[i].set_title(col, fontsize=9)
plt.tight_layout()
savefig("02_outliers")
report("Saved: eda_02_outliers.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Correlation analysis
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n── Correlation Analysis (Pearson | Spearman | Kendall) ──")

def corr_with_target(df, feature_cols, target):
    rows = []
    for col in feature_cols:
        x, y = df[col].values, df[target].values
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        kr, kp = kendalltau(x, y)
        rows.append({"feature": col,
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_r": sr, "spearman_p": sp,
                     "kendall_tau": kr, "kendall_p": kp})
    return pd.DataFrame(rows)

corr_df = corr_with_target(df, FEATURE_COLS, TARGET)
report(corr_df.set_index("feature").round(4).to_string())

# Heatmap — Pearson
# Heatmap — 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
methods = ["pearson", "spearman", "kendall"]
for i, method in enumerate(methods):
    ax = axes.flatten()[i]
    corr_mat = df[FEATURE_COLS + [TARGET]].corr(method=method if method != "kendall" else "kendall")
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(f"{method.capitalize()} Correlation")
axes.flatten()[3].axis("off") # Hide 4th slot
plt.tight_layout()
savefig("03_correlation_heatmaps")
report("Saved: eda_03_correlation_heatmaps.svg")

# Detect high inter-feature correlation
report("\n── High inter-feature correlation (|Pearson| > 0.85) ──")
feat_corr = df[FEATURE_COLS].corr()
high_corr_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
        r = feat_corr.iloc[i, j]
        if abs(r) > 0.85:
            high_corr_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], r))
            report(f"  {FEATURE_COLS[i]} <-> {FEATURE_COLS[j]}: r={r:.3f}")
if not high_corr_pairs:
    report("  None found.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Per-image distribution & PSI / JSD
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n── PSI and JSD: train split vs. test split (80/20 GroupShuffleSplit) ──")
from sklearn.model_selection import GroupShuffleSplit

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)
groups = df["image_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

report(f"Train: {len(df_train):,} tiles | Test: {len(df_test):,} tiles")
report(f"{'Feature':<22} {'PSI':>8} {'JSD':>8}  {'Drift?':>8}")
report("-" * 52)

drift_flags = {}
for col in FEATURE_COLS + [TARGET]:
    p = psi(df_train[col].values, df_test[col].values)
    j = jsd(df_train[col].values, df_test[col].values)
    drifted = p > 0.2 or j > 0.1
    drift_flags[col] = drifted
    report(f"{col:<22} {p:>8.4f} {j:>8.4f}  {str(drifted):>8}")

# PSI / JSD visualization
# PSI / JSD visualization (2x4 grid)
cols_to_plot = FEATURE_COLS + [TARGET]
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(cols_to_plot):
    axes[i].hist(df_train[col], bins=20, alpha=0.6, label="Train", color="#4472C4", density=True)
    axes[i].hist(df_test[col], bins=20, alpha=0.6, label="Test", color="#ED7D31", density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].legend(fontsize=7)
plt.tight_layout()
savefig("04_train_test_distributions")
report("Saved: eda_04_train_test_distributions.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Dimensionality reduction — PCA + t-SNE
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n── Dimensionality Reduction (PCA + t-SNE) ──")
scaler_tmp = StandardScaler()
X_scaled = scaler_tmp.fit_transform(df[FEATURE_COLS].values)

# PCA
pca = PCA(n_components=min(len(FEATURE_COLS), 7))
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
cum_exp = np.cumsum(explained)
report(f"PCA explained variance (cumulative): {cum_exp.round(3).tolist()}")
n_components_95 = int(np.argmax(cum_exp >= 0.95)) + 1
report(f"Components for 95% variance: {n_components_95}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].bar(range(1, len(explained) + 1), explained * 100, color="#4472C4")
axes[0].step(range(1, len(cum_exp) + 1), cum_exp * 100, where="mid",
             color="#ED7D31", lw=2, label="Cumulative")
axes[0].axhline(95, color="k", ls="--", lw=1, label="95%")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance (%)")
axes[0].set_title("PCA Scree Plot")
axes[0].legend()

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df[TARGET].values,
                          cmap="viridis", alpha=0.6, s=15)
plt.colorbar(scatter, ax=axes[1], label="runtime_ms")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("PCA: PC1 vs PC2 (colored by runtime_ms)")

# t-SNE (on sample if large)
sample_n = min(len(df), 400)
idx_sample = np.random.RandomState(42).choice(len(df), sample_n, replace=False)
X_tsne_input = X_scaled[idx_sample]
y_tsne = df[TARGET].values[idx_sample]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
X_tsne = tsne.fit_transform(X_tsne_input)
sc = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap="viridis", alpha=0.7, s=15)
plt.colorbar(sc, ax=axes[2], label="runtime_ms")
axes[2].set_xlabel("t-SNE 1")
axes[2].set_ylabel("t-SNE 2")
axes[2].set_title(f"t-SNE (n={sample_n}, colored by runtime_ms)")

plt.suptitle("Dimensionality Reduction", fontsize=12)
plt.tight_layout()
savefig("05_dimensionality_reduction")
report("Saved: eda_05_dimensionality_reduction.svg")

# PCA loadings
loadings = pd.DataFrame(pca.components_.T, index=FEATURE_COLS,
                        columns=[f"PC{i+1}" for i in range(pca.n_components_)])
report("\nPCA Loadings:")
report(loadings.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(loadings.round(3), annot=True, cmap="RdBu_r", center=0, ax=ax, fmt=".2f")
ax.set_title("PCA Component Loadings")
plt.tight_layout()
savefig("06_pca_loadings")
report("Saved: eda_06_pca_loadings.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Target analysis
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n── Target Variable Analysis: runtime_ms ──")
target_vals = df[TARGET].values
report(f"  Min:    {target_vals.min():.4f} ms")
report(f"  Max:    {target_vals.max():.4f} ms")
report(f"  Mean:   {target_vals.mean():.4f} ms")
report(f"  Median: {np.median(target_vals):.4f} ms")
report(f"  Std:    {target_vals.std():.4f} ms")
report(f"  CV:     {target_vals.std()/target_vals.mean()*100:.2f}%")
report(f"  Skewness: {skew(target_vals):.4f}")
report(f"  Kurtosis: {kurtosis(target_vals):.4f}")
r = normality_results[TARGET]
report(f"  Normal (majority): {r['majority_normal']}")

# Target Analysis (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
axes[0].hist(target_vals, bins=40, color="#4472C4", edgecolor="k")
axes[0].set_title("Runtime Distribution")
axes[1].boxplot(target_vals)
axes[1].set_title("Runtime Boxplot")
per_image = df.groupby("image_id")[TARGET].agg(["mean", "std"]).reset_index()
axes[2].bar(range(len(per_image)), per_image["mean"], yerr=per_image["std"], color="#4472C4")
axes[2].set_title("Mean per Image")
stats.probplot(target_vals, dist="norm", plot=axes[3])
axes[3].set_title("Q-Q Plot")
plt.tight_layout()
savefig("07_target_analysis")
report("Saved: eda_07_target_analysis.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Preprocessing decisions
# ═══════════════════════════════════════════════════════════════════════════════
report("\n\n" + "=" * 70)
report("PREPROCESSING DECISIONS (statistically driven)")
report("=" * 70)

# 9a. Outlier removal — majority vote
total_out_mask = pd.Series([False] * len(df), index=df.index)
for col in FEATURE_COLS + [TARGET]:
    total_out_mask = total_out_mask | outlier_masks[col]
n_remove = total_out_mask.sum()
report(f"\nStep 1 — Outlier removal")
report(f"  Rows flagged by majority vote (any feature or target): {n_remove} "
       f"({100*n_remove/len(df):.1f}%)")
report(f"  Decision: REMOVE these rows")
df_clean = df[~total_out_mask].copy().reset_index(drop=True)
report(f"  Rows after removal: {len(df_clean):,}")

# 9b. Feature transformation (Yeo-Johnson if |skew| > threshold)
report(f"\nStep 2 — Feature transformation (|skew| > {SKEW_THRESH} → Yeo-Johnson)")
transform_cols = []
for col in FEATURE_COLS + [TARGET]:
    sk = skew(df_clean[col].dropna().values)
    r = normality_results[col]
    decision = abs(sk) > SKEW_THRESH and not r["majority_normal"]
    if decision:
        transform_cols.append(col)
    report(f"  {col:<22} skew={sk:>7.3f}  normal={str(r['majority_normal']):>5}  "
           f"→ {'TRANSFORM' if decision else 'keep'}")

pt = PowerTransformer(method="yeo-johnson", standardize=False)
if transform_cols:
    df_clean[transform_cols] = pt.fit_transform(df_clean[transform_cols])
    report(f"  Applied Yeo-Johnson to: {transform_cols}")
else:
    report("  No features require transformation.")

# 9c. Scaling — RobustScaler (handles residual outliers better than StandardScaler)
report(f"\nStep 3 — Scaling (RobustScaler, robust to residual outliers)")
report(f"  Applying RobustScaler to all feature columns.")
rs = RobustScaler()
df_clean[FEATURE_COLS] = rs.fit_transform(df_clean[FEATURE_COLS])
report("  Done.")

# 9d. Duplicate check
n_dup = df_clean.duplicated(subset=FEATURE_COLS).sum()
report(f"\nStep 4 — Duplicate rows (feature space): {n_dup}")
if n_dup > 0:
    df_clean = df_clean.drop_duplicates(subset=FEATURE_COLS).reset_index(drop=True)
    report(f"  Removed. Rows remaining: {len(df_clean):,}")

# 9e. Final statistics after preprocessing
report(f"\nStep 5 — Post-preprocessing statistics:")
report(f"  Final shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} cols")
report(df_clean[FEATURE_COLS + [TARGET]].describe().round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Before vs. after preprocessing comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, len(FEATURE_COLS), figsize=(3 * len(FEATURE_COLS), 8))
for i, col in enumerate(FEATURE_COLS):
    axes[0, i].hist(df[col], bins=25, color="#4472C4", edgecolor="k", alpha=0.8)
    axes[0, i].set_title(f"{col}\n(before)", fontsize=8)
    axes[1, i].hist(df_clean[col], bins=25, color="#70AD47", edgecolor="k", alpha=0.8)
    axes[1, i].set_title(f"{col}\n(after)", fontsize=8)
axes[0, 0].set_ylabel("Before preprocessing", fontsize=9)
axes[1, 0].set_ylabel("After preprocessing", fontsize=9)
plt.suptitle("Feature Distributions: Before vs. After Preprocessing", fontsize=12)
plt.tight_layout()
savefig("08_before_after_preprocessing")
report("\nSaved: eda_08_before_after_preprocessing.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# Save preprocessed data
# ═══════════════════════════════════════════════════════════════════════════════
out_path = LOGS_DIR / "tile_profiles_preprocessed.csv"
df_clean.to_csv(out_path, index=False)
report(f"\n✓ Preprocessed data saved: {out_path}")
report(f"  Rows: {len(df_clean):,}  (original: {len(df):,}  removed: {len(df)-len(df_clean):,})")

# Save report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
report(f"\n✓ EDA report saved: {REPORT_PATH}")

report("\n" + "=" * 70)
report("EDA + PREPROCESSING COMPLETE")
report("=" * 70)
