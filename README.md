# AdaptiveTile: Content-Aware Parallel Image Processing with Predictive Tile Scheduling

**Author:** Humaira Ayesha
**Email:** humaira.ayesha@g.bracu.ac.bd
**Course:** Parallel, Distributed, and High-Performance Computing

This repository contains the AdaptiveTile project — an end-to-end Python framework for content-aware parallel image processing. The project demonstrates how lightweight per-tile features and ML-based complexity prediction can be used to improve scheduling and reduce makespan in parallel image processing pipelines across CPU and GPU execution modes.

Table of contents

1. Project overview
2. Motivation and problem statement
3. System architecture
4. Datasets
5. Methodology
6. Exploratory data analysis (EDA)
7. Predictor evaluation
8. Experimental results
9. Discussion
10. Limitations
11. Future work
12. Notebook workflow
13. Reproducibility (instructions)
14. Module reference

---

## 1. Project Overview

AdaptiveTile is a Python framework that partitions images into halo-padded tiles, extracts lightweight visual features per tile, predicts per-tile processing time using ML regressors, and uses those predictions to inform scheduling across CPU and GPU workers. The framework supports multiple execution modes (serial/parallel CPU and serial/parallel GPU) and several scheduler variants (static, dynamic, predictive, hybrid).

Core pipeline: image ingestion → tiling with halo borders → feature extraction → complexity prediction → scheduling → parallel execution → tile merging → quality evaluation.

## 2. Motivation and Problem Statement

Tiles extracted from images are not computationally uniform: regions with rich texture, strong edges, or high-frequency content are more expensive to process (e.g., under Non-Local Means denoising) than smooth regions. When schedulers treat tiles identically, load imbalance reduces parallel efficiency and increases total runtime (makespan).

Research question: How can a scheduler minimise total makespan using lightweight content-aware predictions without adding prohibitive overhead?

Contributions:

- `AdaptiveTile` framework: end-to-end pipeline supporting CPU/GPU modes and multiple schedulers.
- Comparative predictor evaluation: tabular ML models and CNN regressors across Kodak, DIV2K, BSDS500.
- Scheduler benchmark: static, dynamic, predictive (LPT), hybrid.
- CPU–GPU comparison with analysis of speedup, efficiency, and load imbalance.

## 3. System Architecture

Stages:

- Tiling with halos: extract tiles of side `s` with halo `h` to ensure neighborhood-dependent filters work seamlessly at boundaries.
- Feature extraction: compute 7 features per tile (edge density, gradient variance, intensity variance, histogram entropy, LBP texture score, tile row, tile column).
- Predictor: tabular regressors and CNN regressors that map features (or tile images) to expected per-tile runtime in milliseconds.
- Scheduler: assignment policies — static (round-robin), dynamic (shared queue), predictive (LPT ordering by predicted runtime), hybrid (predictive + online correction).
- Workers: CPU processes (OpenCV) and GPU kernels (PyTorch + Kornia + CUDA streams).

Diagram (logical):

Input Image → Tiler (halo h) → Feature Extractor → Predictor → Scheduler → Workers (CPU/GPU) → Tile Merger → Output

## 4. Datasets

Datasets used:

- Kodak (24 images): primary profiling and benchmarking
- DIV2K (high-resolution images): generalisation tests
- BSDS500: small images to test behaviour across resolutions

Default experimental parameters (used unless otherwise stated): tile size `s=128 px`, halo `h=16 px`, workers `W ∈ {1,2,4,6,8}`, GPU streams `S=4`.

## 5. Methodology

5.1 Tiling with Halo Borders

Given image `I ∈ R^{H×W×C}` and tile size `s`, split into tiles with canonical region and `h`-pixel halo on each side, clamped to image bounds. Halos avoid seam artifacts for neighbourhood filters.

5.2 Feature Extraction

Seven features per tile:

1. Edge density — fraction of Canny edges
2. Gradient variance — variance of Sobel gradient magnitude
3. Intensity variance — variance of grayscale values
4. Histogram entropy — 16-bin intensity entropy
5. LBP texture score — mean LBP value (uniform)
6. Tile row index
7. Tile column index

5.3 Complexity Predictor

Two families of predictors were evaluated:

- Tabular models (scikit-learn, XGBoost, LightGBM): Linear, Ridge, SVR, Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP.
- CNN regressors: SqueezeNet, MobileNetV3-Small, EfficientNet-B0 (pretrained backbones, fine-tuned as regressors on tile images resized to 224×224).

Preprocessing: majority-vote outlier removal, Yeo-Johnson transformation for skewed features, RobustScaler normalisation.

5.4 Scheduler Variants

- Static: round-robin assignment.
- Dynamic: shared FIFO queue with workers pulling tasks.
- Predictive: LPT ordering using predicted runtime (decreasing order).
- Hybrid: predictive ordering with online reordering when prediction error exceeds threshold.

5.5 Processing Pipelines

Two pipelines were used to stress different compute regimes:

- Pipeline A (`edge_gradient`): lightweight edge- and morphology-based pipeline (~7 ms/tile).
- Pipeline B (`denoise_threshold`): heavy denoising pipeline (bilateral + NLM + adaptive thresholding; ≈317 ms/tile).

5.6 Execution Modes

- CPU Serial, CPU Parallel (multiprocessing), GPU Serial, GPU Parallel (CUDA streams).

5.7 Evaluation Metrics

- Speedup `S_p = T_1 / T_p` (vs CPU serial)
- Parallel efficiency `E_p = S_p / p`
- Load imbalance ratio `IR = max(t) / mean(t)`
- Output quality metrics: PSNR, SSIM

## 6. Exploratory Data Analysis (EDA)

Summarised findings:

- Tiling Kodak with `s=128, h=16` yields 576 tiles (24 images × 24 tiles per image). Median of three runs used to reduce jitter.
- Runtime distribution: mean ≈ 0.317 ms (on profiling hardware), CV ≈ 26.55% — significant heterogeneity.
- Top correlated features: histogram entropy, edge density, gradient variance.
- Outlier removal using an ensemble of IQR, MAD, and 3σ rules removed ~11% of tiles.

## 7. Predictor Evaluation

Key results:

- Preprocessing (Yeo-Johnson + RobustScaler) reduced MAE across tabular models by ~70%.
- Random Forest (preprocessed) was the best tabular model: MAE ≈ 0.0153 ms, Spearman ρ ≈ 0.584.
- CNNs underperformed on the small tile dataset due to limited data (~460 training tiles after cleaning).

## 8. Experimental Results

Summary of notable experiments (all using Pipeline B unless noted):

- Scalability (dynamic scheduler): speedup ≈ 1.90× (2 workers), 3.61× (4 workers), 4.76× (6 workers), 5.10× (8 workers). Peak at 6 workers due to memory contention.
- Scheduler comparison (W=6): dynamic > static > predictive ≈ hybrid. Dynamic achieved ~4.76×.
- Tile size sensitivity: `s=128` provided sufficient concurrency; `s>=512` removed parallelism benefits.
- Halo ablation: `h=16` optimal; `h=32` increased runtime ~20%.
- GPU (Pipeline B): GPU parallel (S=4) achieved ≈8.0× speedup vs CPU serial; SSIM drop observed vs CPU reference due to different implementations.

## 9. Discussion

Highlights:

- Dynamic scheduling is robust because it avoids committing to a potentially misordered plan when predictions under-estimate some tiles.
- Predictive LPT can suffer if underestimation places expensive tiles late in the queue. For predictive scheduling to reliably beat dynamic, predictors need higher rank correlation (Spearman ρ ≳ 0.75).
- There exists a compute-to-overhead threshold: lightweight pipelines should remain serial or use adaptive hybrid modes.

## 10. Limitations

- Predictor R² ≈ 0.25 (many factors such as fine-grained pixel relationships and cache effects are unmodelled).
- CNNs need much larger tile datasets to outperform tabular features.
- GPU variants are not bit-exact and show some SSIM degradation.

## 11. Future Work

- Richer features (frequency, saliency, HW counters), asymmetric loss penalties for underestimation, online scheduler selection, heterogeneous CPU+GPU partitioning, video extension, multi-node distributed scheduling.

---

## 12. Notebook Workflow

If you want to run the full project in a single executable notebook, use [adaptive-tile-processing/notebooks/full_project_pipeline.ipynb](adaptive-tile-processing/notebooks/full_project_pipeline.ipynb).

The notebook runs the repository’s main pipeline in order:

- Download data
- Profile tiles
- Train the predictor
- Run EDA and preprocessing
- Retrain on preprocessed data
- Execute benchmark experiments

It uses the existing script entry points, so the notebook stays aligned with the command-line workflow.

---

## 13. Reproducibility (Instructions)

Follow these exact steps to reproduce experiments and figures.

Environment

- OS: Windows or Linux
- Python: 3.10+
- GPU: optional — experiments run on CPU-only but GPU experiments require CUDA-capable device

Recommended setup (PowerShell / bash):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell (Windows)
# or (bash / WSL)
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r adaptive-tile-processing/requirements.txt
# For torch (GPU builds), install as instructed in adaptive-tile-processing/requirements-torch.txt
pip install -r adaptive-tile-processing/requirements-torch.txt
```

Download datasets

```bash
python scripts/01_download_data.py
# Follow printed instructions to manually add DIV2K and BSDS500
```

Run profiling, training, and experiments

```bash
python scripts/02_profile_tiles.py
python scripts/03_train_predictor.py
python scripts/04_run_experiments.py
```

Or run the full pipeline notebook:

```bash
jupyter notebook adaptive-tile-processing/notebooks/full_project_pipeline.ipynb
```

Run tests

```bash
pytest -q
```

Notes

- Large outputs (models, figures) are written to `outputs/` which is ignored by Git.
- If using a GPU, verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected outputs

- `outputs/plots/` — experiment figures
- `outputs/logs/eda_report.txt` — EDA logs
- `outputs/models/` — saved predictors
- `outputs/experiment_results.csv` — benchmark results

## 14. Module Reference

Brief mapping of important modules (see `src/` for implementation):

- `src/config.py` — central configuration (tile size, halo, paths)
- `src/tiling.py` — tile extraction with halos
- `src/features.py` — per-tile feature extraction
- `src/predictor.py` — training and evaluation of predictors
- `src/scheduler.py` — scheduler implementations
- `src/worker_cpu.py` — CPU processing pipelines (OpenCV)
- `src/worker_gpu.py` — GPU processing pipelines (Kornia/PyTorch)
- `src/merge.py` — tile merging and halo cropping
- `src/metrics.py` — speedup, efficiency, imbalance, PSNR, SSIM

---

