from pathlib import Path
import multiprocessing
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

KODAK_DIR = DATA_DIR / "kodak"
DIV2K_DIR = DATA_DIR / "div2k"
BSDS500_DIR = DATA_DIR / "bsds500"

TILE_SIZE = 128
HALO = 16
NUM_WORKERS = min(8, multiprocessing.cpu_count())
NUM_GPU_STREAMS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIPELINE_A = "edge_gradient"
PIPELINE_B = "denoise_threshold"

CANNY_LOW = 50
CANNY_HIGH = 150
BILATERAL_D = 9
BILATERAL_SIGMA = 75
NLM_H = 10
MORPH_KERNEL_A = 5
MORPH_KERNEL_B = 3

PREDICTOR_TRAIN_REPEATS = 3
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
CNN_LR = 1e-4
CNN_IMG_SIZE = 224

TILE_SIZES = [128, 256, 512, 1024]
HALOS = [4, 8, 16, 32]
WORKER_COUNTS = [1, 2, 4, 8]

for d in [LOGS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
