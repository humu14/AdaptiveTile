import numpy as np
import joblib
import time as _time
from pathlib import Path
from copy import deepcopy
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as tv_models
import torchvision.transforms as T
from src.config import CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LR, CNN_IMG_SIZE, DEVICE


TABULAR_MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "svr": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10))]),
    "random_forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "xgboost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42,
                                  n_jobs=-1, verbosity=0),
    "lightgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42,
                                    n_jobs=-1, verbose=-1),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ]),
}


class TileComplexityPredictor:
    def __init__(self):
        self._tabular_models: dict = {}
        self._cnn_models: dict = {}
        self.best_tabular_model: str | None = None
        self.best_cnn_model: str | None = None
        self._best_model = None
        self._mode: str | None = None

    # ── Tabular ──────────────────────────────────────────────────────────────

    def fit_tabular(self, X: np.ndarray, y: np.ndarray):
        best_spearman = -1.0
        for name, model in TABULAR_MODELS.items():
            m = deepcopy(model)
            m.fit(X, y)
            self._tabular_models[name] = m
            preds = m.predict(X)
            sp = spearmanr(y, preds).correlation
            if sp > best_spearman:
                best_spearman = sp
                self.best_tabular_model = name
                self._best_model = m
                self._mode = "tabular"

    def evaluate_tabular(self, X: np.ndarray, y: np.ndarray) -> dict:
        results = {}
        for name, model in self._tabular_models.items():
            preds = model.predict(X)
            results[name] = {
                "mae": float(mean_absolute_error(y, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "r2": float(r2_score(y, preds)),
                "spearman": float(spearmanr(y, preds).correlation),
            }
        return results

    # ── CNN ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_cnn(name: str) -> nn.Module:
        if name == "squeezenet":
            m = tv_models.squeezenet1_1(weights=tv_models.SqueezeNet1_1_Weights.DEFAULT)
            m.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)
            m.num_classes = 1
        elif name == "mobilenet_v3_small":
            m = tv_models.mobilenet_v3_small(
                weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
            )
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
        elif name == "efficientnet_b0":
            m = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.DEFAULT
            )
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
        else:
            raise ValueError(f"Unknown CNN: {name}")
        return m

    def _get_cnn_transform(self):
        return T.Compose([
            T.ToPILImage(),
            T.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def fit_cnn(self, tiles_bgr: list, y: np.ndarray):
        transform = self._get_cnn_transform()
        imgs = torch.stack([transform(t) for t in tiles_bgr])
        labels = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(imgs, labels)
        loader = DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)

        cnn_names = ["squeezenet", "mobilenet_v3_small", "efficientnet_b0"]
        best_spearman = -1.0

        for name in cnn_names:
            print(f"  Training {name}...")
            model = self._build_cnn(name).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
            criterion = nn.MSELoss()

            for epoch in range(CNN_EPOCHS):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(imgs.to(DEVICE)).squeeze().cpu().numpy()
            sp = float(spearmanr(y, preds).correlation)
            self._cnn_models[name] = {"model": model.cpu(), "spearman": sp}
            print(f"    {name} Spearman: {sp:.3f}")

            if sp > best_spearman:
                best_spearman = sp
                self.best_cnn_model = name

    def evaluate_cnn(self, tiles_bgr: list, y: np.ndarray) -> dict:
        transform = self._get_cnn_transform()
        imgs = torch.stack([transform(t) for t in tiles_bgr]).to(DEVICE)
        results = {}
        for name, entry in self._cnn_models.items():
            model = entry["model"].to(DEVICE).eval()
            with torch.no_grad():
                preds = model(imgs).squeeze().cpu().numpy()
            results[name] = {
                "mae": float(mean_absolute_error(y, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y, preds))),
                "r2": float(r2_score(y, preds)),
                "spearman": float(spearmanr(y, preds).correlation),
            }
        return results

    def _predict_cnn(self, tiles_bgr: list) -> np.ndarray:
        if self.best_cnn_model is None:
            raise RuntimeError("No CNN trained.")
        transform = self._get_cnn_transform()
        imgs = torch.stack([transform(t) for t in tiles_bgr]).to(DEVICE)
        model = self._cnn_models[self.best_cnn_model]["model"].to(DEVICE).eval()
        with torch.no_grad():
            return model(imgs).squeeze().cpu().numpy().astype(np.float32)

    def measure_cnn_inference_time(self, tile_bgr: np.ndarray, n_reps: int = 100) -> dict:
        transform = self._get_cnn_transform()
        img = transform(tile_bgr).unsqueeze(0).to(DEVICE)
        times = {}
        for name, entry in self._cnn_models.items():
            model = entry["model"].to(DEVICE).eval()
            t0 = _time.perf_counter()
            for _ in range(n_reps):
                with torch.no_grad():
                    model(img)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times[name] = ((_time.perf_counter() - t0) / n_reps) * 1000
        return times

    # ── Unified interface ─────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._best_model is None:
            raise RuntimeError("No model fitted. Call fit_tabular() or fit_cnn() first.")
        if self._mode == "tabular":
            return self._best_model.predict(X).astype(np.float32)
        else:
            return self._predict_cnn(X)

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TileComplexityPredictor":
        return joblib.load(path)
