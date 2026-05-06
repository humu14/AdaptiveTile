import numpy as np
import pytest
from src.predictor import TileComplexityPredictor


def make_data(n=200):
    X = np.random.randn(n, 7).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(n) * 0.1).astype(np.float32)
    return X, y


def test_fit_predict_tabular():
    X, y = make_data()
    p = TileComplexityPredictor()
    p.fit_tabular(X[:160], y[:160])
    preds = p.predict(X[160:])
    assert preds.shape == (40,)
    assert np.isfinite(preds).all()


def test_evaluate_returns_all_metrics():
    X, y = make_data()
    p = TileComplexityPredictor()
    p.fit_tabular(X[:160], y[:160])
    metrics = p.evaluate_tabular(X[160:], y[160:])
    for key in ["mae", "rmse", "r2", "spearman"]:
        assert key in metrics[list(metrics.keys())[0]]


def test_best_model_name_is_set_after_fit():
    X, y = make_data()
    p = TileComplexityPredictor()
    p.fit_tabular(X[:160], y[:160])
    assert p.best_tabular_model is not None


def test_predict_before_fit_raises():
    p = TileComplexityPredictor()
    with pytest.raises(RuntimeError):
        p.predict(np.zeros((5, 7)))
