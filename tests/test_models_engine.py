from pathlib import Path

import numpy as np

from narcoticsense.models import (
    available_classification_models,
    available_regression_models,
    load_model_bundle,
    save_model_bundle,
    train_classification_model,
    train_regression_model,
)


def test_classification_model_engine(tmp_path: Path):
    rng = np.random.default_rng(7)
    class_a = rng.normal(0.0, 0.2, size=(8, 20))
    class_b = rng.normal(2.0, 0.2, size=(8, 20))
    x = np.vstack([class_a, class_b])
    y = ["A"] * 8 + ["B"] * 8
    ids = [f"s{i}" for i in range(len(y))]
    result = train_classification_model(x, y, ids, model_name="Random Forest", test_size=0.25)
    assert result.task == "classification"
    assert "accuracy" in result.metrics["metric"].tolist()
    assert result.confusion is not None
    assert result.predictions.shape[0] > 0
    path = save_model_bundle(result, tmp_path / "model.joblib")
    bundle = load_model_bundle(path)
    assert bundle["task"] == "classification"


def test_regression_model_engine():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 15))
    y = x[:, 0] * 2.0 + x[:, 1] * -0.5 + 1.0
    ids = [f"r{i}" for i in range(len(y))]
    result = train_regression_model(
        x, y.tolist(), ids, model_name="Ridge Regression", test_size=0.25
    )
    assert result.task == "regression"
    assert "rmse" in result.metrics["metric"].tolist()
    assert result.predictions.shape[0] > 0


def test_model_registry_lists():
    assert "Random Forest" in available_classification_models()
    assert "PLS Regression" in available_regression_models()
