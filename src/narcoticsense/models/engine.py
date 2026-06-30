from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR


@dataclass
class ModelTrainingResult:
    model: Any
    task: str
    model_name: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion: pd.DataFrame | None = None
    report: pd.DataFrame | None = None
    probabilities: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None


def available_classification_models() -> list[str]:
    return [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Logistic Regression",
        "SVM RBF",
        "KNN",
    ]


def available_regression_models() -> list[str]:
    return ["PLS Regression", "Random Forest Regressor", "Ridge Regression", "SVR RBF"]


def build_classifier(model_name: str, random_state: int = 42) -> Pipeline:
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        )
        return Pipeline([("model", estimator)])
    if model_name == "Extra Trees":
        estimator = ExtraTreesClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        )
        return Pipeline([("model", estimator)])
    if model_name == "Gradient Boosting":
        estimator = GradientBoostingClassifier(random_state=random_state)
        return Pipeline([("model", estimator)])
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if model_name == "SVM RBF":
        estimator = SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=random_state
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if model_name == "KNN":
        estimator = KNeighborsClassifier(n_neighbors=3)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    raise ValueError(f"Unknown classification model: {model_name}")


def build_regressor(model_name: str, n_components: int = 2, random_state: int = 42) -> Pipeline:
    if model_name == "PLS Regression":
        return Pipeline(
            [("scale", StandardScaler()), ("model", PLSRegression(n_components=n_components))]
        )
    if model_name == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor

        return Pipeline(
            [("model", RandomForestRegressor(n_estimators=300, random_state=random_state))]
        )
    if model_name == "Ridge Regression":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if model_name == "SVR RBF":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))])
    raise ValueError(f"Unknown regression model: {model_name}")


def _safe_cv_splits(y: list[str]) -> int:
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return 0
    return int(max(2, min(5, counts.min())))


def train_classification_model(
    x: np.ndarray,
    y: list[str],
    sample_ids: list[str],
    model_name: str = "Random Forest",
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelTrainingResult:
    x = np.asarray(x, dtype=float)
    y = [str(v) for v in y]
    if x.shape[0] != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if len(set(y)) < 2:
        raise ValueError("Classification needs at least two classes.")

    model = build_classifier(model_name, random_state=random_state)
    encoder = LabelEncoder().fit(y)
    stratify = y if min(pd.Series(y).value_counts()) >= 2 else None
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model.fit(x[train_idx], np.array(y)[train_idx])
    pred = model.predict(x[test_idx])
    classes = list(encoder.classes_)

    metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy_score(np.array(y)[test_idx], pred)},
            {
                "metric": "balanced_accuracy",
                "value": balanced_accuracy_score(np.array(y)[test_idx], pred),
            },
            {"metric": "macro_f1", "value": f1_score(np.array(y)[test_idx], pred, average="macro")},
            {"metric": "train_samples", "value": int(len(train_idx))},
            {"metric": "test_samples", "value": int(len(test_idx))},
            {"metric": "classes", "value": int(len(classes))},
        ]
    )

    predictions = pd.DataFrame(
        {
            "sample_id": [sample_ids[i] for i in test_idx],
            "true_label": np.array(y)[test_idx],
            "predicted_label": pred,
            "correct": pred == np.array(y)[test_idx],
        }
    )
    cm = pd.DataFrame(
        confusion_matrix(np.array(y)[test_idx], pred, labels=classes),
        index=classes,
        columns=classes,
    )
    report = pd.DataFrame(
        classification_report(np.array(y)[test_idx], pred, output_dict=True)
    ).T.reset_index(names="class_or_metric")

    probabilities = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x[test_idx])
        probabilities = pd.DataFrame(prob, columns=[f"prob_{c}" for c in model.classes_])
        probabilities.insert(0, "sample_id", [sample_ids[i] for i in test_idx])
        probabilities.insert(1, "true_label", np.array(y)[test_idx])

    cv_splits = _safe_cv_splits(y)
    if cv_splits >= 2:
        try:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            cv_model = build_classifier(model_name, random_state=random_state)
            cv_pred = cross_val_predict(cv_model, x, y, cv=cv)
            metrics = pd.concat(
                [
                    metrics,
                    pd.DataFrame(
                        [
                            {"metric": "cv_splits", "value": cv_splits},
                            {"metric": "cv_accuracy", "value": accuracy_score(y, cv_pred)},
                            {
                                "metric": "cv_macro_f1",
                                "value": f1_score(y, cv_pred, average="macro"),
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )
        except Exception as exc:
            metrics = pd.concat(
                [
                    metrics,
                    pd.DataFrame([{"metric": "cv_warning", "value": str(exc)}]),
                ],
                ignore_index=True,
            )

    return ModelTrainingResult(
        model=model,
        task="classification",
        model_name=model_name,
        metrics=metrics,
        predictions=predictions,
        confusion=cm,
        report=report,
        probabilities=probabilities,
        feature_importance=feature_importance_table(model),
    )


def train_regression_model(
    x: np.ndarray,
    y: list[float],
    sample_ids: list[str],
    model_name: str = "PLS Regression",
    test_size: float = 0.25,
    n_components: int = 2,
    random_state: int = 42,
) -> ModelTrainingResult:
    x = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x.shape[0] != len(y_array):
        raise ValueError("X and y must contain the same number of samples.")
    if len(y_array) < 4:
        raise ValueError("Regression needs at least four labeled samples.")

    indices = np.arange(len(y_array))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    safe_components = max(1, min(n_components, len(train_idx) - 1, x.shape[1]))
    model = build_regressor(model_name, n_components=safe_components, random_state=random_state)
    model.fit(x[train_idx], y_array[train_idx])
    pred = np.asarray(model.predict(x[test_idx])).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_array[test_idx], pred)))
    metrics = pd.DataFrame(
        [
            {"metric": "r2", "value": r2_score(y_array[test_idx], pred)},
            {"metric": "rmse", "value": rmse},
            {"metric": "mae", "value": mean_absolute_error(y_array[test_idx], pred)},
            {"metric": "train_samples", "value": int(len(train_idx))},
            {"metric": "test_samples", "value": int(len(test_idx))},
        ]
    )
    predictions = pd.DataFrame(
        {
            "sample_id": [sample_ids[i] for i in test_idx],
            "true_value": y_array[test_idx],
            "predicted_value": pred,
            "residual": y_array[test_idx] - pred,
        }
    )
    return ModelTrainingResult(
        model=model,
        task="regression",
        model_name=model_name,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance_table(model),
    )


def feature_importance_table(model: Any, x_axis: np.ndarray | None = None) -> pd.DataFrame | None:
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    if values is None:
        return None
    df = pd.DataFrame({"variable_index": np.arange(len(values)), "importance": values})
    if x_axis is not None and len(x_axis) == len(values):
        df.insert(1, "x", x_axis)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model_bundle(
    result: ModelTrainingResult,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.model,
        "task": result.task,
        "model_name": result.model_name,
        "metrics": result.metrics,
        "metadata": metadata or {},
    }
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))
