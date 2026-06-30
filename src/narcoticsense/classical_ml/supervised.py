from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(slots=True)
class SupervisedResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    confusion: pd.DataFrame
    report: pd.DataFrame


def evaluate_classifier(
    matrix: np.ndarray,
    labels: list[str],
    sample_ids: list[str],
    *,
    model_name: str = "Random Forest",
    cv: int = 3,
) -> SupervisedResult:
    """Cross-validated supervised baseline for aligned spectra."""
    y = np.asarray(labels)
    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        raise ValueError("At least two labels/classes are required.")
    min_count = int(np.min(counts))
    cv = max(2, min(int(cv), min_count))
    if model_name.lower().startswith("svm"):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=False, class_weight="balanced")),
            ]
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pred = cross_val_predict(estimator, matrix, y, cv=splitter)
    metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": float(accuracy_score(y, pred))},
            {"metric": "balanced_accuracy", "value": float(balanced_accuracy_score(y, pred))},
            {"metric": "cv_folds", "value": float(cv)},
        ]
    )
    classes = sorted(map(str, unique.tolist()))
    confusion = pd.DataFrame(
        confusion_matrix(y, pred, labels=classes), index=classes, columns=classes
    )
    report = pd.DataFrame(
        classification_report(y, pred, output_dict=True, zero_division=0)
    ).T.reset_index(names="class")
    predictions = pd.DataFrame(
        {"sample_id": sample_ids, "true_label": y, "predicted_label": pred, "correct": y == pred}
    )
    return SupervisedResult(
        predictions=predictions, metrics=metrics, confusion=confusion, report=report
    )
