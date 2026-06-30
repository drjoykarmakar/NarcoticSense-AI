from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize


def binary_roc_pr_tables(
    y_true: list[str], probability: np.ndarray, classes: list[str]
) -> dict[str, pd.DataFrame]:
    """Create one-vs-rest ROC and precision-recall tables for classifiers with probabilities."""
    y = np.asarray(y_true, dtype=str)
    classes = [str(c) for c in classes]
    if probability.shape[1] != len(classes):
        raise ValueError("Probability columns must match classes.")
    y_bin = label_binarize(y, classes=classes)
    if y_bin.shape[1] == 1 and len(classes) == 2:
        y_bin = np.column_stack([1 - y_bin.ravel(), y_bin.ravel()])
    tables: dict[str, pd.DataFrame] = {}
    for idx, cls in enumerate(classes):
        positive = y_bin[:, idx]
        if len(np.unique(positive)) < 2:
            continue
        fpr, tpr, roc_thresholds = roc_curve(positive, probability[:, idx])
        precision, recall, pr_thresholds = precision_recall_curve(positive, probability[:, idx])
        tables[f"roc_{cls}"] = pd.DataFrame(
            {
                "class": cls,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": np.r_[roc_thresholds[:-1], np.nan],
                "auc": auc(fpr, tpr),
            }
        )
        tables[f"pr_{cls}"] = pd.DataFrame(
            {
                "class": cls,
                "precision": precision,
                "recall": recall,
                "threshold": np.r_[pr_thresholds, np.nan],
                "auc": auc(recall, precision),
            }
        )
    return tables


def reliability_table(
    y_true: list[str], probability: np.ndarray, classes: list[str], *, n_bins: int = 10
) -> pd.DataFrame:
    """Return calibration/reliability table using max predicted probability."""
    y = np.asarray(y_true, dtype=str)
    pred_idx = np.argmax(probability, axis=1)
    pred_labels = np.asarray(classes, dtype=str)[pred_idx]
    confidence = np.max(probability, axis=1)
    correct = (pred_labels == y).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, confidence, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_confidence": mean_pred, "fraction_correct": frac_pos})
