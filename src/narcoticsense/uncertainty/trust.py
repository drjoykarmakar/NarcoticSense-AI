from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class TrustResult:
    predictions: pd.DataFrame
    thresholds: pd.DataFrame
    summary: pd.DataFrame


def probability_confidence_table(
    model: Any,
    x: np.ndarray,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Create max-probability confidence table for classifiers."""
    x = np.asarray(x, dtype=float)
    sample_ids = sample_ids or [f"sample_{i}" for i in range(x.shape[0])]
    if not hasattr(model, "predict"):
        raise ValueError("Model must implement predict().")
    pred = model.predict(x)
    out = pd.DataFrame({"sample_id": sample_ids, "prediction": pred})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", [str(i) for i in range(proba.shape[1])]))
        out["confidence"] = proba.max(axis=1)
        out["runner_up_confidence"] = (
            np.partition(proba, -2, axis=1)[:, -2] if proba.shape[1] > 1 else 0.0
        )
        out["margin"] = out["confidence"] - out["runner_up_confidence"]
        for idx, cls in enumerate(classes):
            out[f"prob_{cls}"] = proba[:, idx]
    else:
        out["confidence"] = np.nan
        out["margin"] = np.nan
    return out


def conformal_classification_thresholds(
    model: Any,
    x_calibration: np.ndarray,
    y_calibration: list[str] | np.ndarray,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Estimate classwise conformal nonconformity thresholds from calibration data."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Conformal classification requires predict_proba().")
    y = np.asarray([str(v) for v in y_calibration])
    proba = model.predict_proba(np.asarray(x_calibration, dtype=float))
    classes = [str(c) for c in model.classes_]
    rows = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        class_pos = classes.index(cls)
        nonconf = 1.0 - proba[idx, class_pos]
        q = float(np.quantile(nonconf, min(1.0, max(0.0, 1 - alpha))))
        rows.append(
            {"class": cls, "alpha": alpha, "threshold": q, "calibration_samples": int(len(idx))}
        )
    return pd.DataFrame(rows)


def conformal_prediction_sets(
    model: Any,
    x: np.ndarray,
    thresholds: pd.DataFrame,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return conformal prediction sets and ambiguity flags."""
    x = np.asarray(x, dtype=float)
    sample_ids = sample_ids or [f"sample_{i}" for i in range(x.shape[0])]
    proba = model.predict_proba(x)
    classes = [str(c) for c in model.classes_]
    threshold_map = {str(r["class"]): float(r["threshold"]) for _, r in thresholds.iterrows()}
    rows = []
    for i, sid in enumerate(sample_ids):
        members = []
        for j, cls in enumerate(classes):
            if 1.0 - float(proba[i, j]) <= threshold_map.get(cls, 1.0):
                members.append(cls)
        rows.append(
            {
                "sample_id": sid,
                "prediction_set": ";".join(members),
                "set_size": int(len(members)),
                "empty_set": len(members) == 0,
                "ambiguous": len(members) != 1,
            }
        )
    return pd.DataFrame(rows)


def spectral_ood_scores(
    x_train: np.ndarray,
    x_query: np.ndarray,
    sample_ids: list[str] | None = None,
    percentile: float = 95.0,
) -> pd.DataFrame:
    """Distance-based out-of-distribution scoring using scaled nearest-neighbor distance."""
    x_train = np.asarray(x_train, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    sample_ids = sample_ids or [f"sample_{i}" for i in range(x_query.shape[0])]
    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    query_scaled = scaler.transform(x_query)
    train_dist = pairwise_distances(train_scaled, train_scaled)
    np.fill_diagonal(train_dist, np.inf)
    train_nn = train_dist.min(axis=1)
    threshold = float(np.percentile(train_nn, percentile)) if len(train_nn) else 0.0
    query_nn = pairwise_distances(query_scaled, train_scaled).min(axis=1)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "ood_score": query_nn,
            "ood_threshold": threshold,
            "is_ood": query_nn > threshold,
        }
    )


def trustworthy_classification_report(
    model: Any,
    x_train: np.ndarray,
    y_train: list[str],
    x_query: np.ndarray,
    sample_ids: list[str] | None = None,
    confidence_threshold: float = 0.7,
    alpha: float = 0.1,
) -> TrustResult:
    """Combine confidence, conformal sets, and OOD scores into an action table."""
    model.fit(x_train, y_train)
    confidence = probability_confidence_table(model, x_query, sample_ids)
    thresholds = conformal_classification_thresholds(model, x_train, y_train, alpha=alpha)
    sets = conformal_prediction_sets(model, x_query, thresholds, sample_ids)
    ood = spectral_ood_scores(x_train, x_query, sample_ids)
    out = confidence.merge(sets, on="sample_id").merge(ood, on="sample_id")
    out["low_confidence"] = out["confidence"].fillna(0.0) < confidence_threshold
    out["refer_to_confirmatory_testing"] = (
        out["low_confidence"] | out["ambiguous"] | out["empty_set"] | out["is_ood"]
    )
    summary = pd.DataFrame(
        [
            {"metric": "n_samples", "value": int(len(out))},
            {"metric": "low_confidence", "value": int(out["low_confidence"].sum())},
            {
                "metric": "ambiguous_or_empty",
                "value": int((out["ambiguous"] | out["empty_set"]).sum()),
            },
            {"metric": "ood", "value": int(out["is_ood"].sum())},
            {
                "metric": "refer_to_confirmatory_testing",
                "value": int(out["refer_to_confirmatory_testing"].sum()),
            },
        ]
    )
    return TrustResult(predictions=out, thresholds=thresholds, summary=summary)


def cross_validated_model_comparison(
    model_builders: dict[str, Any],
    x: np.ndarray,
    y: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare classifiers by cross-validated accuracy and macro F1."""
    from sklearn.metrics import accuracy_score, f1_score

    y = [str(v) for v in y]
    counts = pd.Series(y).value_counts()
    safe_splits = int(max(2, min(n_splits, counts.min()))) if len(counts) > 1 else 0
    if safe_splits < 2:
        raise ValueError("Need at least two classes with at least two samples each.")
    cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)
    rows = []
    for name, builder in model_builders.items():
        try:
            model = builder() if callable(builder) else clone(builder)
            pred = cross_val_predict(model, x, y, cv=cv)
            rows.append(
                {
                    "model": name,
                    "cv_splits": safe_splits,
                    "accuracy": float(accuracy_score(y, pred)),
                    "macro_f1": float(f1_score(y, pred, average="macro")),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "cv_splits": safe_splits,
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                    "warning": str(exc),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("macro_f1", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
