from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ExplanationResult:
    attribution: pd.DataFrame
    method: str
    target: str | None = None


def _predict_score(model: Any, x: np.ndarray, target_class: str | None = None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = list(getattr(model, "classes_", [str(i) for i in range(proba.shape[1])]))
        if target_class is None:
            return proba.max(axis=1)
        if target_class in classes:
            return proba[:, classes.index(target_class)]
        return proba.max(axis=1)
    pred = np.asarray(model.predict(x))
    if pred.ndim > 1:
        pred = pred.ravel()
    return pred.astype(float, copy=False)


def spectral_occlusion_importance(
    model: Any,
    x_sample: np.ndarray,
    x_axis: np.ndarray | None = None,
    target_class: str | None = None,
    window_size: int = 15,
    baseline: float | None = None,
) -> pd.DataFrame:
    """Estimate spectral-region importance by masking moving windows.

    This is a lightweight, dependency-free explanation method suitable for spectral data.
    It approximates how much a model score changes when a local spectral region is replaced
    by a baseline value.
    """
    sample = np.asarray(x_sample, dtype=float).reshape(1, -1)
    n_features = sample.shape[1]
    if n_features < 2:
        raise ValueError("At least two spectral variables are required.")
    if x_axis is None:
        x_axis = np.arange(n_features, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    if len(x_axis) != n_features:
        x_axis = np.arange(n_features, dtype=float)
    window_size = int(max(1, min(window_size, n_features)))
    fill_value = float(np.nanmedian(sample)) if baseline is None else float(baseline)
    base_score = float(_predict_score(model, sample, target_class=target_class)[0])
    rows: list[dict[str, float]] = []
    half = max(1, window_size // 2)
    for center in range(n_features):
        left = max(0, center - half)
        right = min(n_features, center + half + 1)
        masked = sample.copy()
        masked[:, left:right] = fill_value
        masked_score = float(_predict_score(model, masked, target_class=target_class)[0])
        rows.append(
            {
                "variable_index": center,
                "x": float(x_axis[center]),
                "importance": abs(base_score - masked_score),
                "signed_effect": base_score - masked_score,
                "window_start_x": float(x_axis[left]),
                "window_end_x": float(x_axis[right - 1]),
            }
        )
    df = pd.DataFrame(rows)
    total = float(df["importance"].sum())
    df["relative_importance"] = df["importance"] / total if total > 0 else 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def peak_level_attribution(
    attribution: pd.DataFrame,
    peaks: pd.DataFrame,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Map spectral-variable attribution onto detected peaks."""
    if attribution.empty or peaks.empty or "x" not in attribution.columns:
        return pd.DataFrame()
    peak_pos_col = "x" if "x" in peaks.columns else "position"
    if peak_pos_col not in peaks.columns:
        return pd.DataFrame()
    x_values = np.asarray(attribution["x"], dtype=float)
    step = float(np.nanmedian(np.abs(np.diff(np.sort(x_values))))) if len(x_values) > 1 else 1.0
    tol = float(tolerance) if tolerance is not None else max(step * 3, 1e-12)
    rows = []
    for _, peak in peaks.iterrows():
        pos = float(peak[peak_pos_col])
        local = attribution[np.abs(attribution["x"].astype(float) - pos) <= tol]
        rows.append(
            {
                "peak_x": pos,
                "attribution_sum": float(local["importance"].sum()) if not local.empty else 0.0,
                "attribution_max": float(local["importance"].max()) if not local.empty else 0.0,
                "n_variables": int(len(local)),
            }
        )
    return pd.DataFrame(rows).sort_values("attribution_sum", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    model: Any,
    x: np.ndarray,
    y: list[str] | np.ndarray,
    x_axis: np.ndarray | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return global permutation importance for trained estimators."""
    x = np.asarray(x, dtype=float)
    result = permutation_importance(
        model, x, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    df = pd.DataFrame(
        {
            "variable_index": np.arange(x.shape[1]),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    if x_axis is not None and len(x_axis) == x.shape[1]:
        df.insert(1, "x", np.asarray(x_axis, dtype=float))
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
