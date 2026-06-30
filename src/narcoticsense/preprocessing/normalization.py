from __future__ import annotations

import numpy as np

from narcoticsense.utils.numeric import trapezoid_integral


def minmax(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    span = np.nanmax(y) - np.nanmin(y)
    return np.zeros_like(y) if span == 0 else (y - np.nanmin(y)) / span


def area(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    total = trapezoid_integral(np.abs(y))
    return y if total == 0 else y / total


def standard_normal_variate(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    std = np.nanstd(y)
    return np.zeros_like(y) if std == 0 else (y - np.nanmean(y)) / std
