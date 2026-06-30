from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from narcoticsense.spectroscopy import Spectrum
from narcoticsense.utils.numeric import trapezoid_integral


def spectral_metrics(spectrum: Spectrum) -> pd.DataFrame:
    """Return common scalar quantities useful in a spectroscopy notebook/report."""
    x = np.asarray(spectrum.x, dtype=float)
    y = np.asarray(spectrum.y, dtype=float)
    idx_max = int(np.argmax(y))
    idx_min = int(np.argmin(y))
    area = trapezoid_integral(y, x)
    abs_area = trapezoid_integral(np.abs(y), x)
    rows = [
        ("data points", spectrum.n_points),
        ("x minimum", float(np.min(x))),
        ("x maximum", float(np.max(x))),
        ("x step median", float(np.median(np.diff(x)))),
        ("y minimum", float(np.min(y))),
        ("y maximum", float(np.max(y))),
        ("y mean", float(np.mean(y))),
        ("y standard deviation", float(np.std(y))),
        ("maximum position", float(x[idx_max])),
        ("minimum position", float(x[idx_min])),
        ("signed area", area),
        ("absolute area", abs_area),
        ("root mean square", float(np.sqrt(np.mean(y**2)))),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def derivative_spectrum(
    spectrum: Spectrum,
    *,
    order: int = 1,
    smooth: bool = False,
    window_length: int = 11,
    polyorder: int = 3,
) -> Spectrum:
    if order not in {1, 2}:
        raise ValueError("Only first and second derivatives are supported.")
    y = np.asarray(spectrum.y, dtype=float)
    x = np.asarray(spectrum.x, dtype=float)
    if smooth and len(y) >= window_length:
        window = window_length if window_length % 2 == 1 else window_length + 1
        if window <= len(y):
            y = savgol_filter(y, window, min(polyorder, window - 1))
    for _ in range(order):
        y = np.gradient(y, x)
    return spectrum.copy_with(y=y, derivative_order=order)
