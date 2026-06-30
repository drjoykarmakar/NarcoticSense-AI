from __future__ import annotations

import numpy as np
import pandas as pd

from narcoticsense.spectroscopy import Spectrum


def spectral_metrics(spectrum: Spectrum) -> pd.DataFrame:
    """Return common scalar quantities useful in a spectroscopy notebook/report."""
    x = np.asarray(spectrum.x, dtype=float)
    y = np.asarray(spectrum.y, dtype=float)
    idx_max = int(np.argmax(y))
    idx_min = int(np.argmin(y))
    area = float(np.trapz(y, x))
    abs_area = float(np.trapz(np.abs(y), x))
    rows = [
        ("data points", spectrum.n_points),
        ("x minimum", float(np.min(x))),
        ("x maximum", float(np.max(x))),
        ("y minimum", float(np.min(y))),
        ("y maximum", float(np.max(y))),
        ("y mean", float(np.mean(y))),
        ("y standard deviation", float(np.std(y))),
        ("maximum position", float(x[idx_max])),
        ("minimum position", float(x[idx_min])),
        ("signed area", area),
        ("absolute area", abs_area),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def derivative_spectrum(spectrum: Spectrum, *, order: int = 1) -> Spectrum:
    if order not in {1, 2}:
        raise ValueError("Only first and second derivatives are supported.")
    y = np.asarray(spectrum.y, dtype=float)
    x = np.asarray(spectrum.x, dtype=float)
    for _ in range(order):
        y = np.gradient(y, x)
    return spectrum.copy_with(y=y, derivative_order=order)
