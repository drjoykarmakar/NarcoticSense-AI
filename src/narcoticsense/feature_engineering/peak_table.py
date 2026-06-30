from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from narcoticsense.spectroscopy import Spectrum
from narcoticsense.utils.numeric import trapezoid_integral


def _width_to_x_units(
    x: np.ndarray, left_ips: float, right_ips: float
) -> tuple[float, float, float]:
    index_axis = np.arange(len(x), dtype=float)
    left_x = float(np.interp(left_ips, index_axis, x))
    right_x = float(np.interp(right_ips, index_axis, x))
    return abs(right_x - left_x), left_x, right_x


def integrate_peak_window(
    spectrum: Spectrum, *, center: float, width: float | None = None, window: float | None = None
) -> float:
    """Integrate a peak around a center position.

    ``width`` is normally FWHM in x units. ``window`` overrides it and uses
    center ± window/2. The returned value is a signed trapezoidal area.
    """
    x = np.asarray(spectrum.x, dtype=float)
    y = np.asarray(spectrum.y, dtype=float)
    half = abs(window) / 2 if window is not None else max(abs(width or 0.0), 1e-12)
    mask = (x >= center - half) & (x <= center + half)
    if mask.sum() < 2:
        idx = int(np.argmin(np.abs(x - center)))
        left = max(0, idx - 1)
        right = min(len(x), idx + 2)
        mask = np.zeros_like(x, dtype=bool)
        mask[left:right] = True
    return trapezoid_integral(y[mask], x[mask])


def peak_table(
    spectrum: Spectrum, *, prominence: float = 0.05, max_peaks: int = 50
) -> pd.DataFrame:
    """Return ranked peak metrics including FWHM and local integrated area."""
    x = np.asarray(spectrum.x, dtype=float)
    y = np.asarray(spectrum.y, dtype=float)
    peaks, props = find_peaks(y, prominence=prominence)
    columns = [
        "rank",
        "position",
        "height",
        "prominence",
        "fwhm_x_units",
        "left_fwhm_position",
        "right_fwhm_position",
        "width_points",
        "area_fwhm_window",
    ]
    if len(peaks) == 0:
        return pd.DataFrame(columns=columns)
    widths, _, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
    rows = []
    order = sorted(range(len(peaks)), key=lambda i: y[peaks[i]], reverse=True)[:max_peaks]
    for rank, idx in enumerate(order, start=1):
        peak_idx = int(peaks[idx])
        fwhm, left_x, right_x = _width_to_x_units(x, float(left_ips[idx]), float(right_ips[idx]))
        center = float(x[peak_idx])
        rows.append(
            {
                "rank": rank,
                "position": center,
                "height": float(y[peak_idx]),
                "prominence": float(props["prominences"][idx]),
                "fwhm_x_units": fwhm,
                "left_fwhm_position": left_x,
                "right_fwhm_position": right_x,
                "width_points": float(widths[idx]),
                "area_fwhm_window": integrate_peak_window(spectrum, center=center, width=fwhm),
            }
        )
    return pd.DataFrame(rows, columns=columns)
