from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from narcoticsense.spectroscopy.core import Spectrum


def spectrum_quality_report(spectrum: Spectrum) -> pd.DataFrame:
    """Return simple, chemistry-facing data quality checks for one spectrum."""
    x = np.asarray(spectrum.x, dtype=float)
    y = np.asarray(spectrum.y, dtype=float)
    checks: list[dict[str, object]] = []

    def add(check: str, status: str, detail: str) -> None:
        checks.append({"check": check, "status": status, "detail": detail})

    finite_x = np.isfinite(x).all()
    finite_y = np.isfinite(y).all()
    add(
        "finite values",
        "pass" if finite_x and finite_y else "fail",
        "x and y should not contain NaN or infinity",
    )

    unique_x = np.unique(x).size
    add(
        "unique x-axis",
        "pass" if unique_x == x.size else "warn",
        f"{unique_x} unique x values from {x.size} points",
    )

    monotonic = bool(np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0))
    add(
        "monotonic x-axis",
        "pass" if monotonic else "warn",
        "x-axis should be ordered before interpolation",
    )

    dyn = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    add("dynamic range", "pass" if dyn > 1e-12 else "fail", f"range = {dyn:.4g}")

    noise = float(np.nanstd(np.diff(y))) if y.size > 2 else 0.0
    signal = float(np.nanmax(np.abs(y))) if y.size else 0.0
    snr = signal / noise if noise > 0 else np.inf
    add("rough signal/noise", "pass" if snr >= 10 else "warn", f"approximate SNR = {snr:.3g}")

    peaks, _ = find_peaks(y, prominence=max(dyn * 0.03, 1e-12)) if y.size else ([], {})
    add(
        "peak content", "pass" if len(peaks) > 0 else "warn", f"detected {len(peaks)} rough peak(s)"
    )

    return pd.DataFrame(checks)


def dataset_quality_report(spectra: list[Spectrum]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spectrum in spectra:
        report = spectrum_quality_report(spectrum)
        counts = report["status"].value_counts().to_dict()
        rows.append(
            {
                "sample_id": spectrum.sample_id,
                "points": spectrum.n_points,
                "passes": int(counts.get("pass", 0)),
                "warnings": int(counts.get("warn", 0)),
                "failures": int(counts.get("fail", 0)),
            }
        )
    return pd.DataFrame(rows)
