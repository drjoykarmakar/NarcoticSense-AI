from __future__ import annotations

import pandas as pd
from scipy.signal import find_peaks, peak_widths

from narcoticsense.spectroscopy import Spectrum


def peak_table(
    spectrum: Spectrum, *, prominence: float = 0.05, max_peaks: int = 50
) -> pd.DataFrame:
    peaks, props = find_peaks(spectrum.y, prominence=prominence)
    if len(peaks) == 0:
        return pd.DataFrame(columns=["rank", "position", "height", "prominence", "width_points"])
    widths = peak_widths(spectrum.y, peaks, rel_height=0.5)[0]
    rows = []
    for rank, idx in enumerate(
        sorted(range(len(peaks)), key=lambda i: spectrum.y[peaks[i]], reverse=True)[:max_peaks],
        start=1,
    ):
        peak_idx = peaks[idx]
        rows.append(
            {
                "rank": rank,
                "position": float(spectrum.x[peak_idx]),
                "height": float(spectrum.y[peak_idx]),
                "prominence": float(props["prominences"][idx]),
                "width_points": float(widths[idx]),
            }
        )
    return pd.DataFrame(rows)
