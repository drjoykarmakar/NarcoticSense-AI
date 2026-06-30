from __future__ import annotations

import numpy as np
import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def library_match(query: Spectrum, library: list[Spectrum], *, top_k: int = 5) -> pd.DataFrame:
    """Rank reference spectra by cosine similarity on the shared x-axis."""
    rows: list[dict[str, object]] = []
    q_order = np.argsort(query.x)
    qx = query.x[q_order]
    qy = query.y[q_order]
    for ref in library:
        start = max(float(np.min(qx)), float(np.min(ref.x)))
        stop = min(float(np.max(qx)), float(np.max(ref.x)))
        if start >= stop:
            continue
        grid = np.linspace(start, stop, min(1500, max(100, min(query.n_points, ref.n_points))))
        r_order = np.argsort(ref.x)
        q_interp = np.interp(grid, qx, qy)
        r_interp = np.interp(grid, ref.x[r_order], ref.y[r_order])
        q_centered = q_interp - np.mean(q_interp)
        r_centered = r_interp - np.mean(r_interp)
        rows.append(
            {
                "reference_id": ref.sample_id,
                "modality": ref.modality,
                "similarity": _cosine(q_centered, r_centered),
                "shared_x_min": start,
                "shared_x_max": stop,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["reference_id", "modality", "similarity", "shared_x_min", "shared_x_max"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
