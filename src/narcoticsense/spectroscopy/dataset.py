from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum


@dataclass(slots=True)
class AlignedDataset:
    x: np.ndarray
    matrix: np.ndarray
    sample_ids: list[str]
    modalities: list[str]
    metadata: list[dict]

    @property
    def n_samples(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.matrix, columns=[f"x_{v:.6g}" for v in self.x])
        df.insert(0, "sample_id", self.sample_ids)
        df.insert(1, "modality", self.modalities)
        return df


def align_spectra(spectra: list[Spectrum], *, n_points: int = 1000) -> AlignedDataset:
    """Interpolate spectra onto a common x-axis for overlays and ML.

    Uses the overlapping x-range shared by all spectra. This avoids extrapolating
    outside a measurement range when files come from different instruments.
    """
    if not spectra:
        raise ValueError("At least one spectrum is required.")
    start = max(float(np.min(s.x)) for s in spectra)
    stop = min(float(np.max(s.x)) for s in spectra)
    if start >= stop:
        raise ValueError("Spectra do not share an overlapping x-axis range.")
    x_common = np.linspace(start, stop, int(n_points))
    rows = []
    sample_ids = []
    modalities = []
    metadata = []
    for i, spectrum in enumerate(spectra, start=1):
        order = np.argsort(spectrum.x)
        y = np.interp(x_common, spectrum.x[order], spectrum.y[order])
        rows.append(y)
        sample_ids.append(spectrum.sample_id or f"sample-{i:03d}")
        modalities.append(spectrum.modality)
        metadata.append(dict(spectrum.metadata))
    return AlignedDataset(
        x=x_common,
        matrix=np.vstack(rows),
        sample_ids=sample_ids,
        modalities=modalities,
        metadata=metadata,
    )


def dataset_summary(spectra: list[Spectrum]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(spectra, start=1):
        rows.append({
            "index": i,
            "sample_id": s.sample_id or f"sample-{i:03d}",
            "modality": s.modality,
            "points": s.n_points,
            "x_min": float(np.min(s.x)),
            "x_max": float(np.max(s.x)),
            "y_min": float(np.min(s.y)),
            "y_max": float(np.max(s.y)),
            "y_mean": float(np.mean(s.y)),
        })
    return pd.DataFrame(rows)
