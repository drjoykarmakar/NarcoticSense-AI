from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from narcoticsense.spectroscopy import Spectrum


@dataclass(slots=True)
class PeakFeatureExtractor:
    prominence: float = 0.05
    max_peaks: int = 20

    def extract_one(self, spectrum: Spectrum) -> dict[str, float]:
        peaks, props = find_peaks(spectrum.y, prominence=self.prominence)
        order = np.argsort(spectrum.y[peaks])[::-1][: self.max_peaks]
        selected = peaks[order]
        features: dict[str, float] = {
            "n_peaks": float(len(peaks)),
            "max_intensity": float(np.max(spectrum.y)),
            "mean_intensity": float(np.mean(spectrum.y)),
            "area_abs": float(np.trapezoid(np.abs(spectrum.y), spectrum.x)),
        }
        for i, idx in enumerate(selected):
            features[f"peak_{i}_position"] = float(spectrum.x[idx])
            features[f"peak_{i}_height"] = float(spectrum.y[idx])
        return features

    def transform(self, spectra: list[Spectrum]) -> list[dict[str, float]]:
        return [self.extract_one(s) for s in spectra]
