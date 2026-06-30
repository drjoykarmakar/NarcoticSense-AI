from __future__ import annotations

import numpy as np

from narcoticsense.spectroscopy import Spectrum


def peak_level_attribution(spectrum: Spectrum, top_k: int = 10) -> list[dict[str, float]]:
    order = np.argsort(np.abs(spectrum.y))[::-1][:top_k]
    return [
        {"position": float(spectrum.x[i]), "importance": float(abs(spectrum.y[i]))} for i in order
    ]
