from __future__ import annotations

from pathlib import Path

import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum


def spectra_to_long_dataframe(spectra: list[Spectrum]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for spectrum in spectra:
        frame = pd.DataFrame({"sample_id": spectrum.sample_id, "x": spectrum.x, "y": spectrum.y})
        frame["modality"] = spectrum.modality
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["sample_id", "modality", "x", "y"])
    return pd.concat(rows, ignore_index=True)


def write_spectra_long_csv(spectra: list[Spectrum], path: str | Path) -> None:
    spectra_to_long_dataframe(spectra).to_csv(path, index=False)
