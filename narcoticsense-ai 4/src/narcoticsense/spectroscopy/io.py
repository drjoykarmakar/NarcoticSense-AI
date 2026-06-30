from __future__ import annotations

from pathlib import Path

import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum


def read_csv_spectrum(path: str | Path, *, x_col: str = "x", y_col: str = "y", modality: str = "unknown") -> Spectrum:
    df = pd.read_csv(path)
    missing = {x_col, y_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return Spectrum(x=df[x_col].to_numpy(), y=df[y_col].to_numpy(), modality=modality, metadata={"source": str(path)})


def write_csv_spectrum(spectrum: Spectrum, path: str | Path) -> None:
    pd.DataFrame({"x": spectrum.x, "y": spectrum.y}).to_csv(path, index=False)
