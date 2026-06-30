from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum


@dataclass(slots=True)
class UploadResult:
    spectrum: Spectrum | None
    filename: str
    error: str | None = None


def spectrum_from_dataframe(
    df: pd.DataFrame,
    *,
    modality: str = "unknown",
    sample_id: str | None = None,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str | None = None,
) -> Spectrum:
    missing = {x_col, y_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    metadata = {}
    if label_col and label_col in df.columns:
        labels = df[label_col].dropna().unique().tolist()
        if labels:
            metadata["label"] = str(labels[0])
    return Spectrum(
        x=df[x_col].to_numpy(),
        y=df[y_col].to_numpy(),
        modality=modality,
        sample_id=sample_id,
        metadata=metadata,
    )


def read_uploaded_csv(
    file_obj, *, modality: str = "unknown", sample_id: str | None = None
) -> UploadResult:
    name = getattr(file_obj, "name", "uploaded.csv")
    try:
        content = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
        df = pd.read_csv(BytesIO(content))
        spectrum = spectrum_from_dataframe(df, modality=modality, sample_id=sample_id or name)
        return UploadResult(spectrum=spectrum, filename=name)
    except Exception as exc:  # pragma: no cover - UI helper
        return UploadResult(spectrum=None, filename=name, error=str(exc))
