from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from narcoticsense.spectroscopy.core import Spectrum

X_KEYWORDS = [
    "x",
    "wavelength",
    "wave length",
    "lambda",
    "nm",
    "raman shift",
    "shift",
    "wavenumber",
    "wave number",
    "cm-1",
    "cm^-1",
    "cm⁻¹",
]
Y_KEYWORDS = [
    "y",
    "intensity",
    "counts",
    "abs",
    "absorbance",
    "transmittance",
    "fluorescence",
    "emission",
    "reflectance",
    "signal",
    "au",
]


@dataclass(slots=True)
class ImportResult:
    """Result from universal spectroscopy import."""

    spectrum: Spectrum
    dataframe: pd.DataFrame
    raw_dataframe: pd.DataFrame
    x_column: str
    y_column: str
    header_row: int
    messages: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def summary(self) -> str:
        direction = "increasing" if self.spectrum.x[0] <= self.spectrum.x[-1] else "decreasing"
        return (
            f"Imported {self.spectrum.n_points} points using x='{self.x_column}', y='{self.y_column}', "
            f"header row {self.header_row}, x-axis {direction}."
        )


def _read_text_from_source(source: str | Path | bytes | BinaryIO) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8-sig", errors="replace")
    if isinstance(source, (str, Path)):
        return Path(source).read_text(encoding="utf-8-sig", errors="replace")
    data = source.read()
    if isinstance(data, str):
        return data
    return data.decode("utf-8-sig", errors="replace")


def _keyword_score(name: Any, keywords: list[str]) -> int:
    text = str(name).strip().lower()
    return sum(1 for kw in keywords if kw in text)


def _numeric_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.notna().mean())


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(axis=1, how="all")
    keep = []
    for col in out.columns:
        values = (
            out[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
        if values.notna().any():
            keep.append(col)
    return out[keep]


def _candidate_read(text: str, header: int, sep: str | None = None) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(StringIO(text), header=header, sep=sep, engine="python")
    except Exception:
        return None
    df = _drop_empty_columns(df)
    if df.empty or len(df.columns) < 2:
        return None
    return df


def _choose_header(text: str) -> tuple[pd.DataFrame, int, list[str]]:
    messages: list[str] = []
    candidates: list[tuple[float, int, pd.DataFrame]] = []
    for header in range(0, min(12, len(text.splitlines()))):
        for sep in [None, ",", "\t", ";"]:
            df = _candidate_read(text, header=header, sep=sep)
            if df is None:
                continue
            cols = list(df.columns)
            keyword_hits = sum(_keyword_score(c, X_KEYWORDS + Y_KEYWORDS) for c in cols)
            numeric_cols = sum(_numeric_fraction(df[c]) > 0.75 for c in cols)
            nonempty = min(len(df), 50)
            score = keyword_hits * 4 + numeric_cols * 2 + nonempty * 0.02 - abs(len(cols) - 2) * 0.2
            candidates.append((score, header, df))
    if not candidates:
        raise ValueError("Could not read this file as a numeric spectroscopy table.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, header, df = candidates[0]
    if header > 0:
        messages.append(
            f"Detected metadata/title rows above the table and used row {header + 1} as the header."
        )
    return df, header, messages


def _choose_xy_columns(df: pd.DataFrame) -> tuple[str, str, float, list[str]]:
    messages: list[str] = []
    numeric_cols = [c for c in df.columns if _numeric_fraction(df[c]) > 0.7]
    if len(numeric_cols) < 2:
        raise ValueError("Could not find two numeric columns for x and y values.")

    x_scores: dict[Any, float] = {}
    y_scores: dict[Any, float] = {}
    for col in numeric_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        monotonic = float(numeric.is_monotonic_increasing or numeric.is_monotonic_decreasing)
        unique_ratio = float(numeric.nunique(dropna=True) / max(numeric.notna().sum(), 1))
        x_scores[col] = 3 * _keyword_score(col, X_KEYWORDS) + 1.5 * monotonic + unique_ratio
        y_scores[col] = 3 * _keyword_score(col, Y_KEYWORDS) + float(numeric.std(skipna=True) > 0)

    x_col = max(numeric_cols, key=lambda c: x_scores[c])
    y_candidates = [c for c in numeric_cols if c != x_col]
    y_col = max(y_candidates, key=lambda c: y_scores[c])

    confidence = min(1.0, (x_scores[x_col] + y_scores[y_col]) / 10.0)
    if confidence < 0.55:
        messages.append(
            "Automatic column mapping had moderate confidence; please verify x and y columns visually."
        )
    return str(x_col), str(y_col), confidence, messages


def import_spectrum(
    source: str | Path | bytes | BinaryIO,
    *,
    modality: str = "unknown",
    sample_id: str | None = None,
    sort_axis: bool = True,
) -> ImportResult:
    """Import common CSV/TXT spectroscopy exports into a standard Spectrum.

    Handles simple x/y CSV files and vendor-like exports with title rows, blank
    columns, column names such as 'Wavelength (nm)'/'Abs', and descending axes.
    """
    text = _read_text_from_source(source)
    raw_df = pd.read_csv(StringIO(text), header=None, engine="python")
    df, header_row, messages = _choose_header(text)
    x_col, y_col, confidence, map_messages = _choose_xy_columns(df)
    messages.extend(map_messages)

    clean = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    ).dropna()
    if len(clean) < 2:
        raise ValueError("The selected x/y columns do not contain enough numeric data.")
    clean = clean.drop_duplicates(subset=["x"], keep="first")

    if sort_axis and clean["x"].iloc[0] > clean["x"].iloc[-1]:
        clean = clean.sort_values("x", ascending=True).reset_index(drop=True)
        messages.append(
            "Detected descending x-axis and sorted it into increasing order for analysis."
        )

    metadata = {
        "importer": "universal",
        "source_x_column": x_col,
        "source_y_column": y_col,
        "header_row": header_row,
        "import_messages": messages,
        "import_confidence": confidence,
    }
    spectrum = Spectrum(
        x=clean["x"].to_numpy(),
        y=clean["y"].to_numpy(),
        modality=modality,
        sample_id=sample_id,
        metadata=metadata,
    )
    return ImportResult(
        spectrum=spectrum,
        dataframe=clean,
        raw_dataframe=raw_df,
        x_column=x_col,
        y_column=y_col,
        header_row=header_row,
        messages=messages,
        confidence=confidence,
    )
