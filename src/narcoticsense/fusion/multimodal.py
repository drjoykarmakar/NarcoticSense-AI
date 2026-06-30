from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FusionResult:
    """Sample-aligned spectral and molecular blocks ready for multimodal ML."""

    sample_ids: list[str]
    spectral: np.ndarray
    molecular: np.ndarray
    fused: np.ndarray
    spectral_feature_names: list[str]
    molecular_feature_names: list[str]


def _zscore_block(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return matrix
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[std == 0] = 1.0
    out = (matrix - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def early_fusion_matrix(
    spectral: np.ndarray, molecular: np.ndarray, *, scale: bool = True
) -> np.ndarray:
    """Concatenate spectral and molecular feature blocks with optional block scaling."""

    spectral = np.asarray(spectral, dtype=float)
    molecular = np.asarray(molecular, dtype=float)
    if spectral.shape[0] != molecular.shape[0]:
        raise ValueError("spectral and molecular blocks must contain the same number of samples")
    if scale:
        spectral = _zscore_block(spectral)
        molecular = _zscore_block(molecular)
    return np.hstack([spectral, molecular])


def align_spectral_and_molecular(
    spectral_matrix: np.ndarray,
    sample_ids: list[str],
    molecular_table: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    molecular_columns: list[str] | None = None,
    scale: bool = True,
) -> FusionResult:
    """Align spectral matrix rows with molecular descriptors by sample_id."""

    if sample_id_col not in molecular_table.columns:
        raise ValueError(f"molecular table must contain {sample_id_col!r}")
    if molecular_columns is None:
        molecular_columns = [
            c
            for c in molecular_table.columns
            if c != sample_id_col and pd.api.types.is_numeric_dtype(molecular_table[c])
        ]
    lookup = molecular_table.set_index(sample_id_col)
    keep_ids = [sid for sid in sample_ids if sid in lookup.index]
    if not keep_ids:
        raise ValueError("no matching sample_id values between spectra and molecular table")
    row_indices = [sample_ids.index(sid) for sid in keep_ids]
    spectral = np.asarray(spectral_matrix, dtype=float)[row_indices]
    molecular = lookup.loc[keep_ids, molecular_columns].astype(float).to_numpy()
    fused = early_fusion_matrix(spectral, molecular, scale=scale)
    spectral_feature_names = [f"spectral_{i:04d}" for i in range(spectral.shape[1])]
    return FusionResult(
        sample_ids=keep_ids,
        spectral=spectral,
        molecular=molecular,
        fused=fused,
        spectral_feature_names=spectral_feature_names,
        molecular_feature_names=list(molecular_columns),
    )


def block_summary(result: FusionResult) -> pd.DataFrame:
    """Return a concise summary of multimodal feature blocks."""

    return pd.DataFrame(
        [
            {
                "block": "spectral",
                "samples": len(result.sample_ids),
                "features": result.spectral.shape[1],
            },
            {
                "block": "molecular",
                "samples": len(result.sample_ids),
                "features": result.molecular.shape[1],
            },
            {
                "block": "early_fusion",
                "samples": len(result.sample_ids),
                "features": result.fused.shape[1],
            },
        ]
    )
