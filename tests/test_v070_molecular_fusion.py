from __future__ import annotations

import numpy as np
import pandas as pd

from narcoticsense.fusion import align_spectral_and_molecular, block_summary, early_fusion_matrix
from narcoticsense.molecular import build_molecular_feature_table, compute_molecular_features


def test_compute_molecular_features_returns_descriptors_and_fingerprint() -> None:
    features = compute_molecular_features("CCO", fingerprint_bits=32)
    assert "mol_weight" in features
    assert "logp" in features
    fp_cols = [key for key in features if key.startswith("fp_")]
    assert len(fp_cols) == 32


def test_build_molecular_feature_table_from_metadata() -> None:
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "smiles": ["CCO", "c1ccccc1"],
            "compound_or_class": ["a", "b"],
        }
    )
    result = build_molecular_feature_table(metadata, fingerprint_bits=16)
    assert result.table.shape[0] == 2
    assert "sample_id" in result.table.columns
    assert len(result.fingerprint_columns) == 16
    assert "mol_weight" in result.descriptor_columns


def test_early_fusion_matrix_shapes() -> None:
    spectral = np.ones((3, 5))
    molecular = np.ones((3, 2))
    fused = early_fusion_matrix(spectral, molecular)
    assert fused.shape == (3, 7)


def test_align_spectral_and_molecular_by_sample_id() -> None:
    spectral = np.arange(12, dtype=float).reshape(3, 4)
    sample_ids = ["s1", "s2", "s3"]
    mol = pd.DataFrame(
        {
            "sample_id": ["s3", "s1"],
            "d1": [1.0, 2.0],
            "d2": [3.0, 4.0],
        }
    )
    result = align_spectral_and_molecular(spectral, sample_ids, mol)
    assert result.sample_ids == ["s1", "s3"]
    assert result.spectral.shape == (2, 4)
    assert result.molecular.shape == (2, 2)
    assert result.fused.shape == (2, 6)
    summary = block_summary(result)
    assert set(summary["block"]) == {"spectral", "molecular", "early_fusion"}
