from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import pandas as pd

try:  # optional dependency; the app remains useful without RDKit installed
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski
    from rdkit.Chem import rdMolDescriptors as rd_mol_descriptors  # noqa: N813

    _HAS_RDKIT = True
except Exception:  # pragma: no cover - depends on optional environment
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rd_mol_descriptors = None
    _HAS_RDKIT = False


@dataclass(slots=True)
class MoleculeFeatureResult:
    """Container for molecular descriptors/fingerprints derived from SMILES."""

    table: pd.DataFrame
    descriptor_columns: list[str]
    fingerprint_columns: list[str]
    rdkit_used: bool
    warnings: list[str]


def rdkit_available() -> bool:
    """Return True when RDKit is importable in the current Python environment."""

    return _HAS_RDKIT


def _stable_hash_feature(smiles: str, bits: int = 128) -> np.ndarray:
    """Deterministic fallback fingerprint when RDKit is unavailable or parsing fails."""

    fp = np.zeros(bits, dtype=float)
    tokens = [smiles[i : i + 2] for i in range(max(1, len(smiles) - 1))]
    tokens += list(smiles)
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "little") % bits
        fp[idx] = 1.0
    return fp


def _fallback_descriptors(smiles: str) -> dict[str, float]:
    atoms = sum(ch.isalpha() and ch.isupper() for ch in smiles)
    hetero = sum(ch in "NOSPFClBrIclbr" for ch in smiles)
    rings = sum(ch.isdigit() for ch in smiles)
    branches = smiles.count("(") + smiles.count(")")
    aromatic = sum(ch in "cnosp" for ch in smiles)
    return {
        "mol_weight": float("nan"),
        "logp": float("nan"),
        "h_donors": float("nan"),
        "h_acceptors": float("nan"),
        "tpsa": float("nan"),
        "rotatable_bonds": float("nan"),
        "atom_count_proxy": float(atoms),
        "heteroatom_proxy": float(hetero),
        "ring_token_proxy": float(rings),
        "branch_token_proxy": float(branches),
        "aromatic_token_proxy": float(aromatic),
    }


def compute_molecular_features(smiles: str, *, fingerprint_bits: int = 128) -> dict[str, float]:
    """Compute molecular descriptors plus a compact fingerprint from one SMILES string.

    RDKit is used when available. If RDKit is not installed or a SMILES string cannot be
    parsed, the function returns safe fallback features so that demos and UI workflows do
    not fail. The fallback features are not chemically equivalent to RDKit descriptors and
    should be used only for software testing or exploratory workflows.
    """

    smiles = "" if smiles is None else str(smiles).strip()
    features: dict[str, float] = {}
    if _HAS_RDKIT and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features.update(
                {
                    "mol_weight": float(Descriptors.MolWt(mol)),
                    "logp": float(Crippen.MolLogP(mol)),
                    "h_donors": float(Lipinski.NumHDonors(mol)),
                    "h_acceptors": float(Lipinski.NumHAcceptors(mol)),
                    "tpsa": float(rd_mol_descriptors.CalcTPSA(mol)),
                    "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                    "atom_count_proxy": float(mol.GetNumAtoms()),
                    "heteroatom_proxy": float(
                        sum(atom.GetAtomicNum() not in {1, 6} for atom in mol.GetAtoms())
                    ),
                    "ring_token_proxy": float(rd_mol_descriptors.CalcNumRings(mol)),
                    "branch_token_proxy": float(smiles.count("(") + smiles.count(")")),
                    "aromatic_token_proxy": float(
                        sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
                    ),
                }
            )
            fp = rd_mol_descriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=fingerprint_bits)
            arr = np.asarray(list(fp.ToBitString()), dtype=float)
        else:
            features.update(_fallback_descriptors(smiles))
            arr = _stable_hash_feature(smiles, fingerprint_bits)
    else:
        features.update(_fallback_descriptors(smiles))
        arr = _stable_hash_feature(smiles, fingerprint_bits)

    for idx, value in enumerate(arr):
        features[f"fp_{idx:03d}"] = float(value)
    return features


def build_molecular_feature_table(
    metadata: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    smiles_col: str = "smiles",
    fingerprint_bits: int = 128,
) -> MoleculeFeatureResult:
    """Build a sample-aligned molecular feature table from metadata containing SMILES."""

    if sample_id_col not in metadata.columns:
        raise ValueError(f"metadata must contain a {sample_id_col!r} column")
    if smiles_col not in metadata.columns:
        raise ValueError(f"metadata must contain a {smiles_col!r} column")

    rows = []
    warnings: list[str] = []
    for _, row in metadata.iterrows():
        sample_id = str(row[sample_id_col])
        smiles = "" if pd.isna(row[smiles_col]) else str(row[smiles_col]).strip()
        if not smiles:
            warnings.append(f"{sample_id}: missing SMILES; fallback zero-like features used")
        features = compute_molecular_features(smiles, fingerprint_bits=fingerprint_bits)
        features[sample_id_col] = sample_id
        features[smiles_col] = smiles
        rows.append(features)

    table = pd.DataFrame(rows)
    descriptor_columns = [
        c for c in table.columns if c not in {sample_id_col, smiles_col} and not c.startswith("fp_")
    ]
    fingerprint_columns = [c for c in table.columns if c.startswith("fp_")]
    return MoleculeFeatureResult(
        table=table,
        descriptor_columns=descriptor_columns,
        fingerprint_columns=fingerprint_columns,
        rdkit_used=_HAS_RDKIT,
        warnings=warnings,
    )
