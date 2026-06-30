from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from narcoticsense.models.engine import (
    ModelTrainingResult,
    build_classifier,
    train_classification_model,
)
from narcoticsense.spectroscopy.dataset import AlignedDataset

UNKNOWN_MARKERS = {"", "unknown", "unk", "unlabeled", "na", "n/a", "none", "?"}


@dataclass(slots=True)
class RealAIWorkflowResult:
    """Result for the controlled training workflow.

    The validation result is computed with a train/test split, while the saved
    inference model is refit on all approved labeled spectra. This makes the
    prediction model useful while preserving a validation report for transparency.
    """

    validation: ModelTrainingResult
    model_path: Path
    training_table: pd.DataFrame
    bundle_metadata: dict[str, Any]


@dataclass(slots=True)
class PredictionWorkflowResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame


def _is_labeled(value: Any) -> bool:
    text = str(value).strip().lower()
    return text not in UNKNOWN_MARKERS


def prepare_training_table(
    sample_ids: list[str],
    metadata: pd.DataFrame,
    label_column: str = "compound_or_class",
) -> pd.DataFrame:
    """Merge sample IDs with metadata and mark labeled vs unknown samples."""
    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata must contain a sample_id column.")
    if label_column not in metadata.columns:
        raise ValueError(f"Metadata must contain the label column: {label_column}")
    base = pd.DataFrame({"sample_id": [str(s) for s in sample_ids]})
    table = base.merge(metadata, on="sample_id", how="left")
    labels = table[label_column].fillna("").astype(str).str.strip()
    table["training_status"] = ["labeled" if _is_labeled(v) else "unknown" for v in labels]
    table["ai_label"] = labels
    table.loc[table["training_status"] == "unknown", "ai_label"] = ""
    return table


def validate_training_table(
    training_table: pd.DataFrame,
    label_column: str = "compound_or_class",
    min_classes: int = 2,
    min_samples: int = 4,
) -> pd.DataFrame:
    """Return human-readable checks before model training."""
    rows: list[dict[str, Any]] = []
    n_total = len(training_table)
    labeled = training_table[training_table["training_status"] == "labeled"]
    n_labeled = len(labeled)
    n_unknown = n_total - n_labeled
    class_counts = labeled["ai_label"].value_counts()
    n_classes = int(class_counts.shape[0])
    min_count = int(class_counts.min()) if n_classes else 0
    rows.extend(
        [
            {"check": "total spectra", "value": n_total, "status": "ok" if n_total else "fail"},
            {
                "check": "labeled spectra",
                "value": n_labeled,
                "status": "ok" if n_labeled >= min_samples else "fail",
            },
            {"check": "unknown/unlabeled spectra", "value": n_unknown, "status": "info"},
            {
                "check": "number of classes",
                "value": n_classes,
                "status": "ok" if n_classes >= min_classes else "fail",
            },
            {
                "check": "minimum replicates per class",
                "value": min_count,
                "status": "ok" if min_count >= 2 else "warning",
            },
        ]
    )
    for label, count in class_counts.items():
        rows.append(
            {
                "check": f"class count: {label}",
                "value": int(count),
                "status": "ok" if int(count) >= 2 else "warning",
            }
        )
    if label_column != "ai_label":
        rows.append({"check": "label column", "value": label_column, "status": "info"})
    return pd.DataFrame(rows)


def _safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_") or "model"


def _ood_reference(x: np.ndarray) -> dict[str, Any]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    centroid = np.mean(z, axis=0)
    distances = np.linalg.norm(z - centroid, axis=1)
    threshold = float(np.quantile(distances, 0.95)) if len(distances) > 1 else float(distances[0])
    return {
        "feature_mean": mean,
        "feature_std": std,
        "centroid_z": centroid,
        "ood_threshold": max(threshold, 1e-9),
        "training_distance_median": float(np.median(distances)),
    }


def train_save_real_ai_classifier(
    aligned: AlignedDataset,
    metadata: pd.DataFrame,
    models_dir: str | Path,
    *,
    label_column: str = "compound_or_class",
    model_name: str = "Random Forest",
    project_name: str = "NarcoticSense Project",
    test_size: float = 0.25,
    random_state: int = 42,
) -> RealAIWorkflowResult:
    """Validate labels, train, refit on all labeled spectra, and save a model bundle."""
    training_table = prepare_training_table(aligned.sample_ids, metadata, label_column=label_column)
    checks = validate_training_table(training_table, label_column=label_column)
    if (checks[checks["status"] == "fail"].shape[0]) > 0:
        raise ValueError("Training requirements are not satisfied. Review the training checks.")

    labeled_mask = training_table["training_status"].to_numpy() == "labeled"
    x_labeled = aligned.matrix[labeled_mask]
    y_labeled = training_table.loc[labeled_mask, "ai_label"].astype(str).tolist()
    ids_labeled = training_table.loc[labeled_mask, "sample_id"].astype(str).tolist()

    validation = train_classification_model(
        x_labeled,
        y_labeled,
        ids_labeled,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
    )

    inference_model = build_classifier(model_name, random_state=random_state)
    inference_model.fit(x_labeled, y_labeled)
    ood = _ood_reference(x_labeled)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = f"{_safe_slug(project_name)}_{_safe_slug(model_name)}_{created_at.replace(':', '').replace('-', '')}"
    path = Path(models_dir) / f"{slug}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_bundle: dict[str, Any] = {
        "project_name": project_name,
        "created_at": created_at,
        "task": "classification",
        "model_name": model_name,
        "label_column": label_column,
        "n_points": int(aligned.n_features),
        "x_axis": aligned.x,
        "classes": sorted(set(y_labeled)),
        "training_sample_ids": ids_labeled,
        "training_labels": y_labeled,
        "training_checks": checks,
        "validation_metrics": validation.metrics,
        "responsible_use": "Research, screening, and decision-support only; confirm high-stakes results with validated laboratory methods.",
        **ood,
    }
    bundle = {
        "model": inference_model,
        "task": "classification",
        "model_name": model_name,
        "metadata": metadata_bundle,
    }
    joblib.dump(bundle, path)
    return RealAIWorkflowResult(
        validation=validation,
        model_path=path,
        training_table=training_table,
        bundle_metadata=metadata_bundle,
    )


def list_model_registry(models_dir: str | Path) -> pd.DataFrame:
    """List saved NarcoticSense model bundles."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(models_dir).glob("*.joblib")):
        try:
            bundle = joblib.load(path)
            meta = bundle.get("metadata", {})
            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "task": bundle.get("task", meta.get("task", "unknown")),
                    "model_name": bundle.get("model_name", meta.get("model_name", "unknown")),
                    "project_name": meta.get("project_name", ""),
                    "created_at": meta.get("created_at", ""),
                    "classes": ", ".join(map(str, meta.get("classes", []))),
                    "n_points": meta.get("n_points", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {"file": path.name, "path": str(path), "task": "unreadable", "error": str(exc)}
            )
    return pd.DataFrame(rows)


def load_real_ai_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))


def predict_unknowns_with_bundle(
    bundle: dict[str, Any],
    aligned: AlignedDataset,
    *,
    confidence_threshold: float = 0.70,
) -> PredictionWorkflowResult:
    """Predict unknown spectra and attach safety-oriented decision flags."""
    model = bundle["model"]
    meta = bundle.get("metadata", {})
    pred = model.predict(aligned.matrix)
    rows: list[dict[str, Any]] = []
    probabilities = None
    classes = list(getattr(model, "classes_", meta.get("classes", [])))
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(aligned.matrix), dtype=float)
    mean = meta.get("feature_mean")
    std = meta.get("feature_std")
    centroid = meta.get("centroid_z")
    ood_threshold = float(meta.get("ood_threshold", np.inf))
    distances = [np.nan] * aligned.n_samples
    if mean is not None and std is not None and centroid is not None:
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        centroid = np.asarray(centroid, dtype=float)
        if len(mean) == aligned.n_features:
            z = (aligned.matrix - mean) / np.where(std < 1e-12, 1.0, std)
            distances = np.linalg.norm(z - centroid, axis=1).tolist()

    for i, sid in enumerate(aligned.sample_ids):
        confidence = np.nan
        margin = np.nan
        prob_map: dict[str, float] = {}
        if probabilities is not None:
            probs = probabilities[i]
            order = np.argsort(probs)[::-1]
            confidence = float(probs[order[0]])
            margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else confidence
            prob_map = {f"prob_{classes[j]}": float(probs[j]) for j in range(len(classes))}
        ood_flag = bool(np.isfinite(distances[i]) and distances[i] > ood_threshold)
        low_conf = bool(np.isfinite(confidence) and confidence < confidence_threshold)
        decision = "accept_as_screening_result"
        if low_conf or ood_flag:
            decision = "refer_to_confirmatory_testing"
        rows.append(
            {
                "sample_id": sid,
                "predicted_label": str(pred[i]),
                "confidence": confidence,
                "probability_margin": margin,
                "ood_distance": float(distances[i]) if np.isfinite(distances[i]) else np.nan,
                "ood_threshold": ood_threshold if np.isfinite(ood_threshold) else np.nan,
                "low_confidence": low_conf,
                "ood_flag": ood_flag,
                "decision_flag": decision,
                **prob_map,
            }
        )
    predictions = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"metric": "samples_predicted", "value": int(len(predictions))},
            {
                "metric": "refer_to_confirmatory_testing",
                "value": int(
                    (predictions["decision_flag"] == "refer_to_confirmatory_testing").sum()
                ),
            },
            {"metric": "model_name", "value": bundle.get("model_name", meta.get("model_name", ""))},
            {"metric": "created_at", "value": meta.get("created_at", "")},
        ]
    )
    return PredictionWorkflowResult(predictions=predictions, summary=summary)
