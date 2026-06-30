from __future__ import annotations

import numpy as np

from narcoticsense.classical_ml import evaluate_classifier
from narcoticsense.library import library_match
from narcoticsense.spectroscopy import Spectrum
from narcoticsense.validation import dataset_quality_report, spectrum_quality_report


def make_spectrum(sample_id: str, center: float) -> Spectrum:
    x = np.linspace(200, 400, 120)
    y = np.exp(-0.5 * ((x - center) / 8) ** 2)
    return Spectrum(x=x, y=y, modality="fluorescence", sample_id=sample_id)


def test_quality_report_has_statuses() -> None:
    spec = make_spectrum("a", 260)
    report = spectrum_quality_report(spec)
    assert {"check", "status", "detail"}.issubset(report.columns)
    assert "pass" in set(report["status"])
    dataset_report = dataset_quality_report([spec])
    assert dataset_report.loc[0, "sample_id"] == "a"


def test_library_match_ranks_similar_spectrum_first() -> None:
    query = make_spectrum("query", 260)
    ref_close = make_spectrum("close", 261)
    ref_far = make_spectrum("far", 330)
    matches = library_match(query, [ref_far, ref_close], top_k=2)
    assert matches.loc[0, "reference_id"] == "close"
    assert matches.loc[0, "similarity"] > matches.loc[1, "similarity"]


def test_evaluate_classifier_returns_metrics() -> None:
    rng = np.random.default_rng(42)
    class_a = rng.normal(0, 0.1, size=(6, 20))
    class_b = rng.normal(2, 0.1, size=(6, 20))
    matrix = np.vstack([class_a, class_b])
    labels = ["A"] * 6 + ["B"] * 6
    sample_ids = [f"s{i}" for i in range(12)]
    result = evaluate_classifier(matrix, labels, sample_ids, cv=3)
    assert "accuracy" in set(result.metrics["metric"])
    assert len(result.predictions) == 12
