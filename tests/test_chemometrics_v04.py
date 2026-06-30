from __future__ import annotations

import numpy as np

from narcoticsense.chemometrics import (
    distance_matrix,
    run_hca,
    run_lda,
    run_pls_regression,
    run_simca,
    run_umap,
)
from narcoticsense.classical_ml import evaluate_classifier
from narcoticsense.validation import binary_roc_pr_tables, reliability_table


def _toy_classification_data():
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 0.05, size=(6, 20))
    x1 = rng.normal(1, 0.05, size=(6, 20))
    matrix = np.vstack([x0, x1])
    labels = ["A"] * 6 + ["B"] * 6
    sample_ids = [f"s{i}" for i in range(12)]
    return matrix, labels, sample_ids


def test_lda_simca_hca_distance_and_umap_fallback():
    matrix, labels, sample_ids = _toy_classification_data()
    lda = run_lda(matrix, sample_ids, labels)
    assert {"sample_id", "label", "LD1"}.issubset(lda.coordinates.columns)

    simca = run_simca(matrix, sample_ids, labels)
    assert "predicted_label" in simca.predictions.columns
    assert float(simca.metrics.loc[simca.metrics.metric == "classes_modeled", "value"].iloc[0]) >= 2

    hca = run_hca(matrix, sample_ids)
    assert len(hca) == len(sample_ids) - 1

    dist = distance_matrix(matrix, sample_ids)
    assert dist.shape == (len(sample_ids), len(sample_ids))
    assert np.allclose(np.diag(dist), 0)

    manifold = run_umap(matrix, sample_ids)
    assert {"UMAP1", "UMAP2", "method"}.issubset(manifold.columns)


def test_pls_regression_outputs_metrics_and_predictions():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(8, 30))
    targets = np.linspace(0, 10, 8)
    sample_ids = [f"r{i}" for i in range(8)]
    result = run_pls_regression(matrix, sample_ids, targets.tolist(), n_components=2)
    assert len(result.predictions) == 8
    assert {"rmse_cv", "mae_cv", "r2_cv"}.issubset(set(result.metrics["metric"]))
    assert len(result.coefficients) == matrix.shape[1]


def test_supervised_probabilities_and_validation_tables():
    matrix, labels, sample_ids = _toy_classification_data()
    result = evaluate_classifier(matrix, labels, sample_ids, model_name="Random Forest", cv=3)
    assert result.probabilities is not None
    classes = ["A", "B"]
    prob = result.probabilities[["prob_A", "prob_B"]].to_numpy()
    reliability = reliability_table(labels, prob, classes)
    assert {"mean_confidence", "fraction_correct"}.issubset(reliability.columns)
    curves = binary_roc_pr_tables(labels, prob, classes)
    assert "roc_A" in curves
    assert "pr_B" in curves
