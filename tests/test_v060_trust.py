import numpy as np

from narcoticsense.explainability import peak_level_attribution, spectral_occlusion_importance
from narcoticsense.feature_engineering import peak_table
from narcoticsense.models import build_classifier
from narcoticsense.spectroscopy import Spectrum
from narcoticsense.uncertainty import (
    cross_validated_model_comparison,
    probability_confidence_table,
    spectral_ood_scores,
    trustworthy_classification_report,
)


def _toy_data():
    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 0.1, (8, 30))
    x1 = rng.normal(0, 0.1, (8, 30))
    x0[:, 5:9] += 1.0
    x1[:, 20:24] += 1.0
    x = np.vstack([x0, x1])
    y = ["A"] * 8 + ["B"] * 8
    ids = [f"s{i}" for i in range(len(y))]
    return x, y, ids


def test_probability_confidence_and_ood_scores():
    x, y, ids = _toy_data()
    model = build_classifier("Random Forest")
    model.fit(x, y)
    conf = probability_confidence_table(model, x[:3], ids[:3])
    assert {"sample_id", "prediction", "confidence", "margin"}.issubset(conf.columns)
    ood = spectral_ood_scores(x, x[:3], ids[:3])
    assert {"ood_score", "ood_threshold", "is_ood"}.issubset(ood.columns)


def test_trustworthy_classification_report_outputs_flags():
    x, y, ids = _toy_data()
    model = build_classifier("Random Forest")
    trust = trustworthy_classification_report(model, x, y, x, ids, confidence_threshold=0.5)
    assert not trust.predictions.empty
    assert "refer_to_confirmatory_testing" in trust.predictions.columns
    assert not trust.thresholds.empty
    assert not trust.summary.empty


def test_cross_validated_model_comparison():
    x, y, _ = _toy_data()
    builders = {
        "Random Forest": lambda: build_classifier("Random Forest"),
        "SVM": lambda: build_classifier("SVM RBF"),
    }
    comparison = cross_validated_model_comparison(builders, x, y, n_splits=4)
    assert set(comparison["model"]) == {"Random Forest", "SVM"}
    assert "macro_f1" in comparison.columns


def test_spectral_occlusion_and_peak_attribution():
    x, y, _ = _toy_data()
    model = build_classifier("Random Forest")
    model.fit(x, y)
    x_axis = np.linspace(400, 700, x.shape[1])
    attr = spectral_occlusion_importance(model, x[0], x_axis=x_axis, window_size=5)
    assert {"x", "importance", "relative_importance"}.issubset(attr.columns)
    spec = Spectrum(x=x_axis, y=x[0], modality="fluorescence", sample_id="s0")
    peaks = peak_table(spec, prominence=0.2, max_peaks=10)
    peak_attr = peak_level_attribution(attr, peaks)
    assert "peak_x" in peak_attr.columns or peak_attr.empty
