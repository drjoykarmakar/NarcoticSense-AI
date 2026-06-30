import numpy as np

from narcoticsense.chemometrics import run_kmeans, run_pca
from narcoticsense.feature_engineering import derivative_spectrum, spectral_metrics
from narcoticsense.spectroscopy import Spectrum, align_spectra, dataset_summary


def make_spectrum(sample_id, shift=0.0):
    x = np.linspace(200, 300, 80)
    y = np.sin(x / 20.0) + shift
    return Spectrum(x=x, y=y, modality="fluorescence", sample_id=sample_id)


def test_align_summary_and_metrics():
    spectra = [make_spectrum("a"), make_spectrum("b", 0.1)]
    aligned = align_spectra(spectra, n_points=50)
    assert aligned.matrix.shape == (2, 50)
    summary = dataset_summary(spectra)
    assert len(summary) == 2
    metrics = spectral_metrics(spectra[0])
    assert "absolute area" in metrics["metric"].tolist()
    deriv = derivative_spectrum(spectra[0], order=1)
    assert deriv.y.shape == spectra[0].y.shape


def test_pca_and_kmeans():
    spectra = [make_spectrum(f"s{i}", i * 0.05) for i in range(4)]
    aligned = align_spectra(spectra, n_points=30)
    pca = run_pca(aligned.matrix, aligned.sample_ids, aligned.x, n_components=2)
    assert list(pca.coordinates.columns)[:3] == ["sample_id", "PC1", "PC2"]
    clusters = run_kmeans(aligned.matrix, aligned.sample_ids, n_clusters=2)
    assert len(clusters) == 4
