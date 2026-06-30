from __future__ import annotations

import numpy as np

from narcoticsense.feature_engineering import peak_table
from narcoticsense.preprocessing import PreprocessingPipeline, airpls, arpls, estimate_baseline
from narcoticsense.spectroscopy import Spectrum, spectra_to_long_dataframe


def _synthetic_spectrum() -> Spectrum:
    x = np.linspace(400, 700, 301)
    baseline = 0.001 * (x - 400) + 0.2
    peak = np.exp(-0.5 * ((x - 520) / 8) ** 2)
    return Spectrum(x=x, y=baseline + peak, modality="fluorescence", sample_id="synthetic")


def test_baseline_methods_return_finite_arrays() -> None:
    spectrum = _synthetic_spectrum()
    for method in ["asls", "airpls", "arpls"]:
        baseline = estimate_baseline(spectrum.y, method=method, lam=1e4)
        assert baseline.shape == spectrum.y.shape
        assert np.isfinite(baseline).all()
    assert np.isfinite(airpls(spectrum.y, lam=1e4)).all()
    assert np.isfinite(arpls(spectrum.y, lam=1e4)).all()


def test_pipeline_records_baseline_method() -> None:
    spectrum = _synthetic_spectrum()
    pipe = PreprocessingPipeline(baseline_method="arpls", baseline_lambda=1e4)
    processed = pipe.transform_one(spectrum)
    assert processed.metadata["baseline_method"] == "arpls"
    assert "baseline" in processed.metadata
    assert len(pipe.history()) >= 1


def test_peak_table_reports_fwhm_and_area() -> None:
    spectrum = _synthetic_spectrum()
    peaks = peak_table(spectrum, prominence=0.1)
    assert not peaks.empty
    assert "fwhm_x_units" in peaks.columns
    assert "area_fwhm_window" in peaks.columns
    assert peaks.iloc[0]["fwhm_x_units"] > 0


def test_spectra_to_long_dataframe() -> None:
    spectra = [_synthetic_spectrum(), _synthetic_spectrum().copy_with(y=np.ones(301))]
    frame = spectra_to_long_dataframe(spectra)
    assert set(["sample_id", "modality", "x", "y"]).issubset(frame.columns)
    assert len(frame) == 602
