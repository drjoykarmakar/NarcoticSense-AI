from narcoticsense.feature_engineering import PeakFeatureExtractor
from narcoticsense.spectroscopy import Spectrum


def test_peak_features():
    spectrum = Spectrum(x=[1, 2, 3, 4, 5], y=[0, 1, 0, 2, 0], modality="raman")
    features = PeakFeatureExtractor(prominence=0.1).extract_one(spectrum)
    assert features["n_peaks"] >= 1
