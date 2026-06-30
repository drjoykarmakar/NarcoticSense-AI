import pytest

from narcoticsense.spectroscopy import Spectrum


def test_spectrum_validates_shape():
    with pytest.raises(ValueError):
        Spectrum(x=[1, 2], y=[1], modality="raman")


def test_spectrum_points():
    spectrum = Spectrum(x=[1, 2, 3], y=[0.1, 0.2, 0.3], modality="raman")
    assert spectrum.n_points == 3
