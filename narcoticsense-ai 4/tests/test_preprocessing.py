from narcoticsense.preprocessing import PreprocessingPipeline
from narcoticsense.spectroscopy import Spectrum


def test_preprocessing_preserves_length():
    spectrum = Spectrum(x=list(range(30)), y=[float(i % 5) for i in range(30)], modality="raman")
    processed = PreprocessingPipeline.default().transform_one(spectrum)
    assert processed.n_points == spectrum.n_points
