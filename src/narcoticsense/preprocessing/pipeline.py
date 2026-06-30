from __future__ import annotations

from dataclasses import dataclass

from scipy.signal import savgol_filter

from narcoticsense.preprocessing.baseline import asymmetric_least_squares
from narcoticsense.preprocessing.normalization import minmax, standard_normal_variate
from narcoticsense.spectroscopy import Spectrum


@dataclass(slots=True)
class PreprocessingPipeline:
    smooth: bool = True
    baseline: bool = True
    normalization: str = "minmax"
    window_length: int = 11
    polyorder: int = 3

    @classmethod
    def default(cls) -> "PreprocessingPipeline":
        return cls()

    def transform_one(self, spectrum: Spectrum) -> Spectrum:
        y = spectrum.y.copy()
        if self.smooth and len(y) >= self.window_length:
            y = savgol_filter(y, self.window_length, self.polyorder)
        if self.baseline:
            y = y - asymmetric_least_squares(y)
        if self.normalization == "minmax":
            y = minmax(y)
        elif self.normalization == "snv":
            y = standard_normal_variate(y)
        return spectrum.copy_with(y=y, preprocessing=self.__class__.__name__)

    def transform(self, spectra: list[Spectrum]) -> list[Spectrum]:
        return [self.transform_one(s) for s in spectra]
