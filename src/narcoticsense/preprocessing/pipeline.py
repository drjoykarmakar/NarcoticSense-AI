from __future__ import annotations

from dataclasses import dataclass, field

from scipy.signal import savgol_filter

from narcoticsense.preprocessing.baseline import estimate_baseline
from narcoticsense.preprocessing.normalization import minmax, standard_normal_variate
from narcoticsense.spectroscopy import Spectrum


@dataclass(slots=True)
class PreprocessingStep:
    name: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PreprocessingPipeline:
    smooth: bool = True
    baseline: bool = True
    normalization: str = "minmax"
    window_length: int = 11
    polyorder: int = 3
    baseline_method: str = "asls"
    baseline_lambda: float = 1e5

    @classmethod
    def default(cls) -> PreprocessingPipeline:
        return cls()

    def history(self) -> list[PreprocessingStep]:
        steps: list[PreprocessingStep] = []
        if self.smooth:
            steps.append(
                PreprocessingStep(
                    "savitzky_golay_smoothing",
                    {"window_length": self.window_length, "polyorder": self.polyorder},
                )
            )
        if self.baseline:
            steps.append(
                PreprocessingStep(
                    "baseline_correction",
                    {"method": self.baseline_method, "lambda": self.baseline_lambda},
                )
            )
        if self.normalization != "none":
            steps.append(PreprocessingStep("normalization", {"method": self.normalization}))
        return steps

    def transform_one(self, spectrum: Spectrum) -> Spectrum:
        y = spectrum.y.copy()
        applied: list[str] = []
        if self.smooth and len(y) >= self.window_length:
            window = self.window_length if self.window_length % 2 == 1 else self.window_length + 1
            if window <= len(y):
                y = savgol_filter(y, window, min(self.polyorder, window - 1))
                applied.append(f"savgol(window={window}, poly={self.polyorder})")
        baseline_values = None
        if self.baseline:
            baseline_values = estimate_baseline(
                y, method=self.baseline_method, lam=self.baseline_lambda
            )
            y = y - baseline_values
            applied.append(f"baseline({self.baseline_method})")
        if self.normalization == "minmax":
            y = minmax(y)
            applied.append("minmax")
        elif self.normalization == "snv":
            y = standard_normal_variate(y)
            applied.append("snv")
        return spectrum.copy_with(
            y=y,
            preprocessing=" -> ".join(applied) if applied else "none",
            baseline_method=self.baseline_method if self.baseline else "none",
            baseline=baseline_values,
        )

    def transform(self, spectra: list[Spectrum]) -> list[Spectrum]:
        return [self.transform_one(s) for s in spectra]
