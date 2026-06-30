from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProbabilityConformalPredictor:
    alpha: float = 0.1
    threshold: float | None = None

    def fit(
        self, true_labels: list[str], probabilities: list[dict[str, float]]
    ) -> ProbabilityConformalPredictor:
        scores = [
            1.0 - probs[label] for label, probs in zip(true_labels, probabilities, strict=True)
        ]
        scores = sorted(scores)
        index = min(len(scores) - 1, int((1 - self.alpha) * (len(scores) + 1)) - 1)
        self.threshold = scores[max(0, index)]
        return self

    def predict_set(self, probabilities: dict[str, float]) -> list[str]:
        if self.threshold is None:
            raise RuntimeError("Conformal predictor must be fitted before use")
        return [label for label, prob in probabilities.items() if 1.0 - prob <= self.threshold]
