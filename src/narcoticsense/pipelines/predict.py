from __future__ import annotations

from dataclasses import dataclass

from narcoticsense.classical_ml import SklearnClassifier
from narcoticsense.explainability import peak_level_attribution
from narcoticsense.feature_engineering import PeakFeatureExtractor
from narcoticsense.preprocessing import PreprocessingPipeline
from narcoticsense.spectroscopy import Spectrum


@dataclass(slots=True)
class PredictionPipeline:
    preprocessing: PreprocessingPipeline
    features: PeakFeatureExtractor
    model: SklearnClassifier

    def predict_one(self, spectrum: Spectrum) -> dict:
        processed = self.preprocessing.transform_one(spectrum)
        feats = self.features.extract_one(processed)
        probabilities = self.model.predict_proba([feats])[0]
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "attribution": peak_level_attribution(processed),
            "warning": "Decision-support only; confirm low-confidence or high-stakes results with validated laboratory methods.",
        }
