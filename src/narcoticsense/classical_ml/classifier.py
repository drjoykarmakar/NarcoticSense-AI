from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class SklearnClassifier:
    model: Any | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
            ])

    def fit(self, features: list[dict[str, float]], labels: list[str]) -> "SklearnClassifier":
        self.model.fit(pd.DataFrame(features).fillna(0.0), labels)
        return self

    def predict(self, features: list[dict[str, float]]) -> list[str]:
        return list(self.model.predict(pd.DataFrame(features).fillna(0.0)))

    def predict_proba(self, features: list[dict[str, float]]) -> list[dict[str, float]]:
        frame = pd.DataFrame(features).fillna(0.0)
        probabilities = self.model.predict_proba(frame)
        classes = list(self.model.classes_)
        return [dict(zip(classes, row, strict=False)) for row in probabilities]

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "SklearnClassifier":
        return cls(model=joblib.load(path))
