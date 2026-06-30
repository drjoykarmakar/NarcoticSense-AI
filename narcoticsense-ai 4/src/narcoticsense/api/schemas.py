from __future__ import annotations

from pydantic import BaseModel, Field


class SpectrumIn(BaseModel):
    x: list[float]
    y: list[float]
    modality: str = Field(default="unknown")
    sample_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class PredictionOut(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    warning: str
