from __future__ import annotations

from fastapi import FastAPI

from narcoticsense.api.schemas import PredictionOut, SpectrumIn
from narcoticsense.spectroscopy import Spectrum

app = FastAPI(
    title="NarcoticSense AI API",
    version="0.1.0",
    description="Research API for spectroscopy-driven narcotic sensing decision support.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/models/predict", response_model=PredictionOut)
def predict(payload: SpectrumIn) -> PredictionOut:
    # Placeholder until a validated model artifact is registered.
    Spectrum(
        x=payload.x,
        y=payload.y,
        modality=payload.modality,
        sample_id=payload.sample_id,
        metadata=payload.metadata,
    )
    return PredictionOut(
        label="unvalidated_model_placeholder",
        confidence=0.0,
        probabilities={},
        warning="No validated model is loaded. Train/register a model before scientific use.",
    )
