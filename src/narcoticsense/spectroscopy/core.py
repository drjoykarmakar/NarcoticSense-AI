from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Spectrum:
    """Container for a single spectrum and its scientific metadata."""

    x: Iterable[float]
    y: Iterable[float]
    modality: str
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        if self.x.ndim != 1:
            raise ValueError("Spectrum arrays must be one-dimensional")
        if len(self.x) < 2:
            raise ValueError("A spectrum requires at least two points")

    @property
    def n_points(self) -> int:
        return int(len(self.x))

    def copy_with(
        self, *, x: np.ndarray | None = None, y: np.ndarray | None = None, **metadata: Any
    ) -> Spectrum:
        merged = dict(self.metadata)
        merged.update(metadata)
        return Spectrum(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            modality=self.modality,
            sample_id=self.sample_id,
            metadata=merged,
        )


@dataclass(slots=True)
class SpectralDataset:
    spectra: list[Spectrum]
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        if self.labels is not None and len(self.labels) != len(self.spectra):
            raise ValueError("labels must match spectra length")
