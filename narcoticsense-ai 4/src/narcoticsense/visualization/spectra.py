from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from narcoticsense.spectroscopy import Spectrum


def axis_label(spectrum: Spectrum) -> str:
    if spectrum.modality == "raman":
        return "Raman shift / cm⁻¹"
    if spectrum.modality in {"fluorescence", "uv-vis"}:
        return "Wavelength / nm"
    if spectrum.modality == "infrared":
        return "Wavenumber / cm⁻¹"
    return "Spectral axis"


def y_label(spectrum: Spectrum) -> str:
    if spectrum.modality == "uv-vis":
        return "Absorbance / a.u."
    if spectrum.modality == "fluorescence":
        return "Intensity / a.u."
    return "Signal / a.u."


def plot_spectrum(spectrum: Spectrum, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spectrum.x, y=spectrum.y, mode="lines", name=spectrum.sample_id or spectrum.modality))
    fig.update_layout(
        title=title or f"{spectrum.modality.title()} spectrum",
        xaxis_title=axis_label(spectrum),
        yaxis_title=y_label(spectrum),
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_overlay(spectra: list[Spectrum], title: str = "Spectral overlay") -> go.Figure:
    fig = go.Figure()
    for spectrum in spectra:
        name = spectrum.sample_id or spectrum.modality
        fig.add_trace(go.Scatter(x=spectrum.x, y=spectrum.y, mode="lines", name=name))
    first = spectra[0] if spectra else None
    fig.update_layout(
        title=title,
        xaxis_title=axis_label(first) if first else "Axis",
        yaxis_title=y_label(first) if first else "Signal",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_peaks(spectrum: Spectrum, peaks: pd.DataFrame, title: str = "Peak annotations") -> go.Figure:
    fig = plot_spectrum(spectrum, title=title)
    if not peaks.empty:
        fig.add_trace(go.Scatter(
            x=peaks["position"],
            y=peaks["height"],
            mode="markers+text",
            text=peaks["rank"].astype(str),
            textposition="top center",
            name="Detected peaks",
        ))
    return fig


def plot_projection(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    return px.scatter(df, x=x_col, y=y_col, text="sample_id", title=title, template="plotly_white")
