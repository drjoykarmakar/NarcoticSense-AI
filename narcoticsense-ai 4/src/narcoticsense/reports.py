from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from narcoticsense.spectroscopy import Spectrum


def make_markdown_report(
    *,
    project_name: str,
    spectrum: Spectrum,
    preprocessing: dict,
    peaks: pd.DataFrame,
    notes: str = "",
) -> str:
    top_peaks = "No peaks detected."
    if not peaks.empty:
        top_peaks = peaks.head(10).to_markdown(index=False)
    return f"""# NarcoticSense AI Experiment Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Project

{project_name}

## Sample

- Sample ID: {spectrum.sample_id or "not provided"}
- Modality: {spectrum.modality}
- Points: {spectrum.n_points}

## Preprocessing

```yaml
{preprocessing}
```

## Top Detected Peaks

{top_peaks}

## Scientist Notes

{notes or "No notes provided."}

## Responsible Use

This report is for research and decision support only. High-stakes results require validated confirmatory laboratory methods, calibrated instruments, documented chain of custody when applicable, and domain-expert review.
"""
