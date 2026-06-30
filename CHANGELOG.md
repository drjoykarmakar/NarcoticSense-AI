# Changelog

## v0.8.0 - Real AI Training and Prediction Workflow

- Added controlled real-AI workflow for training models only from approved labeled spectra.
- Added training table generation, readiness checks, and explicit unknown/unlabeled handling.
- Added model registry for saved `.joblib` model bundles.
- Added unknown-sample prediction workflow with confidence, probability margin, OOD distance, and confirmatory-testing flags.
- Added reusable `narcoticsense.models.workflow` utilities for controlled training and inference.
- Added tests for training table validation, model saving, registry listing, and unknown prediction.

## v0.7.0 - Molecular and Multimodal AI

- Added Molecular + Multimodal AI tab.
- Added optional RDKit-aware SMILES descriptors and Morgan fingerprints.
- Added deterministic fallback molecular features so the app remains usable without RDKit.
- Added molecular metadata template support for `sample_id` and `smiles`.
- Added early-fusion spectral + molecular ML matrix export.
- Added fusion block summaries for spectral, molecular, and fused features.
- Added `narcoticsense.molecular` and `narcoticsense.fusion` modules.
- Added tests for molecular descriptors, fingerprints, sample alignment, and early fusion.

## v0.6.0 - Explainability, Uncertainty, and Robust Validation

- Added Trustworthy AI tab.
- Added cross-validated classifier comparison.
- Added confidence and probability-margin tables.
- Added conformal classification thresholds and prediction sets.
- Added distance-based OOD/unknown-sample detection.
- Added automatic refer-to-confirmatory-testing flags.
- Added spectral occlusion explainability.
- Added peak-level attribution mapping model explanations to detected peaks.
- Added tests for uncertainty, explainability, and robust validation.


## v0.5.0 - Chemometrics and Validation Expansion

- Added PLS regression for concentration/response modeling.
- Added LDA score projections.
- Added SIMCA-style one-class PCA class modeling.
- Added hierarchical clustering linkage export.
- Added optional UMAP with PCA fallback when umap-learn is not installed.
- Added spectral distance matrix export.
- Added probability outputs for Random Forest and SVM baselines.
- Added reliability/calibration, ROC, and precision-recall validation tables.
- Expanded unit tests to cover v0.5.0 chemometrics and validation behavior.

# Changelog

## v0.8.0 - Real AI Training and Prediction Workflow

- Added controlled real-AI workflow for training models only from approved labeled spectra.
- Added training table generation, readiness checks, and explicit unknown/unlabeled handling.
- Added model registry for saved `.joblib` model bundles.
- Added unknown-sample prediction workflow with confidence, probability margin, OOD distance, and confirmatory-testing flags.
- Added reusable `narcoticsense.models.workflow` utilities for controlled training and inference.
- Added tests for training table validation, model saving, registry listing, and unknown prediction.

## v0.3.0

- Added spectroscopy engine upgrades: AsLS, airPLS, and arPLS baseline options.
- Added FWHM peak widths in x-axis units.
- Added FWHM-window peak integration.
- Added batch export for processed spectra in long CSV format.
- Added interactive HTML export for Plotly spectral figures.
- Added preprocessing history metadata and additional tests.

# Changelog

## v0.8.0 - Real AI Training and Prediction Workflow

- Added controlled real-AI workflow for training models only from approved labeled spectra.
- Added training table generation, readiness checks, and explicit unknown/unlabeled handling.
- Added model registry for saved `.joblib` model bundles.
- Added unknown-sample prediction workflow with confidence, probability margin, OOD distance, and confirmatory-testing flags.
- Added reusable `narcoticsense.models.workflow` utilities for controlled training and inference.
- Added tests for training table validation, model saving, registry listing, and unknown prediction.

## v0.2.0 - Chemometrics and ML foundation

Added dataset quality checks, metadata upload support, spectral library matching, PLS-DA exploratory projections, and cross-validated supervised ML baselines. Expanded tests to 11 passing tests.


## 0.1.0

Initial public research scaffold.

### Added

- Chemist-friendly Streamlit interface.
- Universal CSV importer for common spectroscopy exports.
- Multi-spectrum dataset support.
- Spectral preprocessing, peak analysis, overlays, PCA, t-SNE, K-means.
- Baseline ML and uncertainty scaffolds.
- FastAPI placeholder service.
- Tests and GitHub Actions CI.
- Apache License 2.0.
