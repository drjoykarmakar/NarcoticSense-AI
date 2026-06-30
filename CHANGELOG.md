
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

## v0.3.0

- Added spectroscopy engine upgrades: AsLS, airPLS, and arPLS baseline options.
- Added FWHM peak widths in x-axis units.
- Added FWHM-window peak integration.
- Added batch export for processed spectra in long CSV format.
- Added interactive HTML export for Plotly spectral figures.
- Added preprocessing history metadata and additional tests.

# Changelog

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
