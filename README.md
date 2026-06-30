# NarcoticSense AI

**Open-source AI spectroscopy operating system for narcotic sensing research and analytical chemistry.**

NarcoticSense AI is a chemist-friendly platform for importing, visualizing, preprocessing, analyzing, and reporting spectroscopy data. It is designed for spectroscopy researchers first, with AI/ML components added in a transparent and reproducible way.

> Responsible use: this software is for research, screening, visualization, and decision support only. High-stakes findings must be confirmed using validated laboratory methods.

## Quick start on Mac

1. Download or clone the repository.
2. Open the project folder.
3. Double-click `START_HERE.command`.
4. Upload CSV/TXT/TSV spectroscopy files.

Manual method:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m streamlit run app/streamlit_app.py
```

## Quick start from GitHub

```bash
git clone https://github.com/drjoykarmakar/NarcoticSense-AI.git
cd NarcoticSense-AI
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m streamlit run app/streamlit_app.py
```

For development:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```


New in v0.2.0:
- Dataset quality checks for uploaded spectra.
- Optional metadata CSV upload with sample labels/classes.
- Spectral library matching by similarity.
- PLS-DA exploratory projections when labels are available.
- Cross-validated supervised ML baseline using Random Forest or SVM.
- Extra unit tests for quality control, library matching, and ML evaluation.

## Current features

- Universal spectroscopy importer for common CSV/TXT/TSV vendor-like exports.
- Auto-detection of columns such as `Wavelength (nm)`, `Raman Shift`, `Abs`, `Intensity`, and `Counts`.
- Automatic handling of title/header rows, empty columns, and descending x-axes.
- Multi-file upload.
- Dataset manager and metadata template export.
- Raw, processed, overlay, and derivative spectra viewer.
- Baseline correction, smoothing, and normalization.
- Peak table and spectral metrics.
- PCA, t-SNE, and K-means chemometrics.
- Aligned spectral matrix export for future ML.
- AI dataset planning tab.
- Markdown research report generator.
- FastAPI starter backend.
- Tests, Docker files, and GitHub Actions CI.

## Example spectrum format

Simple CSV:

```csv
x,y
200,0.12
201,0.15
202,0.19
```

Vendor-like files are also supported, including files with a title row followed by columns such as:

```csv
Wavelength (nm),Abs
800.0,-0.003
799.0,-0.002
```

## Project structure

```text
app/                    Streamlit app
src/narcoticsense/      Python package
tests/                  Unit tests
docs/                   User and developer docs
examples/               Example spectra
configs/                Configuration files
docker/                 Docker components
.github/                GitHub Actions and templates
scripts/                Helper scripts
```

## Scientific modules

- `spectroscopy`: spectrum objects, importers, alignment, datasets.
- `preprocessing`: smoothing, baseline correction, normalization.
- `feature_engineering`: peaks, derivatives, spectral metrics.
- `chemometrics`: PCA, t-SNE, clustering.
- `visualization`: Plotly scientific plots.
- `reports`: Markdown research reports.
- `classical_ml`: starter classifier wrapper.
- `uncertainty`: conformal prediction starter.
- `explainability`: spectral attribution starter.
- `api`: FastAPI starter service.

## Test status

Run locally:

```bash
python3 -m ruff check src tests app
python3 -m black --check src tests app
python3 -m pytest -q
```

## License

Code is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
