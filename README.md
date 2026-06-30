# NarcoticSense AI

**Open-source AI spectroscopy operating system for narcotic sensing research and analytical chemistry.**

NarcoticSense AI is a chemist-friendly platform for importing, visualizing, preprocessing, analyzing, and reporting spectroscopy data. It is designed for spectroscopy researchers first, with AI/ML components added in a transparent and reproducible way.

> Responsible use: this software is for research, screening, visualization, and decision support only. High-stakes findings must be confirmed using validated laboratory methods.

> **Created and maintained by Dr. Joy Karmakar (DyeMind).**
>
> An open-source research platform for spectroscopy, chemometrics, and trustworthy AI.

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



New in v0.8.0:

- Real AI Training + Prediction Workflow tab.
- Controlled training set builder using uploaded metadata and approved labels.
- Clear separation between training mode and unknown prediction mode.
- Saved model registry in the `models/` directory.
- Unknown-sample prediction workflow using saved models.
- Confidence threshold, probability margin, OOD distance, and confirmatory-testing decision flags.
- Training readiness checks for class count, labeled spectra, and replicates.
- Validation metrics, holdout predictions, and confusion matrix before saving models.
- Explicit warning that uploads do not automatically retrain the AI.

New in v0.7.0:

- Molecular + Multimodal AI tab.
- Optional RDKit-aware SMILES descriptor and Morgan fingerprint workflow.
- Safe fallback molecular features when RDKit is not installed.
- Metadata templates for `sample_id` + `smiles`.
- Spectral + molecular early-fusion matrix export.
- Block summaries for spectral, molecular, and fused feature spaces.
- New `narcoticsense.molecular` and `narcoticsense.fusion` modules.
- Expanded tests for molecular descriptors and multimodal fusion alignment.

New in v0.6.0:

- Trustworthy AI tab for confidence, uncertainty, and unknown detection.
- Cross-validated model comparison across available classifiers.
- Confidence tables with probability margins.
- Conformal prediction sets for uncertainty-aware classification.
- Distance-based out-of-distribution / unknown-sample flags.
- Automatic "refer to confirmatory testing" decision-support flag.
- Spectral occlusion explainability for influential spectral regions.
- Peak-level attribution connecting model explanations to detected peaks.
- Expanded tests for uncertainty, explainability, OOD detection, and validation.

New in v0.5.0:

- AI Model Engine tab.
- Classification models: Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, SVM, and KNN.
- Regression models: PLS Regression, Random Forest Regressor, Ridge Regression, and SVR.
- Train/test split controls.
- Classification metrics, regression metrics, predictions, probabilities, and confusion matrix outputs.
- Feature importance for supported models.
- Save trained models to the `models/` directory.
- New `narcoticsense.models` package.
- Expanded tests for model training, evaluation, and serialization.

New in v0.3.0:

- Professional spectroscopy engine upgrades.
- Baseline method selector: AsLS, airPLS, and arPLS.
- Peak FWHM in spectral x-axis units.
- Peak integration over FWHM windows.
- Batch export of all processed spectra in long CSV format.
- Interactive Plotly HTML figure export for overlays and derivatives.
- Preprocessing history metadata for reproducibility.

New in v0.2.0:
- Dataset quality checks for uploaded spectra.
- Optional metadata CSV upload with sample labels/classes.
- Spectral library matching by similarity.
- PLS-DA, LDA, SIMCA-style class modeling, and PLS regression when metadata are available.
- Cross-validated supervised ML baseline using Random Forest or SVM with probability, calibration, ROC, and PR validation exports.
- Extra unit tests for quality control, library matching, and ML evaluation.

## Current features

- Universal spectroscopy importer for common CSV/TXT/TSV vendor-like exports.
- Auto-detection of columns such as `Wavelength (nm)`, `Raman Shift`, `Abs`, `Intensity`, and `Counts`.
- Automatic handling of title/header rows, empty columns, and descending x-axes.
- Multi-file upload.
- Dataset manager and metadata template export.
- Raw, processed, overlay, and derivative spectra viewer.
- Baseline correction using AsLS/airPLS/arPLS, smoothing, and normalization.
- Peak table, FWHM, peak integration, and spectral metrics.
- PCA, t-SNE, and K-means chemometrics.
- Aligned spectral matrix export for future ML.
- AI dataset planning tab.
- Trustworthy AI: confidence, conformal prediction sets, OOD flags, model comparison, and spectral-region explanations.
- Real AI workflow: controlled training, saved model registry, unknown prediction mode, and confirmatory-testing flags.
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
---

# Author

## Dr. Joy Karmakar ##

**Founder & Principal Developer — NarcoticSense AI**

Founder — **DyeMind**

🌐 Website  
https://www.dyemind.com

🆔 ORCID  
https://orcid.org/0000-0002-8232-5639

💻 GitHub  
https://github.com/drjoykarmakar

---

## Research Interests

- Artificial Intelligence for Spectroscopy
- Fluorescence Spectroscopy
- Raman Spectroscopy
- Chemometrics
- Analytical Chemistry
- Molecular Sensing
- Intelligent Sensor Development
- Scientific Machine Learning
- Explainable Artificial Intelligence
- Multimodal Spectroscopic Analysis

---

# Citation

If you use **NarcoticSense AI** in research, please cite the software using the metadata provided in:

```
CITATION.cff
```

or cite the GitHub release DOI (when available through Zenodo).

---

# License

This project is distributed under the **Apache License 2.0**.

See:

- LICENSE
- NOTICE
- DATA_LICENSE.md

---

# Collaboration

Collaborations are welcome from researchers working in:

- Spectroscopy
- Analytical Chemistry
- Chemometrics
- Machine Learning
- Sensor Development
- Biomedical Engineering
- Pharmaceutical Sciences
- Forensic Science

For collaboration opportunities, please contact us through GitHub or via **https://www.dyemind.com**.

---

# Acknowledgements

NarcoticSense AI is an independent open-source research initiative founded and developed by **Dr. Joy Karmakar** through **DyeMind**.

The long-term vision is to build a transparent, reproducible, and extensible AI platform for spectroscopy that supports researchers worldwide in analytical chemistry, sensor science, and molecular detection.
## License

Code is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
