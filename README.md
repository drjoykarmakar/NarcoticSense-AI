# NarcoticSense AI

<p align="center">

**An Open-Source AI Platform for Spectroscopic Narcotic Sensing, Chemometrics, and Analytical Chemistry**

*Designed for researchers, forensic scientists, analytical chemists, and spectroscopy developers.*

</p>

---

## Overview

**NarcoticSense AI** is an open-source, chemist-friendly platform for importing, preprocessing, visualizing, analyzing, and interpreting spectroscopic data using modern chemometrics and artificial intelligence.

The project aims to provide a reproducible research platform for spectroscopy-based narcotic sensing while remaining extensible to broader analytical chemistry applications including fluorescence, Raman, UV–Visible, FTIR, and future multimodal sensing technologies.

Rather than being a single machine-learning model, NarcoticSense AI is designed as a **scientific operating system** for spectroscopy research.

---

## Current Capabilities

### Spectroscopy

- Universal spectroscopy importer
- Automatic vendor-style CSV parsing
- Automatic wavelength/intensity detection
- Multi-file datasets
- Batch processing
- Spectral overlays
- Interactive visualization
- Baseline correction
- Smoothing
- Normalization
- Derivative spectra
- Peak detection
- Peak FWHM
- Peak integration
- Spectral metrics

---

### Chemometrics

- PCA
- PLS Regression
- PLS-DA
- LDA
- SIMCA-style class modeling
- UMAP
- t-SNE
- Hierarchical clustering
- K-Means clustering
- Spectral similarity matrices

---

### Artificial Intelligence

- Random Forest
- Support Vector Machine
- Logistic Regression
- Gradient Boosting
- Extra Trees
- K-Nearest Neighbors
- Ridge Regression
- PLS Regression
- Random Forest Regression
- Support Vector Regression

---

### Trustworthy AI

- Confidence estimation
- Conformal prediction
- Unknown / Out-of-distribution detection
- Cross-validation
- Model comparison
- Peak-level attribution
- Spectral-region explainability
- Decision-support flags for confirmatory testing

---

### Data Management

- Dataset manager
- Metadata templates
- Multi-file upload
- Spectral library matching
- Processed spectrum export
- Markdown report generation

---

### Software Engineering

- Streamlit interface
- FastAPI backend
- Docker support
- GitHub Actions
- Unit testing
- Apache License 2.0
- Reproducible environments

---

# Research Vision

The long-term goal is to build an open scientific platform integrating:

- Fluorescence spectroscopy
- Raman spectroscopy
- UV–Visible spectroscopy
- FTIR spectroscopy
- Hyperspectral imaging
- Chemometrics
- Explainable AI
- Uncertainty quantification
- Active learning
- Molecular descriptors
- RDKit integration
- Large Language Models
- Scientific report generation

---

# Installation

Clone the repository

```bash
git clone https://github.com/drjoykarmakar/NarcoticSense-AI.git
cd NarcoticSense-AI
```

Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

Run

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m streamlit run app/streamlit_app.py
```

---

# Development

Install development environment

```bash
python3 -m pip install -e ".[dev]"
```

Run tests

```bash
python3 -m pytest
```

Run linting

```bash
python3 -m ruff check src tests app
python3 -m black --check src tests app
```

---

# Supported Data Formats

- CSV
- TXT
- TSV
- Vendor exports
- Wavelength–Intensity tables
- Raman Shift–Intensity tables

Future releases will support

- JCAMP-DX
- Excel
- Ocean Insight
- Horiba
- Agilent
- Thermo Fisher
- Shimadzu
- Bruker

---

# Project Structure

```
app/
configs/
docker/
docs/
examples/
scripts/
src/
tests/
```

Core package

```
src/narcoticsense/

    spectroscopy/
    preprocessing/
    feature_engineering/
    chemometrics/
    models/
    uncertainty/
    explainability/
    molecular/
    fusion/
    reports/
    visualization/
    api/
```

---

# Roadmap

### v0.8

- Deep Learning
- Spectral CNN
- Spectral Transformers
- Contrastive Learning
- Foundation Models

### v0.9

- LLM Scientific Assistant
- Experiment Interpretation
- Literature Summarization
- Automated Methods Generation

### v1.0

- Research Edition
- Stable API
- Complete Documentation
- Benchmark Datasets
- Cloud Deployment
- Zenodo DOI
- SoftwareX / JOSS submission

---

# Responsible Use

NarcoticSense AI is intended for:

- Scientific research
- Education
- Method development
- Decision support

It is **not** a certified forensic, clinical, or diagnostic system.

All high-stakes decisions should be confirmed using validated laboratory procedures and appropriate quality assurance protocols.

---

# Author

## Dr. Joy Karmakar, Ph.D.

Founder & Principal Developer — **NarcoticSense AI**

Founder — **DyeMind**

🌐 https://www.dyemind.com

GitHub

https://github.com/drjoykarmakar

ORCID

https://orcid.org/0000-0002-8232-5639

### Research Interests

- Artificial Intelligence for Spectroscopy
- Fluorescence Spectroscopy
- Raman Spectroscopy
- Chemometrics
- Analytical Chemistry
- Molecular Sensing
- Intelligent Sensor Development

---

# Citation

If you use NarcoticSense AI in your research, please cite the software using the metadata in:

```
CITATION.cff
```

---

# License

Apache License 2.0

See

- LICENSE
- NOTICE
- DATA_LICENSE.md

---

# Acknowledgements

NarcoticSense AI is an open research initiative developed to accelerate reproducible spectroscopy research and foster collaboration across analytical chemistry, machine learning, and sensor science.

Contributions, feature requests, and collaborations are welcome.
