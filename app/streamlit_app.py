from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

try:
    from narcoticsense.chemometrics import run_kmeans, run_pca, run_tsne
    from narcoticsense.feature_engineering import derivative_spectrum, peak_table, spectral_metrics
    from narcoticsense.preprocessing import PreprocessingPipeline
    from narcoticsense.reports import make_markdown_report
    from narcoticsense.spectroscopy import align_spectra, dataset_summary, import_spectrum
    from narcoticsense.visualization import plot_overlay, plot_peaks, plot_projection, plot_spectrum
except Exception as exc:
    st.set_page_config(page_title="NarcoticSense AI setup error", page_icon="🔬")
    st.error("NarcoticSense AI could not import its local Python package.")
    st.write("Please run the app from the project folder, not from an old copied app file.")
    st.code(f"Project root: {PROJECT_ROOT}\nSource folder: {SRC_DIR}\nPython error: {exc}")
    st.stop()

st.set_page_config(page_title="NarcoticSense AI", page_icon="🔬", layout="wide")

st.title("NarcoticSense AI")
st.caption("Chemist-friendly AI spectroscopy operating system for research, sensing, and decision support")
st.warning("Research and decision-support only. Confirm high-stakes results with validated laboratory methods.")

if "spectra" not in st.session_state:
    st.session_state.spectra = []
if "processed" not in st.session_state:
    st.session_state.processed = []
if "import_results" not in st.session_state:
    st.session_state.import_results = []

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", value="My Spectroscopy Project")
    modality = st.selectbox("Modality", ["fluorescence", "raman", "infrared", "uv-vis", "hyperspectral", "unknown"], index=0)
    default_label = st.text_input("Default sample label", value="sample")
    st.divider()
    st.header("Preprocessing")
    smooth = st.checkbox("Savitzky-Golay smoothing", value=True)
    baseline = st.checkbox("Baseline correction", value=True)
    normalization = st.selectbox("Normalization", ["minmax", "snv", "none"], index=0)
    window_length = st.slider("Smoothing window", 5, 51, 11, step=2)
    polyorder = st.slider("Polynomial order", 2, 5, 3)
    prominence = st.slider("Peak prominence", 0.001, 1.0, 0.05, step=0.001, format="%.3f")
    st.divider()
    if st.button("Clear uploaded spectra"):
        st.session_state.spectra = []
        st.session_state.processed = []
        st.session_state.import_results = []
        st.rerun()

pipe = PreprocessingPipeline(smooth=smooth, baseline=baseline, normalization=normalization, window_length=window_length, polyorder=polyorder)

tabs = st.tabs([
    "1 Import",
    "2 Dataset",
    "3 Spectra Viewer",
    "4 Peak Analysis",
    "5 Chemometrics",
    "6 AI Training Plan",
    "7 Report",
    "Chemist Guide",
])

def _reprocess_all() -> None:
    st.session_state.processed = [pipe.transform_one(s) for s in st.session_state.spectra]

with tabs[0]:
    st.subheader("Universal spectroscopy import")
    st.write("Upload one or many CSV/TXT/TSV files. NarcoticSense AI will detect columns such as `Wavelength (nm)`, `Raman Shift`, `Abs`, `Intensity`, or `Counts`.")
    uploaded_files = st.file_uploader("Upload spectrum files", type=["csv", "txt", "tsv"], accept_multiple_files=True)
    example = pd.DataFrame({"Wavelength (nm)": [200, 201, 202, 203, 204], "Abs": [0.12, 0.15, 0.19, 0.16, 0.22]})
    st.download_button("Download example CSV", example.to_csv(index=False), file_name="example_spectrum.csv", mime="text/csv")

    if uploaded_files:
        spectra = []
        processed = []
        results = []
        for idx, uploaded in enumerate(uploaded_files, start=1):
            sample_id = Path(uploaded.name).stem or f"{default_label}-{idx:03d}"
            try:
                result = import_spectrum(uploaded.getvalue(), modality=modality, sample_id=sample_id)
            except Exception as exc:
                st.error(f"Could not import {uploaded.name}")
                st.code(str(exc))
                continue
            spectra.append(result.spectrum)
            processed.append(pipe.transform_one(result.spectrum))
            results.append(result)
        if spectra:
            st.session_state.spectra = spectra
            st.session_state.processed = processed
            st.session_state.import_results = results
            st.success(f"Imported {len(spectra)} spectrum/spectra.")

    if st.session_state.import_results:
        rows = []
        for result in st.session_state.import_results:
            rows.append({
                "sample_id": result.spectrum.sample_id,
                "points": result.spectrum.n_points,
                "x column": result.x_column,
                "y column": result.y_column,
                "confidence": f"{result.confidence:.0%}",
                "notes": "; ".join(result.messages),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("Preview first imported file"):
            st.dataframe(st.session_state.import_results[0].dataframe.head(30), use_container_width=True)

with tabs[1]:
    st.subheader("Dataset manager")
    if not st.session_state.spectra:
        st.info("Import spectra first.")
    else:
        summary = dataset_summary(st.session_state.spectra)
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download dataset summary", summary.to_csv(index=False), file_name="dataset_summary.csv", mime="text/csv")
        st.markdown("### Metadata template for your lab notebook")
        meta_template = pd.DataFrame({
            "sample_id": [s.sample_id for s in st.session_state.spectra],
            "compound_or_class": ["" for _ in st.session_state.spectra],
            "concentration": ["" for _ in st.session_state.spectra],
            "solvent": ["" for _ in st.session_state.spectra],
            "pH": ["" for _ in st.session_state.spectra],
            "temperature_C": ["" for _ in st.session_state.spectra],
            "instrument": ["" for _ in st.session_state.spectra],
            "operator": ["" for _ in st.session_state.spectra],
            "notes": ["" for _ in st.session_state.spectra],
        })
        st.dataframe(meta_template, use_container_width=True)
        st.download_button("Download metadata template", meta_template.to_csv(index=False), file_name="metadata_template.csv", mime="text/csv")

with tabs[2]:
    st.subheader("Professional spectra viewer")
    if not st.session_state.spectra:
        st.info("Import spectra first.")
    else:
        view_choice = st.radio("View", ["Processed", "Raw", "Raw vs processed", "Derivative"], horizontal=True)
        spectra_to_plot = st.session_state.processed if view_choice == "Processed" else st.session_state.spectra
        if view_choice == "Raw vs processed":
            selected = st.selectbox("Select spectrum", [s.sample_id for s in st.session_state.spectra])
            idx = [s.sample_id for s in st.session_state.spectra].index(selected)
            st.plotly_chart(plot_overlay([st.session_state.spectra[idx], st.session_state.processed[idx]], title=f"Raw vs processed: {selected}"), use_container_width=True)
        elif view_choice == "Derivative":
            order = st.radio("Derivative order", [1, 2], horizontal=True)
            deriv = [derivative_spectrum(s, order=order) for s in st.session_state.processed]
            st.plotly_chart(plot_overlay(deriv, title=f"{order} derivative overlay"), use_container_width=True)
        else:
            st.plotly_chart(plot_overlay(spectra_to_plot, title=f"{view_choice} spectral overlay"), use_container_width=True)
        if st.session_state.processed:
            selected = st.selectbox("Single-spectrum detail", [s.sample_id for s in st.session_state.processed], key="detail_select")
            idx = [s.sample_id for s in st.session_state.processed].index(selected)
            st.plotly_chart(plot_spectrum(st.session_state.processed[idx], title=f"Processed detail: {selected}"), use_container_width=True)
            out = pd.DataFrame({"x": st.session_state.processed[idx].x, "y": st.session_state.processed[idx].y})
            st.download_button("Download selected processed CSV", out.to_csv(index=False), file_name=f"{selected}_processed.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Peak analysis and spectral metrics")
    if not st.session_state.processed:
        st.info("Import spectra first.")
    else:
        selected = st.selectbox("Select spectrum for peak analysis", [s.sample_id for s in st.session_state.processed], key="peak_select")
        idx = [s.sample_id for s in st.session_state.processed].index(selected)
        spectrum = st.session_state.processed[idx]
        peaks = peak_table(spectrum, prominence=prominence, max_peaks=100)
        metrics = spectral_metrics(spectrum)
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected peaks", len(peaks))
        c2.metric("Max position", f"{metrics.loc[metrics.metric == 'maximum position', 'value'].iloc[0]:.3g}")
        c3.metric("Absolute area", f"{metrics.loc[metrics.metric == 'absolute area', 'value'].iloc[0]:.3g}")
        st.plotly_chart(plot_peaks(spectrum, peaks), use_container_width=True)
        left, right = st.columns(2)
        left.dataframe(peaks, use_container_width=True)
        right.dataframe(metrics, use_container_width=True)
        st.download_button("Download peak table", peaks.to_csv(index=False), file_name=f"{selected}_peaks.csv", mime="text/csv")
        st.download_button("Download metrics", metrics.to_csv(index=False), file_name=f"{selected}_metrics.csv", mime="text/csv")

with tabs[4]:
    st.subheader("Chemometrics")
    if len(st.session_state.processed) < 2:
        st.info("Import at least two spectra for PCA. Four or more are recommended for t-SNE.")
    else:
        n_points = st.slider("Common grid points", 100, 3000, 1000, step=100)
        aligned = align_spectra(st.session_state.processed, n_points=n_points)
        st.write(f"Aligned dataset: {aligned.n_samples} spectra × {aligned.n_features} spectral variables.")
        pca = run_pca(aligned.matrix, aligned.sample_ids, aligned.x, n_components=3)
        ev = [round(v * 100, 2) for v in pca.explained_variance]
        st.write(f"PCA explained variance: {ev}%")
        st.plotly_chart(plot_projection(pca.coordinates, "PC1", "PC2", "PCA score plot"), use_container_width=True)
        st.dataframe(pca.coordinates, use_container_width=True)
        if pca.loadings is not None:
            st.line_chart(pca.loadings.set_index("x"))
            st.download_button("Download PCA loadings", pca.loadings.to_csv(index=False), file_name="pca_loadings.csv", mime="text/csv")
        if aligned.n_samples >= 4:
            try:
                tsne_df = run_tsne(aligned.matrix, aligned.sample_ids)
                st.plotly_chart(plot_projection(tsne_df, "TSNE1", "TSNE2", "t-SNE exploratory map"), use_container_width=True)
            except Exception as exc:
                st.info(f"t-SNE skipped: {exc}")
        if aligned.n_samples >= 2:
            k = st.slider("K-means clusters", 2, min(8, aligned.n_samples), 2)
            try:
                clusters = run_kmeans(aligned.matrix, aligned.sample_ids, n_clusters=k)
                st.dataframe(clusters, use_container_width=True)
            except Exception as exc:
                st.info(f"Clustering skipped: {exc}")
        st.download_button("Download aligned ML matrix", aligned.to_dataframe().to_csv(index=False), file_name="aligned_spectral_matrix.csv", mime="text/csv")

with tabs[5]:
    st.subheader("No-code AI training plan")
    st.info("This tab tells you exactly what to collect before supervised AI is scientifically meaningful.")
    n_classes = st.number_input("How many chemical classes/conditions?", min_value=2, max_value=100, value=5)
    reps = st.number_input("Replicates per class", min_value=3, max_value=1000, value=20)
    blanks = st.number_input("Blank/control spectra", min_value=3, max_value=1000, value=20)
    st.metric("Minimum first dataset", int(n_classes * reps + blanks))
    st.markdown("""
Recommended order:

1. blank/control spectra
2. pure standards
3. concentration series
4. mixtures and adulterants
5. different days/instruments/operators
6. blind validation set never used for model training

Start with PCA/PLS/random forest/SVM. Use CNNs or transformers only after the dataset is large and independently validated.
""")

with tabs[6]:
    st.subheader("Research report")
    notes = st.text_area("Scientist notes", placeholder="Describe sample prep, instrument settings, observations, and concerns.")
    if not st.session_state.processed:
        st.info("Import spectra first.")
    else:
        selected = st.selectbox("Report spectrum", [s.sample_id for s in st.session_state.processed], key="report_select")
        idx = [s.sample_id for s in st.session_state.processed].index(selected)
        spectrum = st.session_state.processed[idx]
        peaks = peak_table(spectrum, prominence=prominence)
        preprocessing_config = {
            "smooth": smooth,
            "baseline": baseline,
            "normalization": normalization,
            "window_length": window_length,
            "polyorder": polyorder,
            "peak_prominence": prominence,
        }
        report = make_markdown_report(project_name=project_name, spectrum=spectrum, preprocessing=preprocessing_config, peaks=peaks, notes=notes)
        if len(st.session_state.processed) > 1:
            report += f"\n\n## Dataset context\n\nThis project currently contains {len(st.session_state.processed)} imported spectra. Use the Dataset and Chemometrics tabs to export aligned matrices and PCA results.\n"
        st.download_button("Download Markdown report", report, file_name=f"{selected}_report.md", mime="text/markdown")
        st.markdown(report)

with tabs[7]:
    st.subheader("Chemist guide")
    st.markdown("""
You do **not** need to be an AI engineer. Your main job is to create trustworthy spectra.

**For every sample, record:** compound/class, concentration, solvent, pH, temperature, instrument, integration time, operator, date, and notes.

**A good first publishable dataset contains:** blanks, pure standards, concentration series, mixtures, adulterants, replicates, and blind validation samples.

**Interpretation rule:** AI predictions are screening and decision-support. Confirm important findings using validated laboratory methods.
""")
