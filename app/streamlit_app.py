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
    from narcoticsense.chemometrics import (
        distance_matrix,
        run_hca,
        run_kmeans,
        run_lda,
        run_pca,
        run_pls_da,
        run_pls_regression,
        run_simca,
        run_tsne,
        run_umap,
    )
    from narcoticsense.explainability import peak_level_attribution, spectral_occlusion_importance
    from narcoticsense.feature_engineering import derivative_spectrum, peak_table, spectral_metrics
    from narcoticsense.library import library_match
    from narcoticsense.models import (
        available_classification_models,
        available_regression_models,
        build_classifier,
        save_model_bundle,
        train_classification_model,
        train_regression_model,
    )
    from narcoticsense.preprocessing import PreprocessingPipeline
    from narcoticsense.reports import make_markdown_report
    from narcoticsense.spectroscopy import (
        align_spectra,
        dataset_summary,
        import_spectrum,
        spectra_to_long_dataframe,
    )
    from narcoticsense.uncertainty import (
        cross_validated_model_comparison,
        trustworthy_classification_report,
    )
    from narcoticsense.validation import (
        dataset_quality_report,
    )
    from narcoticsense.visualization import plot_overlay, plot_peaks, plot_projection, plot_spectrum
except Exception as exc:
    st.set_page_config(page_title="NarcoticSense AI setup error", page_icon="🔬")
    st.error("NarcoticSense AI could not import its local Python package.")
    st.write("Please run the app from the project folder, not from an old copied app file.")
    st.code(f"Project root: {PROJECT_ROOT}\nSource folder: {SRC_DIR}\nPython error: {exc}")
    st.stop()

st.set_page_config(page_title="NarcoticSense AI", page_icon="🔬", layout="wide")

st.title("NarcoticSense AI")
st.caption(
    "Chemist-friendly AI spectroscopy operating system for research, sensing, and decision support"
)
st.warning(
    "Research and decision-support only. Confirm high-stakes results with validated laboratory methods."
)

if "spectra" not in st.session_state:
    st.session_state.spectra = []
if "processed" not in st.session_state:
    st.session_state.processed = []
if "import_results" not in st.session_state:
    st.session_state.import_results = []
if "metadata" not in st.session_state:
    st.session_state.metadata = pd.DataFrame()

with st.sidebar:
    st.header("Project")
    project_name = st.text_input("Project name", value="My Spectroscopy Project")
    modality = st.selectbox(
        "Modality",
        ["fluorescence", "raman", "infrared", "uv-vis", "hyperspectral", "unknown"],
        index=0,
    )
    default_label = st.text_input("Default sample label", value="sample")
    st.divider()
    st.header("Preprocessing")
    smooth = st.checkbox("Savitzky-Golay smoothing", value=True)
    baseline = st.checkbox("Baseline correction", value=True)
    baseline_method = st.selectbox("Baseline method", ["asls", "airpls", "arpls"], index=0)
    baseline_lambda = st.select_slider(
        "Baseline stiffness λ",
        options=[1e3, 1e4, 1e5, 1e6, 1e7],
        value=1e5,
        format_func=lambda v: f"{v:.0e}",
    )
    normalization = st.selectbox("Normalization", ["minmax", "snv", "none"], index=0)
    window_length = st.slider("Smoothing window", 5, 51, 11, step=2)
    polyorder = st.slider("Polynomial order", 2, 5, 3)
    prominence = st.slider("Peak prominence", 0.001, 1.0, 0.05, step=0.001, format="%.3f")
    st.divider()
    if st.button("Clear uploaded spectra"):
        st.session_state.spectra = []
        st.session_state.processed = []
        st.session_state.import_results = []
        st.session_state.metadata = pd.DataFrame()
        st.rerun()

pipe = PreprocessingPipeline(
    smooth=smooth,
    baseline=baseline,
    normalization=normalization,
    window_length=window_length,
    polyorder=polyorder,
    baseline_method=baseline_method,
    baseline_lambda=float(baseline_lambda),
)

tabs = st.tabs(
    [
        "1 Import",
        "2 Dataset",
        "3 Spectra Viewer",
        "4 Peak Analysis",
        "5 Chemometrics",
        "6 Library Match",
        "7 AI Model Engine",
        "8 Trustworthy AI",
        "9 Report",
        "Chemist Guide",
    ]
)


def _reprocess_all() -> None:
    st.session_state.processed = [pipe.transform_one(s) for s in st.session_state.spectra]


with tabs[0]:
    st.subheader("Universal spectroscopy import")
    st.write(
        "Upload one or many CSV/TXT/TSV files. NarcoticSense AI will detect columns such as `Wavelength (nm)`, `Raman Shift`, `Abs`, `Intensity`, or `Counts`."
    )
    uploaded_files = st.file_uploader(
        "Upload spectrum files", type=["csv", "txt", "tsv"], accept_multiple_files=True
    )
    example = pd.DataFrame(
        {"Wavelength (nm)": [200, 201, 202, 203, 204], "Abs": [0.12, 0.15, 0.19, 0.16, 0.22]}
    )
    st.download_button(
        "Download example CSV",
        example.to_csv(index=False),
        file_name="example_spectrum.csv",
        mime="text/csv",
    )

    if uploaded_files:
        spectra = []
        processed = []
        results = []
        for idx, uploaded in enumerate(uploaded_files, start=1):
            sample_id = Path(uploaded.name).stem or f"{default_label}-{idx:03d}"
            try:
                result = import_spectrum(
                    uploaded.getvalue(), modality=modality, sample_id=sample_id
                )
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
            rows.append(
                {
                    "sample_id": result.spectrum.sample_id,
                    "points": result.spectrum.n_points,
                    "x column": result.x_column,
                    "y column": result.y_column,
                    "confidence": f"{result.confidence:.0%}",
                    "notes": "; ".join(result.messages),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with st.expander("Preview first imported file"):
            st.dataframe(
                st.session_state.import_results[0].dataframe.head(30), use_container_width=True
            )

with tabs[1]:
    st.subheader("Dataset manager")
    if not st.session_state.spectra:
        st.info("Import spectra first.")
    else:
        summary = dataset_summary(st.session_state.spectra)
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download dataset summary",
            summary.to_csv(index=False),
            file_name="dataset_summary.csv",
            mime="text/csv",
        )
        st.markdown("### Metadata template for your lab notebook")
        meta_template = pd.DataFrame(
            {
                "sample_id": [s.sample_id for s in st.session_state.spectra],
                "compound_or_class": ["" for _ in st.session_state.spectra],
                "concentration": ["" for _ in st.session_state.spectra],
                "solvent": ["" for _ in st.session_state.spectra],
                "pH": ["" for _ in st.session_state.spectra],
                "temperature_C": ["" for _ in st.session_state.spectra],
                "instrument": ["" for _ in st.session_state.spectra],
                "operator": ["" for _ in st.session_state.spectra],
                "notes": ["" for _ in st.session_state.spectra],
            }
        )
        st.dataframe(meta_template, use_container_width=True)
        st.download_button(
            "Download metadata template",
            meta_template.to_csv(index=False),
            file_name="metadata_template.csv",
            mime="text/csv",
        )
        st.markdown("### Dataset quality checks")
        quality = dataset_quality_report(st.session_state.processed)
        st.dataframe(quality, use_container_width=True)
        metadata_file = st.file_uploader(
            "Optional: upload completed metadata CSV with sample_id and compound_or_class columns",
            type=["csv"],
            key="metadata_upload",
        )
        if metadata_file is not None:
            try:
                metadata = pd.read_csv(metadata_file)
                if "sample_id" not in metadata.columns:
                    st.error("Metadata file must contain a sample_id column.")
                else:
                    st.session_state.metadata = metadata
                    st.success(f"Loaded metadata for {len(metadata)} rows.")
                    st.dataframe(metadata, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read metadata CSV: {exc}")

with tabs[2]:
    st.subheader("Professional spectra viewer")
    if not st.session_state.spectra:
        st.info("Import spectra first.")
    else:
        view_choice = st.radio(
            "View", ["Processed", "Raw", "Raw vs processed", "Derivative"], horizontal=True
        )
        spectra_to_plot = (
            st.session_state.processed if view_choice == "Processed" else st.session_state.spectra
        )
        if view_choice == "Raw vs processed":
            selected = st.selectbox(
                "Select spectrum", [s.sample_id for s in st.session_state.spectra]
            )
            idx = [s.sample_id for s in st.session_state.spectra].index(selected)
            fig = plot_overlay(
                [st.session_state.spectra[idx], st.session_state.processed[idx]],
                title=f"Raw vs processed: {selected}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download interactive figure HTML",
                fig.to_html(include_plotlyjs="cdn"),
                file_name=f"{selected}_raw_vs_processed.html",
                mime="text/html",
            )
        elif view_choice == "Derivative":
            order = st.radio("Derivative order", [1, 2], horizontal=True)
            deriv = [derivative_spectrum(s, order=order) for s in st.session_state.processed]
            fig = plot_overlay(deriv, title=f"{order} derivative overlay")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download interactive derivative HTML",
                fig.to_html(include_plotlyjs="cdn"),
                file_name=f"derivative_order_{order}.html",
                mime="text/html",
            )
        else:
            fig = plot_overlay(spectra_to_plot, title=f"{view_choice} spectral overlay")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download interactive overlay HTML",
                fig.to_html(include_plotlyjs="cdn"),
                file_name=f"{view_choice.lower()}_spectral_overlay.html",
                mime="text/html",
            )
        if st.session_state.processed:
            selected = st.selectbox(
                "Single-spectrum detail",
                [s.sample_id for s in st.session_state.processed],
                key="detail_select",
            )
            idx = [s.sample_id for s in st.session_state.processed].index(selected)
            st.plotly_chart(
                plot_spectrum(
                    st.session_state.processed[idx], title=f"Processed detail: {selected}"
                ),
                use_container_width=True,
            )
            out = pd.DataFrame(
                {"x": st.session_state.processed[idx].x, "y": st.session_state.processed[idx].y}
            )
            st.download_button(
                "Download selected processed CSV",
                out.to_csv(index=False),
                file_name=f"{selected}_processed.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download all processed spectra long CSV",
                spectra_to_long_dataframe(st.session_state.processed).to_csv(index=False),
                file_name="all_processed_spectra_long.csv",
                mime="text/csv",
            )

with tabs[3]:
    st.subheader("Peak analysis and spectral metrics")
    if not st.session_state.processed:
        st.info("Import spectra first.")
    else:
        selected = st.selectbox(
            "Select spectrum for peak analysis",
            [s.sample_id for s in st.session_state.processed],
            key="peak_select",
        )
        idx = [s.sample_id for s in st.session_state.processed].index(selected)
        spectrum = st.session_state.processed[idx]
        peaks = peak_table(spectrum, prominence=prominence, max_peaks=100)
        metrics = spectral_metrics(spectrum)
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected peaks", len(peaks))
        c2.metric(
            "Max position",
            f"{metrics.loc[metrics.metric == 'maximum position', 'value'].iloc[0]:.3g}",
        )
        c3.metric(
            "Absolute area",
            f"{metrics.loc[metrics.metric == 'absolute area', 'value'].iloc[0]:.3g}",
        )
        st.plotly_chart(plot_peaks(spectrum, peaks), use_container_width=True)
        left, right = st.columns(2)
        left.dataframe(peaks, use_container_width=True)
        right.dataframe(metrics, use_container_width=True)
        st.download_button(
            "Download peak table",
            peaks.to_csv(index=False),
            file_name=f"{selected}_peaks.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download metrics",
            metrics.to_csv(index=False),
            file_name=f"{selected}_metrics.csv",
            mime="text/csv",
        )

with tabs[4]:
    st.subheader("Chemometrics and validation")
    if len(st.session_state.processed) < 2:
        st.info("Import at least two spectra for PCA. Four or more are recommended for t-SNE/UMAP.")
    else:
        n_points = st.slider("Common grid points", 100, 3000, 1000, step=100)
        aligned = align_spectra(st.session_state.processed, n_points=n_points)
        st.write(
            f"Aligned dataset: {aligned.n_samples} spectra × {aligned.n_features} spectral variables."
        )
        st.download_button(
            "Download aligned ML matrix",
            aligned.to_dataframe().to_csv(index=False),
            file_name="aligned_spectral_matrix.csv",
            mime="text/csv",
        )

        pca_tab, cluster_tab, supervised_tab, regression_tab, outlier_tab = st.tabs(
            [
                "PCA/Manifold",
                "Clustering",
                "PLS-DA/LDA/SIMCA",
                "PLS Regression",
                "Distance/Outliers",
            ]
        )

        with pca_tab:
            pca = run_pca(aligned.matrix, aligned.sample_ids, aligned.x, n_components=3)
            ev = [round(v * 100, 2) for v in pca.explained_variance]
            st.write(f"PCA explained variance: {ev}%")
            st.plotly_chart(
                plot_projection(pca.coordinates, "PC1", "PC2", "PCA score plot"),
                use_container_width=True,
            )
            st.dataframe(pca.coordinates, use_container_width=True)
            if pca.loadings is not None:
                st.line_chart(pca.loadings.set_index("x"))
                st.download_button(
                    "Download PCA loadings",
                    pca.loadings.to_csv(index=False),
                    file_name="pca_loadings.csv",
                    mime="text/csv",
                )
            if aligned.n_samples >= 4:
                try:
                    tsne_df = run_tsne(aligned.matrix, aligned.sample_ids)
                    st.plotly_chart(
                        plot_projection(tsne_df, "TSNE1", "TSNE2", "t-SNE exploratory map"),
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.info(f"t-SNE skipped: {exc}")
            if aligned.n_samples >= 3:
                try:
                    umap_df = run_umap(aligned.matrix, aligned.sample_ids)
                    st.plotly_chart(
                        plot_projection(umap_df, "UMAP1", "UMAP2", "UMAP/PCA manifold map"),
                        use_container_width=True,
                    )
                    st.caption(str(umap_df["method"].iloc[0]))
                    st.download_button(
                        "Download UMAP/manifold scores",
                        umap_df.to_csv(index=False),
                        file_name="umap_scores.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.info(f"UMAP skipped: {exc}")

        with cluster_tab:
            if aligned.n_samples >= 2:
                k = st.slider("K-means clusters", 2, min(8, aligned.n_samples), 2)
                try:
                    clusters = run_kmeans(aligned.matrix, aligned.sample_ids, n_clusters=k)
                    st.dataframe(clusters, use_container_width=True)
                except Exception as exc:
                    st.info(f"K-means skipped: {exc}")
                try:
                    hca = run_hca(aligned.matrix, aligned.sample_ids)
                    st.markdown("Hierarchical clustering linkage table")
                    st.dataframe(hca, use_container_width=True)
                    st.download_button(
                        "Download HCA linkage table",
                        hca.to_csv(index=False),
                        file_name="hca_linkage.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.info(f"HCA skipped: {exc}")

        labels = None
        label_col = None
        if not st.session_state.metadata.empty and "sample_id" in st.session_state.metadata.columns:
            candidate_cols = [c for c in st.session_state.metadata.columns if c != "sample_id"]
            if candidate_cols:
                label_col = st.selectbox(
                    "Classification label column",
                    candidate_cols,
                    index=0,
                    key="chemometrics_label_col",
                )
                labels_frame = pd.DataFrame({"sample_id": aligned.sample_ids}).merge(
                    st.session_state.metadata[["sample_id", label_col]],
                    on="sample_id",
                    how="left",
                )
                labels = labels_frame[label_col].fillna("").astype(str).tolist()

        with supervised_tab:
            if labels is None or len(set([label for label in labels if label])) < 2:
                st.info("Upload metadata with at least two classes to run PLS-DA, LDA, and SIMCA.")
            else:
                try:
                    plsda = run_pls_da(aligned.matrix, aligned.sample_ids, labels, n_components=2)
                    st.plotly_chart(
                        plot_projection(
                            plsda.coordinates, "LV1", "LV2", "PLS-DA exploratory score plot"
                        ),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download PLS-DA scores",
                        plsda.coordinates.to_csv(index=False),
                        file_name="pls_da_scores.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.info(f"PLS-DA skipped: {exc}")
                try:
                    lda = run_lda(aligned.matrix, aligned.sample_ids, labels)
                    y_axis = "LD2" if "LD2" in lda.coordinates.columns else "LD1"
                    st.plotly_chart(
                        plot_projection(lda.coordinates, "LD1", y_axis, "LDA score plot"),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download LDA scores",
                        lda.coordinates.to_csv(index=False),
                        file_name="lda_scores.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.info(f"LDA skipped: {exc}")
                try:
                    simca = run_simca(aligned.matrix, aligned.sample_ids, labels)
                    st.markdown("SIMCA-style one-class PCA results")
                    st.dataframe(simca.metrics, use_container_width=True)
                    st.dataframe(simca.predictions, use_container_width=True)
                    st.download_button(
                        "Download SIMCA predictions",
                        simca.predictions.to_csv(index=False),
                        file_name="simca_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.info(f"SIMCA skipped: {exc}")

        with regression_tab:
            if (
                st.session_state.metadata.empty
                or "sample_id" not in st.session_state.metadata.columns
            ):
                st.info(
                    "Upload metadata with a numeric concentration/response column for PLS regression."
                )
            else:
                numeric_cols = []
                for c in st.session_state.metadata.columns:
                    if c == "sample_id":
                        continue
                    values = pd.to_numeric(st.session_state.metadata[c], errors="coerce")
                    if values.notna().sum() >= 3:
                        numeric_cols.append(c)
                if not numeric_cols:
                    st.info("No numeric metadata column with at least three values was found.")
                else:
                    target_col = st.selectbox("PLS regression target", numeric_cols)
                    joined = pd.DataFrame({"sample_id": aligned.sample_ids}).merge(
                        st.session_state.metadata[["sample_id", target_col]],
                        on="sample_id",
                        how="left",
                    )
                    y_numeric = pd.to_numeric(joined[target_col], errors="coerce")
                    keep = y_numeric.notna().to_numpy()
                    if keep.sum() < 3:
                        st.warning("Need at least three numeric target values.")
                    else:
                        n_comp = st.slider("PLS components", 1, min(10, int(keep.sum()) - 1), 2)
                        try:
                            pls = run_pls_regression(
                                aligned.matrix[keep],
                                joined.loc[keep, "sample_id"].astype(str).tolist(),
                                y_numeric.loc[keep].astype(float).tolist(),
                                n_components=n_comp,
                            )
                            st.dataframe(pls.metrics, use_container_width=True)
                            st.dataframe(pls.predictions, use_container_width=True)
                            st.line_chart(pls.coefficients.set_index("variable_index"))
                            st.download_button(
                                "Download PLS regression predictions",
                                pls.predictions.to_csv(index=False),
                                file_name="pls_regression_predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as exc:
                            st.info(f"PLS regression skipped: {exc}")

        with outlier_tab:
            try:
                distances = distance_matrix(aligned.matrix, aligned.sample_ids)
                st.markdown("Scaled spectral distance matrix")
                st.dataframe(distances, use_container_width=True)
                st.download_button(
                    "Download distance matrix",
                    distances.to_csv(),
                    file_name="spectral_distance_matrix.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.info(f"Distance matrix skipped: {exc}")

with tabs[5]:
    st.subheader("Spectral library matching")
    if len(st.session_state.processed) < 2:
        st.info(
            "Import at least two spectra. Select one as the unknown/query and use the others as references."
        )
    else:
        query_id = st.selectbox(
            "Query spectrum", [s.sample_id for s in st.session_state.processed], key="library_query"
        )
        top_k = st.slider(
            "Top matches",
            1,
            min(10, len(st.session_state.processed) - 1),
            min(5, len(st.session_state.processed) - 1),
        )
        query = next(s for s in st.session_state.processed if s.sample_id == query_id)
        refs = [s for s in st.session_state.processed if s.sample_id != query_id]
        matches = library_match(query, refs, top_k=top_k)
        st.dataframe(matches, use_container_width=True)
        st.download_button(
            "Download library matches",
            matches.to_csv(index=False),
            file_name=f"{query_id}_library_matches.csv",
            mime="text/csv",
        )

with tabs[6]:
    st.subheader("AI Model Engine")
    st.info(
        "This v0.5.0 engine trains transparent classical ML models from your uploaded spectra and metadata. It is for research screening and method development only."
    )
    n_classes = st.number_input(
        "How many chemical classes/conditions?", min_value=2, max_value=100, value=5
    )
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

    if len(st.session_state.processed) >= 4 and not st.session_state.metadata.empty:
        st.markdown("### Train a model")
        task = st.radio("Task", ["Classification", "Regression"], horizontal=True)
        n_points_ml = st.slider("ML grid points", 100, 2000, 500, step=100, key="ai_ml_grid")
        test_size = st.slider("Test set fraction", 0.10, 0.50, 0.25, step=0.05)
        aligned_ml = align_spectra(st.session_state.processed, n_points=n_points_ml)

        if task == "Classification":
            candidate_cols = [c for c in st.session_state.metadata.columns if c != "sample_id"]
            label_col = st.selectbox("Class label column", candidate_cols, index=0)
            model_name = st.selectbox("Classifier", available_classification_models())
            joined = pd.DataFrame({"sample_id": aligned_ml.sample_ids}).merge(
                st.session_state.metadata[["sample_id", label_col]], on="sample_id", how="left"
            )
            keep = joined[label_col].notna() & (joined[label_col].astype(str).str.len() > 0)
            if keep.sum() < 4 or joined.loc[keep, label_col].nunique() < 2:
                st.warning(
                    "Need at least two classes and enough labeled spectra for supervised learning."
                )
            elif st.button("Train classification model"):
                try:
                    result = train_classification_model(
                        aligned_ml.matrix[keep.to_numpy()],
                        joined.loc[keep, label_col].astype(str).tolist(),
                        joined.loc[keep, "sample_id"].astype(str).tolist(),
                        model_name=model_name,
                        test_size=float(test_size),
                    )
                    st.session_state["last_model_result"] = result
                    st.success(f"Trained {model_name} classification model.")
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")
        else:
            numeric_cols = []
            for c in st.session_state.metadata.columns:
                if c == "sample_id":
                    continue
                values = pd.to_numeric(st.session_state.metadata[c], errors="coerce")
                if values.notna().sum() >= 4:
                    numeric_cols.append(c)
            if not numeric_cols:
                st.warning(
                    "Need a numeric metadata column such as concentration with at least four values."
                )
            else:
                target_col = st.selectbox("Numeric target column", numeric_cols)
                model_name = st.selectbox("Regressor", available_regression_models())
                n_comp = st.slider("PLS components", 1, 10, 2)
                joined = pd.DataFrame({"sample_id": aligned_ml.sample_ids}).merge(
                    st.session_state.metadata[["sample_id", target_col]], on="sample_id", how="left"
                )
                y_numeric = pd.to_numeric(joined[target_col], errors="coerce")
                keep = y_numeric.notna().to_numpy()
                if keep.sum() < 4:
                    st.warning("Need at least four numeric target values.")
                elif st.button("Train regression model"):
                    try:
                        result = train_regression_model(
                            aligned_ml.matrix[keep],
                            y_numeric.loc[keep].astype(float).tolist(),
                            joined.loc[keep, "sample_id"].astype(str).tolist(),
                            model_name=model_name,
                            test_size=float(test_size),
                            n_components=int(n_comp),
                        )
                        st.session_state["last_model_result"] = result
                        st.success(f"Trained {model_name} regression model.")
                    except Exception as exc:
                        st.error(f"Model training failed: {exc}")

        result = st.session_state.get("last_model_result")
        if result is not None:
            st.markdown("### Latest model result")
            st.dataframe(result.metrics, use_container_width=True)
            st.markdown("Predictions")
            st.dataframe(result.predictions, use_container_width=True)
            if result.confusion is not None:
                st.markdown("Confusion matrix")
                st.dataframe(result.confusion, use_container_width=True)
            if result.report is not None:
                st.markdown("Classification report")
                st.dataframe(result.report, use_container_width=True)
            if result.probabilities is not None:
                st.markdown("Class probabilities")
                st.dataframe(result.probabilities, use_container_width=True)
            if result.feature_importance is not None:
                st.markdown("Feature importance")
                st.dataframe(result.feature_importance.head(30), use_container_width=True)
            st.download_button(
                "Download model metrics CSV",
                result.metrics.to_csv(index=False),
                file_name="model_metrics.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download model predictions CSV",
                result.predictions.to_csv(index=False),
                file_name="model_predictions.csv",
                mime="text/csv",
            )
            model_path = (
                PROJECT_ROOT
                / "models"
                / f"narcoticsense_{result.task}_{result.model_name.lower().replace(' ', '_')}.joblib"
            )
            if st.button("Save trained model to models/ folder"):
                saved = save_model_bundle(
                    result,
                    model_path,
                    metadata={
                        "project_name": project_name,
                        "n_points": n_points_ml,
                        "modality": modality,
                        "responsible_use": "Research and decision-support only.",
                    },
                )
                st.success(f"Saved model bundle: {saved}")
    else:
        st.info("Import at least four spectra and upload metadata to train supervised models.")

with tabs[7]:
    st.subheader("Trustworthy AI: explainability, uncertainty, and unknown detection")
    st.info(
        "v0.6.0 adds model trust tools: confidence, OOD/unknown flags, conformal prediction sets, cross-validated model comparison, and spectral-region explanations."
    )
    if len(st.session_state.processed) < 4 or st.session_state.metadata.empty:
        st.info("Import at least four spectra and upload metadata to use trustworthy AI tools.")
    else:
        n_points_trust = st.slider(
            "Trust analysis grid points", 100, 2000, 500, step=100, key="trust_grid"
        )
        aligned_trust = align_spectra(st.session_state.processed, n_points=n_points_trust)
        candidate_cols = [c for c in st.session_state.metadata.columns if c != "sample_id"]
        if not candidate_cols:
            st.warning("Metadata needs at least one label column besides sample_id.")
        else:
            label_col_trust = st.selectbox(
                "Trust label column", candidate_cols, key="trust_label_col"
            )
            joined_trust = pd.DataFrame({"sample_id": aligned_trust.sample_ids}).merge(
                st.session_state.metadata[["sample_id", label_col_trust]],
                on="sample_id",
                how="left",
            )
            keep_trust = joined_trust[label_col_trust].notna() & (
                joined_trust[label_col_trust].astype(str).str.len() > 0
            )
            y_trust = joined_trust.loc[keep_trust, label_col_trust].astype(str).tolist()
            x_trust = aligned_trust.matrix[keep_trust.to_numpy()]
            ids_trust = joined_trust.loc[keep_trust, "sample_id"].astype(str).tolist()
            if len(set(y_trust)) < 2 or len(y_trust) < 4:
                st.warning("Need at least two classes and four labeled spectra.")
            else:
                trust_model_name = st.selectbox(
                    "Trust model", available_classification_models(), key="trust_model"
                )
                conf_threshold = st.slider("Low-confidence threshold", 0.50, 0.99, 0.70, step=0.01)
                alpha = st.slider("Conformal alpha", 0.01, 0.30, 0.10, step=0.01)
                model_builders = {
                    name: (lambda n=name: build_classifier(n))
                    for name in available_classification_models()
                }
                if st.button("Run robust validation and trust analysis"):
                    try:
                        comparison = cross_validated_model_comparison(
                            model_builders, x_trust, y_trust
                        )
                        model = build_classifier(trust_model_name)
                        trust = trustworthy_classification_report(
                            model,
                            x_trust,
                            y_trust,
                            x_trust,
                            sample_ids=ids_trust,
                            confidence_threshold=float(conf_threshold),
                            alpha=float(alpha),
                        )
                        st.session_state["trust_analysis"] = {
                            "model": model,
                            "comparison": comparison,
                            "trust": trust,
                            "aligned": aligned_trust,
                            "keep": keep_trust.to_numpy(),
                            "ids": ids_trust,
                            "labels": y_trust,
                        }
                        st.success("Trust analysis complete.")
                    except Exception as exc:
                        st.error(f"Trust analysis failed: {exc}")
                analysis = st.session_state.get("trust_analysis")
                if analysis is not None:
                    st.markdown("### Cross-validated model comparison")
                    st.dataframe(analysis["comparison"], use_container_width=True)
                    st.download_button(
                        "Download model comparison",
                        analysis["comparison"].to_csv(index=False),
                        file_name="model_comparison_cv.csv",
                        mime="text/csv",
                    )
                    trust = analysis["trust"]
                    st.markdown("### Trust summary")
                    st.dataframe(trust.summary, use_container_width=True)
                    st.markdown("### Confidence, conformal prediction sets, and OOD flags")
                    st.dataframe(trust.predictions, use_container_width=True)
                    st.download_button(
                        "Download trust predictions",
                        trust.predictions.to_csv(index=False),
                        file_name="trust_predictions.csv",
                        mime="text/csv",
                    )
                    st.markdown("### Conformal thresholds")
                    st.dataframe(trust.thresholds, use_container_width=True)
                    st.markdown("### Explain one spectrum")
                    explain_id = st.selectbox(
                        "Spectrum to explain", analysis["ids"], key="explain_id"
                    )
                    idx = analysis["ids"].index(explain_id)
                    target_class = str(
                        analysis["model"].predict(
                            analysis["aligned"].matrix[analysis["keep"]][idx].reshape(1, -1)
                        )[0]
                    )
                    window = st.slider("Explanation window size", 3, 101, 21, step=2)
                    try:
                        attr = spectral_occlusion_importance(
                            analysis["model"],
                            analysis["aligned"].matrix[analysis["keep"]][idx],
                            x_axis=analysis["aligned"].x,
                            target_class=target_class,
                            window_size=int(window),
                        )
                        st.write(f"Explaining predicted class: **{target_class}**")
                        st.dataframe(attr.head(30), use_container_width=True)
                        st.line_chart(attr.sort_values("x").set_index("x")[["importance"]])
                        st.download_button(
                            "Download spectral attribution",
                            attr.to_csv(index=False),
                            file_name=f"{explain_id}_spectral_attribution.csv",
                            mime="text/csv",
                        )
                        try:
                            spectrum = next(
                                s for s in st.session_state.processed if s.sample_id == explain_id
                            )
                            peaks = peak_table(spectrum, prominence=prominence, max_peaks=100)
                            peak_attr = peak_level_attribution(attr, peaks)
                            if not peak_attr.empty:
                                st.markdown("### Peak-level attribution")
                                st.dataframe(peak_attr, use_container_width=True)
                                st.download_button(
                                    "Download peak-level attribution",
                                    peak_attr.to_csv(index=False),
                                    file_name=f"{explain_id}_peak_attribution.csv",
                                    mime="text/csv",
                                )
                        except Exception as exc:
                            st.caption(f"Peak-level attribution skipped: {exc}")
                    except Exception as exc:
                        st.info(f"Explanation skipped: {exc}")

with tabs[8]:
    st.subheader("Research report")
    notes = st.text_area(
        "Scientist notes",
        placeholder="Describe sample prep, instrument settings, observations, and concerns.",
    )
    if not st.session_state.processed:
        st.info("Import spectra first.")
    else:
        selected = st.selectbox(
            "Report spectrum",
            [s.sample_id for s in st.session_state.processed],
            key="report_select",
        )
        idx = [s.sample_id for s in st.session_state.processed].index(selected)
        spectrum = st.session_state.processed[idx]
        peaks = peak_table(spectrum, prominence=prominence)
        preprocessing_config = {
            "smooth": smooth,
            "baseline": baseline,
            "normalization": normalization,
            "window_length": window_length,
            "polyorder": polyorder,
            "baseline_method": baseline_method,
            "baseline_lambda": float(baseline_lambda),
            "peak_prominence": prominence,
        }
        report = make_markdown_report(
            project_name=project_name,
            spectrum=spectrum,
            preprocessing=preprocessing_config,
            peaks=peaks,
            notes=notes,
        )
        if len(st.session_state.processed) > 1:
            report += f"\n\n## Dataset context\n\nThis project currently contains {len(st.session_state.processed)} imported spectra. Use the Dataset and Chemometrics tabs to export aligned matrices and PCA results.\n"
        st.download_button(
            "Download Markdown report",
            report,
            file_name=f"{selected}_report.md",
            mime="text/markdown",
        )
        st.markdown(report)

with tabs[9]:
    st.subheader("Chemist guide")
    st.markdown("""
You do **not** need to be an AI engineer. Your main job is to create trustworthy spectra.

**For every sample, record:** compound/class, concentration, solvent, pH, temperature, instrument, integration time, operator, date, and notes.

**A good first publishable dataset contains:** blanks, pure standards, concentration series, mixtures, adulterants, replicates, and blind validation samples.

**Interpretation rule:** AI predictions are screening and decision-support. Confirm important findings using validated laboratory methods.
""")
