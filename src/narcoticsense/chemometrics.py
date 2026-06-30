from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import OneClassSVM


@dataclass(slots=True)
class ProjectionResult:
    coordinates: pd.DataFrame
    explained_variance: list[float]
    loadings: pd.DataFrame | None = None


@dataclass(slots=True)
class RegressionResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame


@dataclass(slots=True)
class SimcaResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _scale(matrix: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(matrix, dtype=float))


def run_pca(
    matrix: np.ndarray, sample_ids: list[str], x_axis: np.ndarray, *, n_components: int = 3
) -> ProjectionResult:
    if matrix.shape[0] < 2:
        raise ValueError("PCA needs at least two spectra.")
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    scaled = _scale(matrix)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)
    coord_df = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(n_components)])
    coord_df.insert(0, "sample_id", sample_ids)
    loadings = pd.DataFrame({"x": x_axis})
    for i in range(n_components):
        loadings[f"PC{i + 1}_loading"] = pca.components_[i]
    return ProjectionResult(coord_df, pca.explained_variance_ratio_.tolist(), loadings)


def run_tsne(matrix: np.ndarray, sample_ids: list[str]) -> pd.DataFrame:
    if matrix.shape[0] < 4:
        raise ValueError("t-SNE needs at least four spectra.")
    scaled = _scale(matrix)
    perplexity = max(2, min(30, matrix.shape[0] // 2))
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42
    ).fit_transform(scaled)
    return pd.DataFrame({"sample_id": sample_ids, "TSNE1": coords[:, 0], "TSNE2": coords[:, 1]})


def run_umap(matrix: np.ndarray, sample_ids: list[str], *, n_neighbors: int = 15) -> pd.DataFrame:
    """Run UMAP if installed, otherwise use a documented PCA fallback.

    UMAP is optional to keep installation lightweight. The returned table always has
    UMAP1 and UMAP2 columns plus a method column describing the actual algorithm.
    """
    if matrix.shape[0] < 3:
        raise ValueError("UMAP needs at least three spectra.")
    scaled = _scale(matrix)
    try:
        from umap import UMAP  # type: ignore

        n_neighbors = max(2, min(int(n_neighbors), matrix.shape[0] - 1))
        coords = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(
            scaled
        )
        method = "UMAP"
    except Exception:
        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
        method = "PCA fallback (install umap-learn for UMAP)"
    return pd.DataFrame(
        {"sample_id": sample_ids, "UMAP1": coords[:, 0], "UMAP2": coords[:, 1], "method": method}
    )


def run_kmeans(matrix: np.ndarray, sample_ids: list[str], *, n_clusters: int = 2) -> pd.DataFrame:
    if matrix.shape[0] < n_clusters:
        raise ValueError("Number of clusters cannot exceed number of spectra.")
    scaled = _scale(matrix)
    labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(scaled)
    return pd.DataFrame({"sample_id": sample_ids, "cluster": labels.astype(int)})


def run_hca(matrix: np.ndarray, sample_ids: list[str], *, method: str = "ward") -> pd.DataFrame:
    """Return hierarchical clustering linkage table for dendrogram export."""
    if matrix.shape[0] < 2:
        raise ValueError("HCA needs at least two spectra.")
    scaled = _scale(matrix)
    z = linkage(scaled, method=method)
    rows = []
    for i, row in enumerate(z):
        rows.append(
            {
                "merge_step": i + 1,
                "cluster_1": int(row[0]),
                "cluster_2": int(row[1]),
                "distance": float(row[2]),
                "n_samples": int(row[3]),
            }
        )
    return pd.DataFrame(rows)


def hca_dendrogram_coordinates(
    matrix: np.ndarray, sample_ids: list[str], *, method: str = "ward"
) -> dict[str, object]:
    if matrix.shape[0] < 2:
        raise ValueError("HCA needs at least two spectra.")
    scaled = _scale(matrix)
    z = linkage(scaled, method=method)
    return dendrogram(z, labels=sample_ids, no_plot=True)


def run_pls_da(
    matrix: np.ndarray, sample_ids: list[str], labels: list[str], *, n_components: int = 2
) -> ProjectionResult:
    """Run a simple PLS-DA style projection using one-hot labels.

    This is intended for exploratory visualization, not regulatory decision-making.
    """
    if len(set(labels)) < 2:
        raise ValueError("PLS-DA needs at least two classes.")
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    if n_components < 1:
        raise ValueError("Not enough samples/features for PLS-DA.")
    scaled = _scale(matrix)
    y = LabelBinarizer().fit_transform(labels)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    pls = PLSRegression(n_components=n_components)
    coords = pls.fit_transform(scaled, y)[0]
    coord_df = pd.DataFrame(coords, columns=[f"LV{i + 1}" for i in range(n_components)])
    coord_df.insert(0, "sample_id", sample_ids)
    coord_df.insert(1, "label", labels)
    loadings = pd.DataFrame({"variable_index": np.arange(matrix.shape[1])})
    for i in range(n_components):
        loadings[f"LV{i + 1}_x_weight"] = pls.x_weights_[:, i]
    return ProjectionResult(coord_df, [], loadings)


def run_pls_regression(
    matrix: np.ndarray,
    sample_ids: list[str],
    targets: list[float],
    *,
    n_components: int = 2,
    cv: str = "loo",
) -> RegressionResult:
    """Cross-validated PLS regression for concentration or continuous responses."""
    y = np.asarray(targets, dtype=float)
    if matrix.shape[0] != y.size:
        raise ValueError("Targets must match number of spectra.")
    if matrix.shape[0] < 3:
        raise ValueError("PLS regression needs at least three labeled spectra.")
    n_components = max(1, min(int(n_components), matrix.shape[0] - 1, matrix.shape[1]))
    scaled = _scale(matrix)
    pls = PLSRegression(n_components=n_components)
    splitter = LeaveOneOut() if cv == "loo" else int(cv)
    pred = cross_val_predict(pls, scaled, y, cv=splitter).ravel()
    residuals = y - pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pls.fit(scaled, y)
    predictions = pd.DataFrame(
        {"sample_id": sample_ids, "observed": y, "predicted": pred, "residual": residuals}
    )
    metrics = pd.DataFrame(
        [
            {"metric": "rmse_cv", "value": rmse},
            {"metric": "mae_cv", "value": mae},
            {"metric": "r2_cv", "value": float(r2)},
            {"metric": "n_components", "value": float(n_components)},
        ]
    )
    coefficients = pd.DataFrame(
        {"variable_index": np.arange(matrix.shape[1]), "coefficient": pls.coef_.ravel()}
    )
    return RegressionResult(predictions, metrics, coefficients)


def run_lda(matrix: np.ndarray, sample_ids: list[str], labels: list[str]) -> ProjectionResult:
    """Linear discriminant analysis score projection."""
    y = np.asarray(labels, dtype=str)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("LDA needs at least two classes.")
    max_components = min(classes.size - 1, matrix.shape[0] - classes.size, matrix.shape[1])
    if max_components < 1:
        raise ValueError("Not enough samples per class for LDA projection.")
    scaled = _scale(matrix)
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    coords = lda.fit_transform(scaled, y)
    coord_df = pd.DataFrame(coords, columns=[f"LD{i + 1}" for i in range(coords.shape[1])])
    coord_df.insert(0, "sample_id", sample_ids)
    coord_df.insert(1, "label", labels)
    loadings = pd.DataFrame({"variable_index": np.arange(matrix.shape[1])})
    scalings = np.asarray(lda.scalings_)[:, : coords.shape[1]]
    for i in range(coords.shape[1]):
        loadings[f"LD{i + 1}_coefficient"] = scalings[:, i]
    return ProjectionResult(coord_df, [], loadings)


def run_simca(
    matrix: np.ndarray,
    sample_ids: list[str],
    labels: list[str],
    *,
    n_components: int = 2,
    quantile: float = 0.95,
) -> SimcaResult:
    """Simple SIMCA-style one-class PCA model per class.

    Each class is modeled independently by PCA reconstruction error. A sample is
    assigned to the class with the lowest normalized reconstruction distance.
    """
    y = np.asarray(labels, dtype=str)
    classes = sorted(np.unique(y).tolist())
    if len(classes) < 2:
        raise ValueError("SIMCA needs at least two classes.")
    scaled = _scale(matrix)
    rows = []
    thresholds: dict[str, float] = {}
    class_models: dict[str, tuple[PCA, np.ndarray, float]] = {}
    for cls in classes:
        idx = y == cls
        if int(np.sum(idx)) < 2:
            continue
        n_comp = max(1, min(int(n_components), int(np.sum(idx)) - 1, matrix.shape[1]))
        pca = PCA(n_components=n_comp, random_state=42).fit(scaled[idx])
        recon = pca.inverse_transform(pca.transform(scaled[idx]))
        dist = np.sqrt(np.mean((scaled[idx] - recon) ** 2, axis=1))
        threshold = float(np.quantile(dist, quantile))
        threshold = threshold if threshold > 0 else float(np.max(dist) + 1e-12)
        thresholds[cls] = threshold
        class_models[cls] = (pca, scaled[idx].mean(axis=0), threshold)
    if len(class_models) < 2:
        raise ValueError("SIMCA needs at least two classes with two or more samples.")
    for sample_id, vector, true in zip(sample_ids, scaled, y, strict=True):
        scores = {}
        for cls, (pca, _center, threshold) in class_models.items():
            recon = pca.inverse_transform(pca.transform(vector.reshape(1, -1)))[0]
            dist = float(np.sqrt(np.mean((vector - recon) ** 2)))
            scores[cls] = dist / threshold
        predicted = min(scores, key=scores.get)
        accepted = [cls for cls, score in scores.items() if score <= 1.0]
        rows.append(
            {
                "sample_id": sample_id,
                "true_label": true,
                "predicted_label": predicted,
                "accepted_classes": ";".join(accepted) if accepted else "none",
                "normalized_distance": scores[predicted],
                "correct": predicted == true,
            }
        )
    predictions = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": float(predictions["correct"].mean())},
            {"metric": "classes_modeled", "value": float(len(class_models))},
            {"metric": "n_components", "value": float(n_components)},
        ]
    )
    return SimcaResult(predictions, metrics)


def distance_matrix(
    matrix: np.ndarray, sample_ids: list[str], *, metric: str = "euclidean"
) -> pd.DataFrame:
    scaled = _scale(matrix)
    d = pairwise_distances(scaled, metric=metric)
    return pd.DataFrame(d, index=sample_ids, columns=sample_ids)


def one_class_outlier_scores(matrix: np.ndarray, sample_ids: list[str]) -> pd.DataFrame:
    if matrix.shape[0] < 4:
        raise ValueError("One-class outlier scoring needs at least four spectra.")
    scaled = _scale(matrix)
    model = OneClassSVM(gamma="scale", nu=0.1).fit(scaled)
    score = model.decision_function(scaled).ravel()
    pred = model.predict(scaled)
    return pd.DataFrame({"sample_id": sample_ids, "outlier_score": score, "is_outlier": pred == -1})
