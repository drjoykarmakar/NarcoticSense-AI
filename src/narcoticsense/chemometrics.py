from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ProjectionResult:
    coordinates: pd.DataFrame
    explained_variance: list[float]
    loadings: pd.DataFrame | None = None


def run_pca(
    matrix: np.ndarray, sample_ids: list[str], x_axis: np.ndarray, *, n_components: int = 3
) -> ProjectionResult:
    if matrix.shape[0] < 2:
        raise ValueError("PCA needs at least two spectra.")
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(scaled)
    coord_df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_components)])
    coord_df.insert(0, "sample_id", sample_ids)
    loadings = pd.DataFrame({"x": x_axis})
    for i in range(n_components):
        loadings[f"PC{i+1}_loading"] = pca.components_[i]
    return ProjectionResult(coord_df, pca.explained_variance_ratio_.tolist(), loadings)


def run_tsne(matrix: np.ndarray, sample_ids: list[str]) -> pd.DataFrame:
    if matrix.shape[0] < 4:
        raise ValueError("t-SNE needs at least four spectra.")
    scaled = StandardScaler().fit_transform(matrix)
    perplexity = max(2, min(30, matrix.shape[0] // 2))
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42
    ).fit_transform(scaled)
    return pd.DataFrame({"sample_id": sample_ids, "TSNE1": coords[:, 0], "TSNE2": coords[:, 1]})


def run_kmeans(matrix: np.ndarray, sample_ids: list[str], *, n_clusters: int = 2) -> pd.DataFrame:
    if matrix.shape[0] < n_clusters:
        raise ValueError("Number of clusters cannot exceed number of spectra.")
    scaled = StandardScaler().fit_transform(matrix)
    labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(scaled)
    return pd.DataFrame({"sample_id": sample_ids, "cluster": labels.astype(int)})
