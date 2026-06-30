from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _difference_matrix(length: int, order: int = 2):
    if length <= order:
        raise ValueError("Spectrum is too short for baseline correction.")
    return (
        sparse.eye(length, format="csc")[order:]
        - 2 * sparse.eye(length, format="csc")[order - 1 : -1]
        + sparse.eye(length, format="csc")[:-order]
    )


def asymmetric_least_squares(
    y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10
) -> np.ndarray:
    """Asymmetric least-squares baseline correction.

    Eilers-style AsLS is widely used for Raman/fluorescence/IR spectra.
    Returns the estimated baseline, not the corrected signal.
    """
    y = np.asarray(y, dtype=float)
    length = y.size
    if length < 3:
        return np.zeros_like(y)
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float).tocsr()
    weights = np.ones(length)
    for _ in range(n_iter):
        w = sparse.spdiags(weights, 0, length, length).tocsr()
        z = spsolve((w + lam * d.dot(d.transpose())).tocsc(), weights * y)
        weights = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def airpls(y: np.ndarray, lam: float = 1e5, n_iter: int = 15) -> np.ndarray:
    """Adaptive iteratively reweighted penalized least-squares baseline.

    This lightweight implementation is intended for exploratory research UI use.
    """
    y = np.asarray(y, dtype=float)
    m = y.size
    if m < 3:
        return np.zeros_like(y)
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m - 2), dtype=float).tocsc()
    h = lam * d.dot(d.transpose())
    w = np.ones(m)
    z = np.zeros(m)
    for i in range(1, n_iter + 1):
        w_mat = sparse.spdiags(w, 0, m, m).tocsc()
        z = spsolve(w_mat + h, w * y)
        residual = y - z
        negative = residual[residual < 0]
        if negative.size == 0 or np.sum(np.abs(negative)) < 1e-12:
            break
        w[residual >= 0] = 0
        scale = np.mean(np.abs(negative)) or 1.0
        w[residual < 0] = np.exp(i * np.abs(residual[residual < 0]) / scale)
        w[0] = w[-1] = np.max(w)
    return np.asarray(z)


def arpls(y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6, n_iter: int = 50) -> np.ndarray:
    """Asymmetrically reweighted penalized least-squares baseline.

    Returns the estimated baseline. Robust for sloping fluorescence backgrounds.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.zeros_like(y)
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2), dtype=float).tocsc()
    h = lam * d.dot(d.transpose())
    w = np.ones(n)
    z = np.zeros(n)
    for _ in range(n_iter):
        w_old = w.copy()
        z = spsolve(sparse.spdiags(w, 0, n, n).tocsc() + h, w * y)
        diff = y - z
        negative = diff[diff < 0]
        if negative.size == 0:
            break
        mean = np.mean(negative)
        std = np.std(negative) or 1.0
        expo = np.clip(2.0 * (diff - (2 * std - mean)) / std, -60, 60)
        w = 1.0 / (1.0 + np.exp(expo))
        if np.linalg.norm(w - w_old) / max(np.linalg.norm(w_old), 1e-12) < ratio:
            break
    return np.asarray(z)


def estimate_baseline(y: np.ndarray, *, method: str = "asls", lam: float = 1e5) -> np.ndarray:
    method = method.lower()
    if method in {"asls", "als", "asymmetric least squares"}:
        return asymmetric_least_squares(y, lam=lam)
    if method == "airpls":
        return airpls(y, lam=lam)
    if method == "arpls":
        return arpls(y, lam=lam)
    if method in {"none", "off", "no baseline"}:
        return np.zeros_like(np.asarray(y, dtype=float))
    raise ValueError(f"Unknown baseline method: {method}")
