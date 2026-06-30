from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def asymmetric_least_squares(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    length = y.size
    d = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float).tocsr()
    weights = np.ones(length)
    for _ in range(n_iter):
        w = sparse.spdiags(weights, 0, length, length).tocsr()
        z = spsolve((w + lam * d.dot(d.transpose())).tocsc(), weights * y)
        weights = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)
