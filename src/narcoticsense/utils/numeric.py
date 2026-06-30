from __future__ import annotations

import numpy as np


def trapezoid_integral(y, x=None) -> float:
    """Return trapezoidal integral across NumPy versions.

    NumPy 2.x exposes ``np.trapezoid``. Older NumPy versions exposed
    ``np.trapz``. This helper keeps the app working on Anaconda/Mac installs
    that may have either API.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
