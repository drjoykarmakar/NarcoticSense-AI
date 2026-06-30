from __future__ import annotations


def uncertainty_sampling(probability_rows: list[dict[str, float]], k: int = 10) -> list[int]:
    """Return indices with the lowest maximum class probability."""
    scored = [(idx, max(row.values()) if row else 0.0) for idx, row in enumerate(probability_rows)]
    return [idx for idx, _ in sorted(scored, key=lambda item: item[1])[:k]]
