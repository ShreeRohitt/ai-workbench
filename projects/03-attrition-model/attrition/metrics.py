"""Ranking metrics for a capacity-limited HR team. Core logic: written by Rohit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Fraction of the k highest-scored people who actually left.

    y_true: 1 = left, 0 = stayed (one entry per person)
    scores: model risk score per person, same order as y_true (higher = more likely to leave)
    k:      how many people HR contacts, 1 <= k <= len(y_true)

    Ties: people with equal scores keep their original order (the earlier row ranks higher).
    Raises ValueError if lengths differ or k is out of range.
    """
    y_true = np.array(y_true)
    scores = np.array(scores)

    if len(y_true) != len(scores):
        raise ValueError("Lengths of y_true and scores differ")

    if k < 1 or k > len(y_true):
        raise ValueError("k is out of range")

    # Rank people by score, highest first
    ranked_indices = np.argsort(-scores, kind="stable")  # stable sort keeps original order for ties

    # Take the top k and return the fraction of them who left
    top_k_indices = ranked_indices[:k]
    top_k_left = y_true[top_k_indices]

    return float(np.mean(top_k_left))
