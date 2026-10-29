"""Ranking metrics for a capacity-limited HR team. Core logic: written by Rohit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np  # noqa: F401  (you will need it)


def precision_at_k(y_true: Sequence[int], scores: Sequence[float], k: int) -> float:
    """Fraction of the k highest-scored people who actually left.

    y_true: 1 = left, 0 = stayed (one entry per person)
    scores: model risk score per person, same order as y_true (higher = more likely to leave)
    k:      how many people HR contacts, 1 <= k <= len(y_true)

    Ties: people with equal scores keep their original order (the earlier row ranks higher).
    Raises ValueError if lengths differ or k is out of range.
    """
    raise NotImplementedError("Rohit: implement me")
