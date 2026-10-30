"""Baseline models in the shape `cross_validate` expects: (fit_df, eval_df) -> scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from attrition.cv import ScoreFn


def random_model(seed: int = 0) -> ScoreFn:
    """Random scores. Expected precision@k = base rate (~16%). (Boilerplate: Claude.)"""
    rng = np.random.default_rng(seed)

    def score(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        return rng.random(len(eval_df))

    return score


def overtime_level1_rule(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    """Score 1.0 if OverTime == "Yes" and JobLevel == 1, else 0.0.

    Ignores fit_df: a fixed rule has nothing to learn.

    Core logic: Rohit.
    """
    raise NotImplementedError("Rohit: implement me")
