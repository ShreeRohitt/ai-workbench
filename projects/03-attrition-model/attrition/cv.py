"""Cross-validation harness: every model is scored on the same folds. Core logic: Rohit."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np  # noqa: F401  (you will need it)
import pandas as pd
from sklearn.metrics import average_precision_score  # noqa: F401  (this is PR-AUC)
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: F401

from attrition.load import TARGET  # noqa: F401
from attrition.metrics import precision_at_k  # noqa: F401

# A model, as the harness sees it: learn from fit_df, return one risk score per row of eval_df.
ScoreFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def cross_validate(
    train: pd.DataFrame,
    make_scores: ScoreFn,
    n_splits: int = 5,
    n_repeats: int = 5,
    k_frac: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Score `make_scores` on repeated stratified folds of `train`.

    For each of n_splits * n_repeats folds:
      1. split train into fit_df / eval_df (stratified on TARGET, so each fold keeps ~16% leavers)
      2. scores = make_scores(fit_df, eval_df)   <- eval rows must never reach fitting
      3. k = max(1, round(k_frac * len(eval_df)))
      4. record PR-AUC and precision@k on eval_df

    Returns one row per fold with columns:
      repeat, fold, n, n_leavers, k, pr_auc, precision_at_k
    Same seed -> same folds, so different models are compared on identical splits.
    """
    raise NotImplementedError("Rohit: implement me")
