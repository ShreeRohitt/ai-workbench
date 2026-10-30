"""Cross-validation harness: every model is scored on the same folds. Core logic: Rohit."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from attrition.load import TARGET
from attrition.metrics import precision_at_k

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
    if TARGET not in train.columns:
        raise KeyError(f"train has no {TARGET!r} column")
    if not 0 < k_frac <= 1:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")

    y = train[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    rows = []
    # split() yields all folds of repeat 0, then repeat 1, ... so i maps cleanly to (repeat, fold).
    for i, (fit_idx, eval_idx) in enumerate(cv.split(np.zeros(len(train)), y)):
        repeat, fold = divmod(i, n_splits)

        # Defensive: .iloc[array] already returns a new frame (and pandas 3 is Copy-on-Write),
        # but an explicit copy makes "each fold owns its data" obvious and version-proof.
        fit_df = train.iloc[fit_idx].copy()
        eval_df = train.iloc[eval_idx].copy()
        y_eval = eval_df[TARGET].to_numpy()

        # The model never sees eval labels: hand it eval features only.
        scores = np.asarray(make_scores(fit_df, eval_df.drop(columns=[TARGET])), dtype=float)

        if scores.shape != (len(eval_df),):
            raise ValueError(
                f"repeat {repeat} fold {fold}: make_scores returned shape {scores.shape}, "
                f"expected ({len(eval_df)},)"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"repeat {repeat} fold {fold}: make_scores returned NaN/inf scores")

        k = max(1, int(round(k_frac * len(eval_df))))

        rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n": len(eval_df),
                "n_leavers": int(y_eval.sum()),
                "k": k,
                "pr_auc": float(average_precision_score(y_eval, scores)),
                "precision_at_k": precision_at_k(y_eval, scores, k),
            }
        )

    return pd.DataFrame(
        rows, columns=["repeat", "fold", "n", "n_leavers", "k", "pr_auc", "precision_at_k"]
    )
