"""Why does the model score someone high? Per-column contributions (SHAP for a linear model).

Core logic: written by Claude at Rohit's request (2026-11-21), walked through in session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _dense(matrix) -> np.ndarray:
    """ColumnTransformer may return a sparse matrix; contributions need a plain array."""
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=float)


def lr_contributions(
    model: Pipeline, X: pd.DataFrame, background: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """Each original column's push on each row's log-odds score, relative to the average row.

    model:      a fitted `logistic_pipeline` (steps "preprocess" -> "clf")
    X:          rows to explain, raw columns (as passed to model.fit)
    background: rows that define "average", normally the rows the model was fitted on

    For transformed column j:  contribution = coef_j * (z_j - mean of z_j over background)
    One-hot columns are summed back to their original column (all JobRole_* -> JobRole).

    Returns (base, contributions):
      base:          log-odds score of the average background row
      contributions: index = X.index, one column per original column, in the order the
                     model was given them (numeric first, then categorical)
    So that:  base + contributions.sum(axis=1) == model.decision_function(X)
    """
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    z = _dense(pre.transform(X))  # rows x transformed columns
    average = _dense(pre.transform(background)).mean(axis=0)
    coef = clf.coef_[0]

    per_column = coef * (z - average)  # broadcasting: every row, every transformed column
    base = float(clf.intercept_[0] + coef @ average)

    # Sum one-hot columns back to their original column: cat__JobRole_Sales Executive -> JobRole
    names = pre.get_feature_names_out()
    originals = [col for _, _, cols in pre.transformers_ if _ != "remainder" for col in cols]
    contributions = pd.DataFrame(
        {
            col: per_column[
                :, [n == f"num__{col}" or n.startswith(f"cat__{col}_") for n in names]
            ].sum(axis=1)
            for col in originals
        },
        index=X.index,
    )
    return base, contributions
