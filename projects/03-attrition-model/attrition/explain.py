"""Why does the model score someone high? Per-column contributions (SHAP for a linear model).

Core logic: Rohit.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline


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
    raise NotImplementedError("Rohit: Step 3 of 03-M4")
