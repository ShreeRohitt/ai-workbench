"""Baseline models in the shape `cross_validate` expects: (fit_df, eval_df) -> scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.cv import ScoreFn
from attrition.load import TARGET


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
    if "OverTime" not in eval_df.columns or "JobLevel" not in eval_df.columns:
        raise KeyError("eval_df must have OverTime and JobLevel columns")
    if not set(eval_df["OverTime"].unique()).issubset({"Yes", "No"}):
        raise ValueError("OverTime must be 'Yes' or 'No'")

    return np.where((eval_df["OverTime"] == "Yes") & (eval_df["JobLevel"] == 1.0), 1.0, 0.0)


def logistic_model(numeric: list[str], categorical: list[str]) -> ScoreFn:
    """Logistic regression on the named columns. Core logic: Rohit.

    Numeric columns are scaled, categorical columns are one-hot encoded, and both steps
    are fitted on fit_df only. Returns P(leaves) for each row of eval_df.
    """

    def score(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        preprocess = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        model = Pipeline(
            [
                ("preprocess", preprocess),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        features = list(numeric) + list(categorical)
        model.fit(fit_df[features], fit_df[TARGET])
        return model.predict_proba(eval_df[features])[:, 1]

    return score
