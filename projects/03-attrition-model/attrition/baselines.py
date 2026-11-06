"""Baseline models in the shape `cross_validate` expects: (fit_df, eval_df) -> scores."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.cv import ScoreFn
from attrition.load import TARGET

# Fixed 2026-11-05 before any LightGBM score (TASKS.md 03-M3).
LGBM_DEFAULTS = {
    "num_leaves": 6,
    "min_child_samples": 40,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
    "random_state": 0,
    "verbose": -1,
}


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


def lightgbm_model(
    numeric: list[str], categorical: list[str], class_weight: str | None = None, **overrides
) -> ScoreFn:
    """LightGBM classifier on the named columns. Core logic: Rohit.

    Numeric columns are passed through, categorical columns are one-hot encoded, and both steps
    are fitted on fit_df only. Returns P(leaves) for each row of eval_df.
    class_weight is passed to LGBMClassifier ("balanced" or None); all other settings are
    LGBM_DEFAULTS. overrides replace LGBM_DEFAULTS values; unknown keys raise ValueError.
    """
    # reject unknown keys: LightGBM would silently ignore a typo
    unknown = set(overrides) - set(LGBM_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown LightGBM settings {sorted(unknown)}; valid: {sorted(LGBM_DEFAULTS)}"
        )

    param = {**LGBM_DEFAULTS, **overrides}

    def score(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        preprocess = ColumnTransformer(
            [
                ("num", "passthrough", numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ]
        )
        model = Pipeline(
            [
                ("preprocess", preprocess),
                ("clf", lgb.LGBMClassifier(**param, class_weight=class_weight)),
            ]
        )
        features = list(numeric) + list(categorical)

        model.fit(fit_df[features], fit_df[TARGET])
        return model.predict_proba(eval_df[features])[:, 1]

    return score


def tuned_lightgbm_model(
    numeric: list[str],
    categorical: list[str],
    grid: list[dict],
    inner_splits: int = 3,
    chosen: list[dict] | None = None,
) -> ScoreFn:
    """Picks settings from `grid` by inner CV on fit_df only, refits, scores eval_df.

    If `chosen` is given, the winning setting of every call is appended to it (one per fold).
    Core logic: Rohit.
    """

    if grid is None or len(grid) == 0:
        raise ValueError("Grid must be a non-empty list of parameter dictionaries")

    def score(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
        best_score = -np.inf
        best_params = None
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=0)
        splits = list(skf.split(fit_df, fit_df[TARGET]))
        for params in grid:
            inner_scores = []
            for train_idx, val_idx in splits:
                train_df = fit_df.iloc[train_idx]
                val_df = fit_df.iloc[val_idx]

                model = lightgbm_model(numeric, categorical, **params)
                val_scores = model(train_df, val_df.drop(columns=[TARGET]))
                inner_scores.append(average_precision_score(val_df[TARGET], val_scores))

            mean_inner_score = np.mean(inner_scores)
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = params

        if chosen is not None:
            chosen.append(best_params)
        # Refit on the entire fit_df with the best parameters
        final_model = lightgbm_model(numeric, categorical, **best_params)
        return final_model(fit_df, eval_df)

    return score
