import numpy as np
import pandas as pd

from attrition.baselines import overtime_level1_rule, random_model


def people():
    # index labels deliberately not 0..n-1, like a CV fold
    return pd.DataFrame(
        {"OverTime": ["Yes", "Yes", "No", "No", "Yes"], "JobLevel": [1, 2, 1, 3, 1]},
        index=[40, 7, 13, 2, 99],
    )


def test_rule_scores_group_members_one():
    scores = overtime_level1_rule(pd.DataFrame(), people())
    assert list(scores) == [1.0, 0.0, 0.0, 0.0, 1.0]


def test_rule_returns_float_numpy_array():
    scores = overtime_level1_rule(pd.DataFrame(), people())
    assert isinstance(scores, np.ndarray)
    assert scores.dtype == float
    assert scores.shape == (5,)


def test_rule_ignores_fit_df():
    a = overtime_level1_rule(pd.DataFrame(), people())
    b = overtime_level1_rule(people().assign(OverTime="No"), people())
    assert list(a) == list(b)


def test_random_model_is_reproducible():
    a = random_model(seed=1)(pd.DataFrame(), people())
    b = random_model(seed=1)(pd.DataFrame(), people())
    assert list(a) == list(b) and len(a) == 5
