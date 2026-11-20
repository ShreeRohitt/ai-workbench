import numpy as np
import pandas as pd
import pytest

from attrition.capacity import at_k_by_fold, summarise
from attrition.cv import cross_validate
from attrition.load import TARGET


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    y = np.array([1] * 80 + [0] * 420)
    rng.shuffle(y)
    return pd.DataFrame({"EmployeeNumber": range(500), "noise": rng.random(500), TARGET: y})


@pytest.fixture
def perfect(train):
    answers = train.set_index("EmployeeNumber")[TARGET]

    def score(fit_df, eval_df):
        return answers.loc[eval_df.EmployeeNumber].to_numpy().astype(float)

    return score


def noisy(fit_df, eval_df):
    return eval_df["noise"].to_numpy()


def test_matches_cross_validate_at_same_k(train):
    # every fold has 100 eval rows, so cross_validate's k = 10% = 10
    by_fold = at_k_by_fold(train, noisy, ks=[10])
    res = cross_validate(train, noisy)
    np.testing.assert_allclose(by_fold["precision"].to_numpy(), res["precision_at_k"].to_numpy())


def test_perfect_model_recall(train, perfect):
    # 16 leavers per fold: perfect recall is k/16 until k reaches 16, then 1.0
    by_fold = at_k_by_fold(train, perfect, ks=[8, 16, 30])
    s = summarise(by_fold).set_index("k")
    assert s.loc[8, "recall_mean"] == pytest.approx(0.5)
    assert s.loc[16, "recall_mean"] == pytest.approx(1.0)
    assert s.loc[30, "recall_mean"] == pytest.approx(1.0)
    assert s.loc[30, "precision_mean"] == pytest.approx(16 / 30)


def test_recall_rises_with_k(train):
    s = summarise(at_k_by_fold(train, noisy, ks=[10, 20, 40]))
    assert s["recall_mean"].is_monotonic_increasing
