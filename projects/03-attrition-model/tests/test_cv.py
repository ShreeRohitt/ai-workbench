import numpy as np
import pandas as pd
import pytest

from attrition.cv import cross_validate
from attrition.load import TARGET

COLUMNS = ["repeat", "fold", "n", "n_leavers", "k", "pr_auc", "precision_at_k"]


@pytest.fixture
def train():
    # 500 rows, 16% leavers, with an ID so tests can track which rows went where
    rng = np.random.default_rng(0)
    y = np.array([1] * 80 + [0] * 420)
    rng.shuffle(y)
    return pd.DataFrame({"EmployeeNumber": range(500), TARGET: y})


@pytest.fixture
def perfect(train):
    # cheats by looking up each person's true label by ID: should score 1.0 on everything
    answers = train.set_index("EmployeeNumber")[TARGET]

    def score(fit_df, eval_df):
        return answers.loc[eval_df.EmployeeNumber].to_numpy().astype(float)

    return score


def test_shape_and_columns(train, perfect):
    res = cross_validate(train, perfect, n_splits=5, n_repeats=3)
    assert len(res) == 15
    assert list(res.columns) == COLUMNS


def test_perfect_model_scores_one(train, perfect):
    res = cross_validate(train, perfect)
    assert (res["pr_auc"] == 1.0).all()
    assert (res["precision_at_k"] == 1.0).all()


def test_k_is_ten_percent_of_fold(train, perfect):
    res = cross_validate(train, perfect, n_splits=5)
    assert (res["n"] == 100).all()
    assert (res["k"] == 10).all()


def test_folds_are_stratified(train, perfect):
    res = cross_validate(train, perfect, n_splits=5)
    assert res["n_leavers"].between(15, 17).all()  # 16% of 100


def test_no_eval_row_reaches_fitting(train):
    # the leakage test: fit_df and eval_df must never share a person
    def spy(fit_df, eval_df):
        assert set(fit_df.EmployeeNumber).isdisjoint(eval_df.EmployeeNumber)
        assert len(fit_df) + len(eval_df) == len(train)
        return np.zeros(len(eval_df))

    cross_validate(train, spy)


def test_model_never_sees_eval_labels(train):
    def spy(fit_df, eval_df):
        assert TARGET in fit_df.columns  # the model needs labels to learn from
        assert TARGET not in eval_df.columns  # ...but must not see the answers it is graded on
        return np.zeros(len(eval_df))

    cross_validate(train, spy)


def test_each_row_evaluated_once_per_repeat(train):
    seen = []

    def spy(fit_df, eval_df):
        seen.extend(eval_df.EmployeeNumber)
        return np.zeros(len(eval_df))

    cross_validate(train, spy, n_splits=5, n_repeats=2)
    counts = pd.Series(seen).value_counts()
    assert len(counts) == 500 and (counts == 2).all()


def test_same_seed_same_folds(train):
    def ids(seed):
        seen = []

        def spy(fit_df, eval_df):
            seen.append(tuple(sorted(eval_df.EmployeeNumber)))
            return np.zeros(len(eval_df))

        cross_validate(train, spy, seed=seed)
        return seen

    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_random_model_near_base_rate(train):
    rng = np.random.default_rng(0)

    def random_scores(fit_df, eval_df):
        return rng.random(len(eval_df))

    res = cross_validate(train, random_scores, n_repeats=20)
    # averaged over 100 folds, random should land near the 16% base rate
    assert res["pr_auc"].mean() == pytest.approx(0.16, abs=0.04)
    assert res["precision_at_k"].mean() == pytest.approx(0.16, abs=0.04)
