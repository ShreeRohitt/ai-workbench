import numpy as np
import pandas as pd
import pytest

from attrition.load import TARGET
from attrition.report import compare, format_table


@pytest.fixture
def train():
    # 500 rows, 16% leavers, with an ID so a model can look up the true label
    rng = np.random.default_rng(0)
    y = np.array([1] * 80 + [0] * 420)
    rng.shuffle(y)
    return pd.DataFrame({"EmployeeNumber": range(500), TARGET: y})


@pytest.fixture
def perfect(train):
    answers = train.set_index("EmployeeNumber")[TARGET]

    def score(fit_df, eval_df):
        return answers.loc[eval_df.EmployeeNumber].to_numpy().astype(float)

    return score


def test_compare_gives_one_row_per_model_in_order(train, perfect):
    summary = compare(train, {"b": perfect, "a": perfect})
    assert list(summary["model"]) == ["b", "a"]


def test_compare_summarises_all_folds(train, perfect):
    row = compare(train, {"perfect": perfect}).iloc[0]
    assert row["folds"] == 25
    assert row["pr_auc_mean"] == pytest.approx(1.0)
    assert row["pr_auc_std"] == pytest.approx(0.0)
    assert row["precision_at_k_min"] == pytest.approx(1.0)


def test_compare_gives_every_model_identical_folds(train):
    seen = []

    def spy(fit_df, eval_df):
        seen.append(tuple(eval_df.EmployeeNumber))
        return np.zeros(len(eval_df))

    compare(train, {"first": spy, "second": spy})
    assert seen[:25] == seen[25:]


def test_format_table_shows_mean_std_and_range(train, perfect):
    text = format_table(compare(train, {"perfect": perfect}))
    assert text.splitlines()[0].split() == ["model", "PR-AUC", "precision@10%"]
    assert "perfect" in text
    assert "1.000 ± 0.000 (1.000 to 1.000)" in text
