import numpy as np
import pandas as pd
import pytest

from attrition.final_test import score_test
from attrition.load import TARGET


@pytest.fixture
def split():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 250)
    rng.shuffle(y)
    df = pd.DataFrame({"EmployeeNumber": range(300), "noise": rng.random(300), TARGET: y})
    return df.iloc[:200], df.iloc[200:]


def test_fit_sees_all_train_and_scoring_sees_only_test_without_labels(split):
    train, test = split
    calls = []

    def spy(fit_df, eval_df):
        calls.append((set(fit_df.EmployeeNumber), set(eval_df.EmployeeNumber), TARGET in eval_df))
        return np.zeros(len(eval_df))

    score_test(train, test, spy, [10])

    assert len(calls) == 1  # one fit, one scoring pass
    fit_ids, eval_ids, eval_has_label = calls[0]
    assert fit_ids == set(train.EmployeeNumber)
    assert eval_ids == set(test.EmployeeNumber)
    assert not eval_has_label


def test_perfect_scores_give_perfect_metrics(split):
    train, test = split
    n_leavers = int(test[TARGET].sum())
    answers = test.set_index("EmployeeNumber")[TARGET]

    def perfect(fit_df, eval_df):
        return answers.loc[eval_df.EmployeeNumber].to_numpy().astype(float)

    out = score_test(train, test, perfect, [5, n_leavers])
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["n_test"] == 100 and out["n_test_leavers"] == n_leavers
    by_k = {r["k"]: r for r in out["capacity"]}
    assert by_k[5]["precision"] == 1.0 and by_k[5]["leavers_found"] == 5
    assert by_k[n_leavers]["recall"] == pytest.approx(1.0)
    assert by_k[5]["random_finds"] == pytest.approx(n_leavers / 100 * 5)
