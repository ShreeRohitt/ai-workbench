"""Spec tests for lr_contributions. (Tests: Claude. Implementation: Rohit.)"""

import numpy as np
import pandas as pd
import pytest
import shap

from attrition.baselines import logistic_pipeline
from attrition.explain import lr_contributions
from attrition.load import TARGET

NUMERIC = ["income", "years"]
CATEGORICAL = ["overtime", "role"]


@pytest.fixture
def data():
    # 300 noisy people; index labels deliberately not 0..n-1
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "income": rng.normal(6.5, 2.0, n),
            "years": rng.integers(0, 20, n).astype(float),
            "overtime": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "role": rng.choice(["Sales", "Research", "Manager"], n),
        },
        index=np.arange(1000, 1000 + n),
    )
    log_odds = -1.5 + 1.5 * (df.overtime == "Yes") - 0.3 * (df.income - 6.5) - 0.05 * df.years
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return df


@pytest.fixture
def model(data):
    return logistic_pipeline(NUMERIC, CATEGORICAL).fit(data[NUMERIC + CATEGORICAL], data[TARGET])


def X(df):
    return df[NUMERIC + CATEGORICAL]


def test_one_column_per_original_column_same_index(data, model):
    _, contrib = lr_contributions(model, X(data).iloc[:10], X(data))
    assert list(contrib.columns) == NUMERIC + CATEGORICAL
    assert list(contrib.index) == list(data.index[:10])


def test_base_plus_contributions_is_the_model_score(data, model):
    base, contrib = lr_contributions(model, X(data), X(data))
    np.testing.assert_allclose(base + contrib.sum(axis=1), model.decision_function(X(data)))


def test_average_value_contributes_zero(data, model):
    # someone whose income is exactly the background average: income pushes neither way
    person = X(data).iloc[[0]].copy()
    person["income"] = data["income"].mean()
    _, contrib = lr_contributions(model, person, X(data))
    assert contrib["income"].iloc[0] == pytest.approx(0.0, abs=1e-10)


def test_contributions_average_to_zero_over_background(data, model):
    _, contrib = lr_contributions(model, X(data), X(data))
    np.testing.assert_allclose(contrib.mean(), 0.0, atol=1e-10)


def test_base_is_the_score_of_the_average_row(data, model):
    base, _ = lr_contributions(model, X(data), X(data))
    assert base == pytest.approx(model.decision_function(X(data)).mean())


def test_matches_shap_linear_explainer(data, model):
    pre, clf = model.named_steps["preprocess"], model.named_steps["clf"]
    z_bg = np.asarray(pre.transform(X(data)), dtype=float)
    masker = shap.maskers.Independent(z_bg, max_samples=len(z_bg))
    values = shap.LinearExplainer(clf, masker).shap_values(z_bg)
    names = pre.get_feature_names_out()
    expected = pd.DataFrame(
        {
            col: values[:, [n == f"num__{col}" or n.startswith(f"cat__{col}_") for n in names]].sum(
                axis=1
            )
            for col in NUMERIC + CATEGORICAL
        },
        index=data.index,
    )
    _, contrib = lr_contributions(model, X(data), X(data))
    np.testing.assert_allclose(contrib.to_numpy(), expected.to_numpy(), atol=1e-8)


def test_group_contributions_sums_groups_and_keeps_the_rest(data, model):
    from attrition.explain import group_contributions

    _, contrib = lr_contributions(model, X(data), X(data))
    grouped = group_contributions(contrib, {"pay and time": ["income", "years"]})
    assert list(grouped.columns) == ["pay and time", "overtime", "role"]
    np.testing.assert_allclose(grouped["pay and time"], contrib["income"] + contrib["years"])
    np.testing.assert_allclose(grouped.sum(axis=1), contrib.sum(axis=1))
