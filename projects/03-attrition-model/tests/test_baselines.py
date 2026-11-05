import numpy as np
import pandas as pd

from attrition.baselines import logistic_model, overtime_level1_rule, random_model


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


def staff():
    # 40 people: leavers are the overtime Sales staff on low pay
    n = 40
    return pd.DataFrame(
        {
            "MonthlyIncome": [2000 + 100 * i for i in range(n)],
            "OverTime": ["Yes" if i < 20 else "No" for i in range(n)],
            "Department": ["Sales" if i % 2 == 0 else "R&D" for i in range(n)],
            "Attrition": [1 if i < 20 else 0 for i in range(n)],
        }
    )


MODEL = logistic_model(["MonthlyIncome"], ["OverTime", "Department"])


def test_lr_returns_a_probability_per_eval_row():
    fit, new = staff(), staff().drop(columns=["Attrition"]).iloc[:7]
    scores = MODEL(fit, new)
    assert isinstance(scores, np.ndarray) and scores.shape == (7,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_lr_learns_the_signal():
    fit, new = staff(), staff().drop(columns=["Attrition"])
    scores = MODEL(fit, new)
    assert scores[:20].mean() > scores[20:].mean()


def test_lr_handles_category_never_seen_in_fit():
    fit, new = staff(), staff().drop(columns=["Attrition"]).iloc[:3].copy()
    new["Department"] = "Legal"
    assert np.isfinite(MODEL(fit, new)).all()


def test_lr_uses_only_the_named_columns():
    fit, new = staff(), staff().drop(columns=["Attrition"])
    a = MODEL(fit, new)
    b = MODEL(fit.assign(Gender="Female"), new.assign(Gender="Male"))
    assert np.array_equal(a, b)


def test_lr_preprocessing_is_not_fitted_on_eval_rows():
    # the leakage test: an extreme eval salary must not change anyone else's score
    fit, new = staff(), staff().drop(columns=["Attrition"]).iloc[:5].copy()
    a = MODEL(fit, new)
    new.loc[new.index[-1], "MonthlyIncome"] = 10_000_000
    b = MODEL(fit, new)
    assert np.array_equal(a[:4], b[:4])


def test_lr_score_does_not_depend_on_other_eval_rows():
    # staff 0-6 are all OverTime=Yes: on their own, an encoder fitted per frame loses that column
    fit, everyone = staff(), staff().drop(columns=["Attrition"])
    in_full_frame = MODEL(fit.copy(), everyone.copy())[:7]
    alone = MODEL(fit.copy(), everyone.iloc[:7].copy())
    np.testing.assert_allclose(alone, in_full_frame)
