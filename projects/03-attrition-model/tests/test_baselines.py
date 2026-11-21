import numpy as np
import pandas as pd
import pytest

import attrition.baselines as baselines
from attrition.baselines import (
    lightgbm_model,
    logistic_model,
    overtime_level1_rule,
    random_model,
    tuned_lightgbm_model,
)


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


# LightGBM tests (Claude, boilerplate). staff() is too small: min_child_samples=40 can't split
# 40 rows, so these use 200 people. Labels overlap on purpose (50% of overtime staff leave, 13%
# of the rest; 22.5% overall): with perfectly separable labels class weights have nothing to move.
def big_staff():
    n = 200
    return pd.DataFrame(
        {
            "MonthlyIncome": [2000 + 50 * i for i in range(n)],
            "OverTime": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
            "Department": ["Sales" if i % 2 == 0 else "R&D" for i in range(n)],
            "Attrition": [int(i % 8 == 0 or i % 10 == 1) for i in range(n)],
        },
        index=range(1000, 1000 + n),  # not 0..n-1, like a CV fold
    )


LGBM = lightgbm_model(["MonthlyIncome"], ["OverTime", "Department"])


def eval_rows(df):
    return df.drop(columns=["Attrition"])


def test_lgbm_returns_a_probability_per_eval_row():
    scores = LGBM(big_staff(), eval_rows(big_staff()).iloc[:7])
    assert isinstance(scores, np.ndarray) and scores.shape == (7,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_lgbm_learns_the_signal():
    df = big_staff()
    scores = LGBM(df, eval_rows(df))
    leaver = df["Attrition"].to_numpy() == 1
    assert scores[leaver].mean() > scores[~leaver].mean()


def test_lgbm_is_deterministic():
    df = big_staff()
    np.testing.assert_array_equal(LGBM(df, eval_rows(df)), LGBM(df, eval_rows(df)))


def test_lgbm_handles_category_never_seen_in_fit():
    new = eval_rows(big_staff()).iloc[:3].copy()
    new["Department"] = "Legal"
    assert np.isfinite(LGBM(big_staff(), new)).all()


def test_lgbm_uses_only_the_named_columns():
    df = big_staff()
    a = LGBM(df, eval_rows(df))
    b = LGBM(df.assign(Gender="Female"), eval_rows(df).assign(Gender="Male"))
    np.testing.assert_array_equal(a, b)


def test_lgbm_score_does_not_depend_on_other_eval_rows():
    df = big_staff()
    in_full_frame = LGBM(df, eval_rows(df))[:7]
    alone = LGBM(df, eval_rows(df).iloc[:7])
    np.testing.assert_allclose(alone, in_full_frame)


def test_lgbm_balanced_class_weight_reaches_the_classifier():
    # balanced weights make 20% leavers count as 50%, so average P(leaves) must rise
    df = big_staff()
    plain = LGBM(df, eval_rows(df))
    balanced = lightgbm_model(["MonthlyIncome"], ["OverTime", "Department"], "balanced")(
        df, eval_rows(df)
    )
    assert balanced.mean() > plain.mean() + 0.1


def test_lgbm_rejects_unknown_settings():
    # overrides are checked: a typo must fail loudly, not be silently ignored by LightGBM
    with pytest.raises(ValueError):
        lightgbm_model(["MonthlyIncome"], ["OverTime"], num_leafs=2)


# Nested-CV tests (Claude, boilerplate).
GRID = [{"min_child_samples": 40}, {"min_child_samples": 80}]


def test_tuning_never_sees_eval_rows(monkeypatch):
    # spy on every lightgbm_model call: only the final refit may touch eval_df (features only)
    calls = []
    real = baselines.lightgbm_model

    def spy(numeric, categorical, **kw):
        inner = real(numeric, categorical, **kw)

        def score(fit_df, eval_df):
            calls.append((set(fit_df.index), set(eval_df.index), "Attrition" in eval_df.columns))
            return inner(fit_df, eval_df)

        return score

    monkeypatch.setattr(baselines, "lightgbm_model", spy)
    df = big_staff()
    fit, held_out = df.iloc[:150], eval_rows(df.iloc[150:])
    tuned_lightgbm_model(["MonthlyIncome"], ["OverTime", "Department"], GRID)(fit, held_out)

    *tuning, final = calls
    assert len(tuning) == len(GRID) * 3
    for fit_idx, eval_idx, has_label in tuning:
        assert not (fit_idx | eval_idx) & set(held_out.index)
        assert not fit_idx & eval_idx
        assert not has_label
    assert final[0] == set(fit.index) and final[1] == set(held_out.index)


def test_tuned_records_one_grid_setting_per_call():
    df, chosen = big_staff(), []
    model = tuned_lightgbm_model(["MonthlyIncome"], ["OverTime", "Department"], GRID, chosen=chosen)
    scores = model(df, eval_rows(df).iloc[:7])
    model(df, eval_rows(df).iloc[:7])
    assert scores.shape == (7,) and len(chosen) == 2 and chosen[0] in GRID


def test_tuned_rejects_empty_grid():
    with pytest.raises(ValueError):
        tuned_lightgbm_model(["MonthlyIncome"], ["OverTime"], [])
