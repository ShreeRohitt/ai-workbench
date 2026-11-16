# LightGBM vs logistic regression on the same 25 folds: settles the M2 bet. TRAINING set only.
# In VS Code: click "Run Cell" above each `# %%` line (pick the .venv interpreter when asked).
# Setup, helper and gap cells: Claude (boilerplate). Conclusion: Rohit.

# %% Setup: per-fold results for both models, all on the same 25 folds
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from attrition.baselines import lightgbm_model, logistic_model
from attrition.cv import cross_validate
from attrition.features import CATEGORICAL, NUMERIC
from attrition.load import TARGET, load_raw
from attrition.split import make_split

train, test = make_split(load_raw())
del test  # locked away until the very end

lr = cross_validate(train, logistic_model(NUMERIC, CATEGORICAL))
lgbm = cross_validate(train, lightgbm_model(NUMERIC, CATEGORICAL))


# %% Helper: paired comparison across folds (same as notebooks/02_paired_comparison.py)
def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str, a_name: str, b_name: str) -> None:
    """Compare two models fold by fold. A positive diff means a beat b."""
    diff = a[metric] - b[metric]
    ties = (diff.abs() < 1e-9).sum()
    wins = (diff > 1e-9).sum()
    losses = len(diff) - wins - ties
    print(f"{a_name} vs {b_name} on {metric}")
    print(f"  wins {wins}, ties {ties}, losses {losses} of {len(diff)} folds")
    print(f"  mean {diff.mean():+.3f}, min {diff.min():+.3f}, max {diff.max():+.3f}")


# %% The bet: LightGBM vs LR
compare(lgbm, lr, "pr_auc", "LightGBM", "LR")
compare(lgbm, lr, "precision_at_k", "LightGBM", "LR")


# %% Over- or under-fitting? PR-AUC on the fit rows vs the eval rows, same 25 folds
def fit_vs_eval(make_scores, n_splits=5, n_repeats=5, seed=42) -> pd.DataFrame:
    """Score each fold's model on its own fit rows too. A big gap = memorising."""
    y = train[TARGET].to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for fit_idx, eval_idx in cv.split(np.zeros(len(train)), y):
        fit_df, eval_df = train.iloc[fit_idx], train.iloc[eval_idx]
        rows.append(
            {
                "fit": average_precision_score(
                    fit_df[TARGET], make_scores(fit_df, fit_df.drop(columns=[TARGET]))
                ),
                "eval": average_precision_score(
                    eval_df[TARGET], make_scores(fit_df, eval_df.drop(columns=[TARGET]))
                ),
            }
        )
    return pd.DataFrame(rows)


for name, model in [
    ("LR", logistic_model(NUMERIC, CATEGORICAL)),
    ("LightGBM", lightgbm_model(NUMERIC, CATEGORICAL)),
]:
    g = fit_vs_eval(model)
    print(
        f"{name:9s} PR-AUC fit {g['fit'].mean():.3f}  eval {g['eval'].mean():.3f}  "
        f"gap {(g['fit'] - g['eval']).mean():+.3f}"
    )

# %% Nested CV: settings tuned inside each fold's fit rows only (12-setting TUNING_GRID)
from collections import Counter  # noqa: E402

from attrition.baselines import tuned_lightgbm_model  # noqa: E402
from attrition.report import TUNING_GRID  # noqa: E402

chosen: list[dict] = []
tuned = cross_validate(
    train, tuned_lightgbm_model(NUMERIC, CATEGORICAL, TUNING_GRID, chosen=chosen)
)
compare(tuned, lgbm, "pr_auc", "tuned LightGBM", "LightGBM defaults")
compare(tuned, lr, "pr_auc", "tuned LightGBM", "LR")
compare(tuned, lr, "precision_at_k", "tuned LightGBM", "LR")
print("settings chosen across the 25 folds:")
for setting, n in Counter(tuple(sorted(c.items())) for c in chosen).most_common():
    print(f"  {n:2d}x {dict(setting)}")

# %% Class weights: balanced vs plain, same 25 folds (predictions in TASKS.md, written first)
lr_bal = cross_validate(train, logistic_model(NUMERIC, CATEGORICAL, "balanced"))
lgbm_bal = cross_validate(train, lightgbm_model(NUMERIC, CATEGORICAL, "balanced"))
for metric in ("pr_auc", "precision_at_k"):
    compare(lr_bal, lr, metric, "LR balanced", "LR")
    compare(lgbm_bal, lgbm, metric, "LightGBM balanced", "LightGBM")

# Effective sample size under balanced weights: (sum w)^2 / sum w^2
w = np.where(
    train[TARGET] == 1,
    len(train) / (2 * train[TARGET].sum()),
    len(train) / (2 * (1 - train[TARGET]).sum()),
)
print(f"rows {len(train)}, effective rows with balanced weights {w.sum() ** 2 / (w**2).sum():.0f}")

# %% Conclusion (Rohit's reasoning; written up by Claude)
# 1. The bet was wrong. Rohit bet LightGBM +0.20 PR-AUC over LR; measured +0.020 (18 wins,
#    7 losses of 25 folds), tuned +0.022 (19 / 6). A tenth of the bet: on ~150 leavers per fold
#    most interactions trees could find are noise, and LR already covers the main effects.
# 2. HR would not notice. Tuned LightGBM vs LR on precision@10%: +0.017 (10 wins, 9 ties,
#    6 losses) = 0.4 extra real leavers per 24 calls. Fold-to-fold swing is -0.167 to +0.167
#    (+-4 people of 24), ten times the gain.
# 3. The limit is the information, not the model. min_child_samples 40 -> 100 cut fit PR-AUC
#    0.812 -> 0.744 but eval stayed 0.554: the gap was memorised noise. LR has almost no gap
#    (+0.056) and similar eval. The one big gain so far came from new columns (survey, +0.101
#    in M2), not from a stronger model; more rows may also help but is unmeasured.
# 4. Tuning is not worth it here. Nested CV gained +0.002 over the defaults (12 wins, 3 ties,
#    10 losses); 7 of 12 settings won some fold, so the settings are near-equal and the pick is
#    inner-fold noise. Only signal: 4-leaf trees won 21 of 25 folds.
# Caveat: the 25 folds reuse the same 1,176 people, so win counts overstate certainty.
