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

# %% Conclusion (Rohit): was the +20 bet right? What do the fit/eval gaps say about the settings?
