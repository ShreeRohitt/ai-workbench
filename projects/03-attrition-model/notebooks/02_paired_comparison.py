# Paired comparison: is a model's gain real, or fold-to-fold noise? TRAINING set only.
# In VS Code: click "Run Cell" above each `# %%` line (pick the .venv interpreter when asked).
# Setup cell: Claude (boilerplate). Comparison cells: Rohit.

# %% Setup: per-fold results for each model, all on the same 25 folds
import pandas as pd

from attrition.baselines import logistic_model, overtime_level1_rule
from attrition.cv import cross_validate
from attrition.features import CATEGORICAL, NUMERIC, SURVEY
from attrition.load import load_raw
from attrition.split import make_split

train, test = make_split(load_raw())
del test  # locked away until the very end

rule = cross_validate(train, overtime_level1_rule)
lr = cross_validate(train, logistic_model(NUMERIC, CATEGORICAL))
lr_survey = cross_validate(train, logistic_model(NUMERIC + SURVEY, CATEGORICAL))
# One row per fold; row i is the same fold in all three frames.
pd.DataFrame(
    {"rule": rule.precision_at_k, "lr": lr.precision_at_k, "lr_survey": lr_survey.precision_at_k}
).head()


# %% Helper: paired comparison across folds
def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str, a_name: str, b_name: str) -> None:
    """Compare two models fold by fold. A positive diff means a beat b."""
    diff = a[metric] - b[metric]
    ties = (diff.abs() < 1e-9).sum()
    wins = (diff > 1e-9).sum()
    losses = len(diff) - wins - ties
    print(f"{a_name} vs {b_name} on {metric}")
    print(f"  wins {wins}, ties {ties}, losses {losses} of {len(diff)} folds")
    print(f"  mean {diff.mean():+.3f}, min {diff.min():+.3f}, max {diff.max():+.3f}")


# %% Exercise 1: logistic regression vs rule, on precision_at_k
compare(lr, rule, "precision_at_k", "LR", "rule")

# %% Exercise 2: logistic regression + survey vs plain logistic regression
compare(lr_survey, lr, "precision_at_k", "LR+survey", "LR")

# %% Exercise 3: repeat 1 and 2 on pr_auc. Do both metrics tell the same story?
compare(lr, rule, "pr_auc", "LR", "rule")
compare(lr_survey, lr, "pr_auc", "LR+survey", "LR")
# %% Conclusion (Rohit's reasoning; written up by Claude)
# 1. LR beats the rule on precision@10%: 20 wins, 3 ties, 2 losses of 25 folds, mean +0.115
#    (range -0.125 to +0.417). That is ~3 more real leavers per 24 calls, +24% over the rule.
# 2. PR-AUC overstates that gap (+0.255, 25/25): the rule gives only two scores, so it can't
#    rank within its flagged group, and PR-AUC scores the whole list. HR acts on the top 24
#    only, so precision@10% is the fair comparison.
# 3. The survey gain is not noise: 22 wins, 2 ties, 1 loss, mean +0.113 (range -0.042 to
#    +0.250); PR-AUC agrees (24/25, +0.101). But a leak would also win on nearly every fold,
#    so "not noise" does not mean "safe": the survey's timing is unknown.
# 4. Decision: deploy LR without survey (DECISIONS 2026-11-02). Caveat: the 25 folds reuse the
#    same 1,176 people, so they are not independent and these counts overstate certainty.

# %% Sensitivity (03-M3): LR without PercentSalaryHike + PerformanceRating vs plain LR. Claude.
from attrition.report import NO_HIKE_RATING  # noqa: E402

lr_no_hike = cross_validate(train, logistic_model(NO_HIKE_RATING, CATEGORICAL))
for metric in ["pr_auc", "precision_at_k"]:
    compare(lr_no_hike, lr, metric, "LR without hike + rating", "LR")
    print(f"  std of per-fold diff {(lr_no_hike[metric] - lr[metric]).std():.3f}")
