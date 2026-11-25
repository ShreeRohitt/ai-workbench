# SHAP cross-check: does LightGBM agree with plain LR on what drives risk? TRAINING set only.
# In VS Code: click "Run Cell" above each `# %%` line (pick the .venv interpreter when asked).
# Setup and table cells: Claude (boilerplate). Interpretation: Rohit.

# %% Setup: both models fit on all of train, explained on train (log-odds scale)
import pandas as pd

from attrition.baselines import lightgbm_pipeline, logistic_pipeline
from attrition.explain import group_contributions, lr_contributions, tree_contributions
from attrition.features import CATEGORICAL, GROUPS, NUMERIC
from attrition.load import TARGET, load_raw
from attrition.split import make_split

train, test = make_split(load_raw())
del test  # locked away until M5
X, y = train[NUMERIC + CATEGORICAL], train[TARGET]

lr = logistic_pipeline(NUMERIC, CATEGORICAL).fit(X, y)
lgbm = lightgbm_pipeline(NUMERIC, CATEGORICAL).fit(X, y)
_, lr_c = lr_contributions(lr, X, X)
_, lgbm_c = tree_contributions(lgbm, X)
lr_g, lgbm_g = group_contributions(lr_c, GROUPS), group_contributions(lgbm_c, GROUPS)

# %% 1. Same top drivers? Mean |contribution| and rank, grouped
ranking = pd.DataFrame(
    {
        "LR": lr_g.abs().mean(),
        "LightGBM": lgbm_g.abs().mean(),
    }
)
ranking["LR rank"] = ranking["LR"].rank(ascending=False).astype(int)
ranking["LightGBM rank"] = ranking["LightGBM"].rank(ascending=False).astype(int)
ranking.sort_values("LR", ascending=False).round(3)

# %% 2. Shape: average StockOptionLevel push per level (LR must draw a straight line)
pd.DataFrame(
    {
        "n": train.groupby("StockOptionLevel").size(),
        "leave rate": y.groupby(train["StockOptionLevel"]).mean(),
        "LR push": lr_c["StockOptionLevel"].groupby(train["StockOptionLevel"]).mean(),
        "LightGBM push": lgbm_c["StockOptionLevel"].groupby(train["StockOptionLevel"]).mean(),
    }
).round(3)

# %% 3. Interaction: is overtime worse for JobLevel 1? (LR's overtime push can't depend on level)
level1 = (train["JobLevel"] == 1).map({True: "JobLevel 1", False: "JobLevel 2+"})
overtime = train["OverTime"]
pd.DataFrame(
    {
        "n": train.groupby([overtime, level1]).size(),
        "leave rate": y.groupby([overtime, level1]).mean(),
        "LR OverTime push": lr_c["OverTime"].groupby([overtime, level1]).mean(),
        "LightGBM OverTime push": lgbm_c["OverTime"].groupby([overtime, level1]).mean(),
    }
).round(3)

# %% Interpretation (Rohit)
