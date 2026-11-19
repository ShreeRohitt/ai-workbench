# Pandas practice on the TRAINING set only.
# In VS Code: click "Run Cell" above each `# %%` line (pick the .venv interpreter when asked).

# %% Setup
import pandas as pd  # noqa: F401  (you will need it in Exercise 4)

from attrition.load import load_raw
from attrition.split import make_split

df = load_raw()
train, test = make_split(df)
del test  # locked away until the very end

# %% Exercise 1: how many employees in each JobRole? Which role is most common?
train["JobRole"].value_counts()
#sales executive    273
# %% Exercise 2: attrition rate by MaritalStatus. Include group sizes.
train.groupby("MaritalStatus")[["Attrition"]].agg(["mean", "count"])
#single employees are more likely to leave the company than married or divorced employees.
# %% Exercise 3: attrition rate by JobSatisfaction (1 = low, 4 = high). Is there a pattern?
train.groupby("JobSatisfaction")[["Attrition"]].agg(["mean", "count"])
#maybe a pattern, but not a strong one. Employees with low job satisfaction are more likely to leave the company than those with high job satisfaction.
# %% Exercise 4: split MonthlyIncome into 4 bands with pd.cut, then attrition rate per band.
train["MonthlyIncome"].describe()
monthly_income_band = pd.qcut(train["MonthlyIncome"], q=4, labels=["Low", "Medium", "High", "VeryHigh"])
train.groupby(monthly_income_band)[["Attrition"]].agg(["mean", "count"])
# Hint: look at train["MonthlyIncome"].describe() first to choose sensible edges.
#people wuth low monthly income are more likely to leave the company than those with high monthly income, low has 29% attrition rate, medium has 13% attrition rate, high has 10% attrition rate, and very high has 10% attrition rate. Low is more than double the attrition rate of other bands, and medium is slightly higher than high and very high.
# %%
train[["DailyRate", "HourlyRate", "MonthlyRate", "MonthlyIncome"]].corr()
# %%
