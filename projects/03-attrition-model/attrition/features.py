"""Model inputs, from docs/column_audit.md. (Boilerplate: Claude.)"""

import pandas as pd

NUMERIC = [
    "DistanceFromHome",
    "Education",
    "JobLevel",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "JobRole", "OverTime"]

# Suspicious: survey timing unknown, so M2 runs with and without them (DECISIONS 2026-10-28).
SURVEY = [
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]


def add_overtime_and_level1(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with an int 0/1 column `overtime_and_level1`.

    1 when OverTime == "Yes" and JobLevel == 1, else 0. df itself is not changed.

    Core logic: Rohit.
    """
    if "OverTime" not in df.columns or "JobLevel" not in df.columns:
        raise KeyError("df must have OverTime and JobLevel columns")
    if not set(df["OverTime"].unique()).issubset({"Yes", "No"}):
        raise ValueError("OverTime must be 'Yes' or 'No'")
    if not pd.api.types.is_integer_dtype(df["JobLevel"]):
        raise ValueError("JobLevel must be an integer column")

    df_copy = df.copy()

    df_copy["overtime_and_level1"] = (
        (df_copy["OverTime"] == "Yes") & (df_copy["JobLevel"] == 1)
    ).astype(int)

    return df_copy
