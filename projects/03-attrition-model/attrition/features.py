"""Model inputs, from docs/column_audit.md. (Boilerplate: Claude.)"""

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
