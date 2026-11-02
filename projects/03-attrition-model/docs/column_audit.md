# Column audit (03-M1)

Computed on the **training set** (1,176 rows). Fill in the last two columns for every row.

**Category** — pick one:
- `use` — known *before* someone leaves; fair input for the model
- `drop: id/constant` — an identifier, or the same value for everyone
- `suspicious` — might be measured *after* (or because of) the decision to leave, or we can't tell when it was measured
- `protected: fairness only` — known before leaving, but kept out of the model; used only for fairness checks (M5)

**Reason** — one short sentence. "Seems fine" is not a reason.

| Column | Type | Distinct values | Examples | Category | Reason |
|---|---|---:|---|---|---|
| Age | int64 | 43 | 47, 22, 46 | protected: fairness only | known before leaving; kept out of the model, baseline for M5 fairness checks (proxied by TotalWorkingYears, ρ = 0.67) |
| Attrition | int64 | 2 | 0, 1 | target | What we predict (1 = left). |
| BusinessTravel | str | 3 | Travel_Rarely, Travel_Frequently, Non-Travel | use | fair input w get before leaving |
| DailyRate | int64 | 787 | 1225, 594, 406 | drop: noise | correlation with MonthlyIncome = 0.02; not pay, looks like synthetic noise |
| Department | str | 3 | Sales, Research & Development, Human Resources |  use| fair input to categorize on dept |
| DistanceFromHome | int64 | 29 | 2, 3, 13 | use | we can calculate if people far away from work leave |
| Education | int64 | 5 | 4, 1, 5 | use | can be used as a metric to see if highly educated peopel leave |
| EducationField | str | 6 | Life Sciences, Technical Degree, Marketing |  use| can be used to see which field tends to leave |
| EmployeeCount | int64 | 1 | 1 | drop: constant | unncessary |
| EmployeeNumber | int64 | 1176 | 1676, 169, 1124 | drop:id | unique to everyone   |
| EnvironmentSatisfaction | int64 | 4 | 2, 3, 1 | suspicious | can be used as a metric but don't know wwhen they've taken |
| Gender | str | 2 | Female, Male | protected: fairness only | known before leaving; kept out of the model, baseline for M5 fairness checks |
| HourlyRate | int64 | 71 | 47, 100, 52 | drop: noise | correlation with MonthlyIncome = 0.00; not pay, looks like synthetic noise |
| JobInvolvement | int64 | 4 | 4, 3, 1 | suspicious| unknown timing |
| JobLevel | int64 | 5 | 4, 1, 2 |  use| to see which job level has attrition |
| JobRole | str | 9 | Manager, Laboratory Technician, Sales Representative | use |  can be used to calculate|
| JobSatisfaction | int64 | 4 | 2, 4, 3 | suspicious | unknown timing|
| MaritalStatus | str | 3 | Divorced, Married, Single | protected: fairness only | known before leaving; kept out of the model, baseline for M5 fairness checks (proxied by StockOptionLevel: all Single = 0) |
| MonthlyIncome | int64 | 1098 | 15972, 2523, 17465 | use | used as a metric |
| MonthlyRate | int64 | 1153 | 21086, 19299, 15596 | drop: noise | correlation with MonthlyIncome = 0.04; not pay, looks like synthetic noise |
| NumCompaniesWorked | int64 | 10 | 6, 0, 3 | use | can be used to see if they're jumpers |
| Over18 | str | 1 | Y | drop |  everyone would be above 18|
| OverTime | str | 2 | No, Yes | use | good metric |
| PercentSalaryHike | int64 | 15 | 14, 12, 11 | use | set by the company at review, not an employee answer (assumed; dataset does not document timing) |
| PerformanceRating | int64 | 2 | 3, 4 | use | set by the company at review, not an employee answer (assumed; dataset does not document timing) |
| RelationshipSatisfaction | int64 | 4 | 3, 4, 2 | suspicious | unknown timing|
| StandardHours | int64 | 1 | 80 | drop | 80 for all |
| StockOptionLevel | int64 | 4 | 3, 1, 0 | use | known before |
| TotalWorkingYears | int64 | 40 | 29, 3, 23 | use | to see experience has any correlation |
| TrainingTimesLastYear | int64 | 7 | 2, 3, 1 | use  | good metric |
| WorkLifeBalance | int64 | 4 | 3, 2, 4 | suspicious | unknown timing |
| YearsAtCompany | int64 | 36 | 3, 2, 12 | use | to see if less exp people of the company leave |
| YearsInCurrentRole | int64 | 18 | 2, 1, 9 | use | same as above |
| YearsSinceLastPromotion | int64 | 16 | 1, 2, 4 | use | to see people who get promotion and leave |
| YearsWithCurrManager | int64 | 18 | 2, 1, 9 |  use|  too see if manager's affect|
