# Three recommendations for the CHRO

Data: IBM HR Analytics (fictional, created by IBM), **training split only**: 1,176 employees, 16.1% left.
Leave rates are over the dataset's label window, which is undocumented and probably a year or more, **not per month**.
Model evidence is from plain logistic regression (the chosen model) and a LightGBM cross-check, explained with SHAP (`attrition.importance`, `notebooks/04_shap.py`).

Written by Claude at Rohit's request (2026-11-23), from Rohit's drafts and review answers in session: overtime + Level 1 targeting (Rohit's draft), the Single/stock confound (Rohit), the stock step pattern (Rohit's reading of LightGBM).

**Read every finding as "linked to leaving", not "causes leaving".** The model shows which employees look like leavers. It can't show that changing one input would keep them. So each recommendation comes with a randomised pilot.

---

## 1. Cap overtime for junior staff first

- **Finding:** employees on overtime leave at **28.5% (n = 340) vs 11.1% without overtime (n = 836)**. Among Level 1 staff on overtime it is **51.7% (n = 116)**, vs 17.1% for Level 1 without overtime (n = 304).
  OverTime is the #1 driver in logistic regression and #2 in LightGBM, and it was in the top 5 in all 25 cross-validation folds.
- **Lever:** an overtime cap (e.g. hours per month), extra headcount in the teams with the most overtime, or comp-off days. HR and line managers control these.
- **Who:** Level 1 employees who do overtime come first. Job level isn't a lever, but it shows where overtime goes with the highest leave rate.
- **How to test:** randomly split Level 1 overtime staff into two halves. One half gets the cap for 6 months and the other carries on as now. Compare their leave rates.
  Size limit: about 145 such people company-wide means about 72 per half. A pilot that size reliably detects (80% power, 5% significance) only a drop of about **23 points** or more. A smaller real effect could show up as "no result".
- **Caveat:** overtime may be a symptom, not a cause. It could reflect understaffed teams, or junior roles that are both overtime-heavy and low-paid. If so, a cap alone won't keep people, and the pilot will show that.

## 2. Support employees in their first two years of work

- **Finding:** employees with **≤ 2 years of total work experience leave at 51.0% (n = 96) vs 13.1% (n = 1,080)** for everyone else.
  Career stage (TotalWorkingYears, JobLevel and MonthlyIncome, read as one group because they move together) is #2 in logistic regression and #1 in LightGBM, and was in the top 5 in all 25 folds.
  This group also covers much of the Sales Representative signal (41.7% leave, n = 60). Sales Reps are mostly junior and low-paid, which is why JobRole falls from #3 to #7 once LightGBM can use career stage directly.
- **Lever:** structured onboarding, a buddy or mentor for year one, and a visible path to the next level (e.g. Sales Representative → Sales Executive) with a pay step attached.
- **Who:** new joiners with ≤ 2 years of total experience, starting with the highest-scored on the model's list.
- **How to test:** randomly assign new early-career joiners to the programme or to normal onboarding, and compare leave rates after 6–12 months.
  Size limit: about 120 such people company-wide means about 60 per group, so only a drop of about **26 points** or more is reliably detectable. Running it over two or more hiring cohorts would help.
- **Caveat:** early-career people may leave mainly for reasons a programme can't touch (further study, better outside offers). Career stage also can't be changed, so this recommendation is about **targeting** support, not changing the input.

## 3. Move people from no stock options to some, not "more stock for everyone"

- **Finding:** employees with **no stock options leave at 24.3% (n = 503) vs 10.1% with any (n = 673)**.
  StockOptionLevel was in the top 5 in 24 of 25 folds. The shape matters: LightGBM gives level 0 a push of +0.51 and levels 1, 2 and 3 about the same push (−0.38, −0.39, −0.37), so the difference is **none vs any**. Logistic regression draws a straight line through the levels and wrongly makes level 3 look safest. Level 3 actually leaves at 15.2% (n = 66), which is uncertain at that size.
- **Lever:** a first stock grant (level 0 → 1). Compensation policy, owned by HR.
- **Who:** level-0 employees who are high on the model's list.
- **How to test:** randomly give a first grant to half of the eligible level-0 employees and compare leave rates after 12 months (grants usually vest over time).
  Size: about 630 level-0 employees company-wide. Even a pilot on about 314 per half detects only a drop of about **10 points** or more, and the grant cost should be priced before choosing the size.
- **Caveat (Rohit):** all 373 Single employees on train have stock level 0, and marital status isn't a model input. "No stock" may partly stand in for "single", a group with other reasons to move. If so, grants won't change much. This is the recommendation the pilot is most likely to overturn.

---

## What these recommendations don't use

- **Age, gender and marital status.** They aren't model inputs (DECISIONS 2026-10-28), and targeting by them wouldn't be fair or acceptable. Their links to the inputs (e.g. Single ↔ stock level 0) are checked in M5.
- **Salary hike and performance rating.** These have almost no push in either model (0.07–0.08). That means the model can't use them to separate leavers, not that raises don't matter.

## How the pilot sizes were worked out

Rough two-group comparison of leave rates: normal approximation, 80% power, 5% two-sided. Company-wide counts scale the training-split counts by 1,470 / 1,176. These are estimates for planning, not measured results.
