# Decisions

## 2026-10-25 · Separate repo from Ford interview prep
- Chose: `~/ai-workbench` as its own git repo.
- Alternatives: a subfolder of `~/prep`.
- Why: a portfolio repo must be public-clean; the prep repo holds resume notes and personal context.

## 2026-10-25 · Unpriced models log `cost_usd = null`
- Chose: price only models listed in `shared/pricing.py`; anything else logs null.
- Alternatives: guess a price; hard-fail on unknown models.
- Why: "never invent results" also covers costs. Null is visible in reports and prompts us to add the real rate.

## 2026-10-25 · Server-side refusal fallback on by default (Anthropic)
- Chose: `fallbacks="default"` on models that support it; the log records the model that actually served the request.
- Alternatives: no fallback (a refused request just stops).
- Why: eval runs shouldn't silently lose rows to a refusal. Cost is computed on the serving model. Remove it in `shared/llm.py` if unwanted.

## 2026-10-25 · Chroma before pgvector
- Chose: Chroma for 01-M2 until Docker is installed.
- Alternatives: pgvector now (blocked: no Docker).
- Why: unblocks retrieval work; revisit at M6 when we containerise anyway.

## 2026-10-26 · 03: lock a stratified 20% test set before EDA
- Chose: `make_split(df, test_size=0.2, seed=42)`, stratified on Attrition; EDA and all model selection use train only.
- Alternatives: EDA on all rows + CV only; nested CV with no holdout.
- Why: anything seen during EDA shapes modelling choices, so the test set must be unseen. Cost: ~47 test leavers, so test metrics get wide CIs — we'll report bootstrap intervals.

## 2026-10-28 · 03: protected attributes are described, not modelled
- Chose: Age, Gender, MaritalStatus are not model inputs. EDA describes them (notebook §5) as the baseline for M5 fairness checks; they stay out of the top-5 findings and recommendations.
- Alternatives: use them as features; drop them from the data entirely.
- Why: a retention list ranked by age or marital status isn't something HR can act on fairly. Keeping the columns lets us test whether the model rebuilds them through proxies. Known proxies on train: all 373 Single employees have StockOptionLevel 0; Age has rank correlation 0.67 with TotalWorkingYears and 0.51 with JobLevel. M5 must check model scores by these groups.

## 2026-10-28 · 03: survey columns kept, but flagged and tested both ways
- Chose: JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement stay `suspicious`; M2 trains with and without them and reports both.
- Alternatives: use them without caveat; drop them.
- Why: the dataset doesn't say when the surveys were taken. If they were taken at exit, they leak the label. Running both shows how much of the score depends on them.

## 2026-10-28 · 03: primary metric is precision@k, k = 30 per month (assumed)
- Chose: rank employees by predicted risk; report precision@k against the random baseline (= base rate, 16%). k = 30 assumes 2 HR partners × 15 conversations a month.
- Alternatives: ROC-AUC (ranks the whole list, but HR only acts on the top); accuracy (84% by predicting "nobody leaves"); a fixed probability threshold.
- Why: HR capacity is fixed, so what matters is the quality of the top of the list. At fixed k, precision@k and recall@k rise together, so there's no precision/recall trade-off to choose. No target is set before measuring: if 16% leave per year (time window unknown), about 20 leave per month, which caps precision@30 at about 67%. Open: on the 294-row test set, k should probably be a percentage (top 10%) rather than 30.

## 2026-10-28 · 03: EDA claims need a gap larger than about 2× the standard error
- Chose: a group difference is reported as a finding only if it's larger than about 2× its standard error (√(p(1−p)/n) per group); otherwise "no clear difference".
- Alternatives: report every visible gap; run formal tests with multiple-comparison correction.
- Why: with n ≈ 50 a group's rate swings ±10 points from chance alone. This rule removed three claims (Sales Exec, EducationField, RelationshipSatisfaction). It's a screening rule, not a test: with ~30 comparisons, one or two false positives are still expected.

## 2026-10-28 · 03: drop DailyRate, HourlyRate, MonthlyRate as noise
- Chose: drop all three (audit category `drop: noise`).
- Alternatives: keep them and let M2 feature importance show they add nothing.
- Why: on train their Pearson correlation is ≈0 with MonthlyIncome (0.02, 0.00, 0.04) and with each other, so they don't measure pay and look like synthetic noise. Fewer noise columns means less chance of fitting noise with ~190 training leavers.

## 2026-11-01 · 03: PercentSalaryHike and PerformanceRating are used in the model
- Chose: move both from `suspicious` to `use`.
- Alternatives: leave them out; test them with and without, like the survey columns.
- Why: they are set by the company at review, so the exit-survey concern doesn't apply. This is an assumption: the dataset doesn't document when they were recorded. If they were recorded after someone resigned they would leak the label, so an unexpectedly high score should send us back to these columns.

## 2026-11-02 · 03: the baseline to beat is logistic regression without the survey columns
- Chose: plain logistic regression (precision@10% 0.603 ± 0.091 over 25 folds) is the model we'd deploy and the one M3 must beat.
- Alternatives: logistic regression + 5 survey columns, which scores higher (0.717 ± 0.071); the overtime + Level 1 rule (0.488 ± 0.089).
- Why (Rohit): HR makes a fixed number of calls, so precision at the top decides it. The survey's timing is unknown, so its gain can't be trusted. If the answers were given after someone decided to leave, the gain is leakage and disappears in production. Evidence either way: stayers have survey answers too (0 missing in either group), so it isn't an exit-only survey. Leavers' averages are 0.13–0.33 lower on a 1–4 scale, which fits both "real signal" and "leak". Both rows stay in the results table. Paired comparison on the same folds (`notebooks/02_paired_comparison.py`): LR beats the rule on 20 of 25 folds (3 ties, 2 losses); + survey beats LR on 22 (2 ties, 1 loss). Neither gain is fold noise, but that can't rule out a survey leak.

## 2026-11-04 · 03: no hand-made overtime × Level 1 column in the baseline
- Chose: keep plain logistic regression as the baseline; the `LR + overtime_and_level1` row stays in the results table as a recorded experiment.
- Alternatives: add the column to the baseline.
- Why: on the same 25 folds it moved PR-AUC 0.533 → 0.541 (16 wins, 9 losses) and precision@10% 0.603 → 0.605 (8 wins, 9 ties, 8 losses), both within fold noise. The survey columns, by contrast, won 22 of 25. LR already gets close on that group without the column (45.9% predicted vs 51.7% actual on train). This weakens the stated reason for the "LightGBM +20 PR-AUC" bet but doesn't settle it: trees may still find other combinations or curved effects. M3 settles it.

## 2026-11-20 · 03: plain logistic regression stays the model; LightGBM is not adopted
- Chose (Rohit): plain LR (NUMERIC + CATEGORICAL, no survey, no class weights) is the M3 model and goes into M4 SHAP.
- Alternatives: LightGBM with fixed small-data defaults; LightGBM tuned by nested CV (12-setting grid, 3 inner folds); carry both to M5 and pick on the test set.
- Why (Rohit): LR is simpler and LightGBM gains under half a person per list. Paired, same 25 folds: LightGBM PR-AUC +0.020 (18 wins / 7 losses), precision@10% +0.003 (13 / 5 ties / 7); tuned LightGBM PR-AUC +0.022 (19 / 6), precision@10% +0.017 (10 / 9 / 6), i.e. ~0.4 more leavers per 24 calls. Fit-vs-eval PR-AUC gap: LR +0.056, LightGBM +0.258. Picking between models on the test set was rejected: the test set is for one final score, not for choosing. Settles the 03-M1 bet: +0.02 measured vs +0.20 bet; the limit is the information in the columns, not the model.

## 2026-11-20 · 03: no class weights
- Chose: no `class_weight="balanced"` for either model.
- Alternatives: balanced weights (leavers count ~5× in the loss).
- Why: it hurt LR on the same 25 folds (PR-AUC −0.024, 2 wins / 23 losses; precision@10% −0.030) and did nothing for LightGBM (PR-AUC −0.007, 9 / 16). We rank people and take the top k, so shifting predicted probabilities up doesn't help; reweighting cuts the effective sample from 1,176 to 637 rows.

## 2026-11-20 · 03: PercentSalaryHike and PerformanceRating stay in, with a measured fallback
- Chose: keep both in NUMERIC (confirms 2026-11-01).
- Alternatives: drop both now, since their recording time is undocumented.
- Why: the sensitivity row shows the model doesn't depend on them: dropping both moved PR-AUC +0.003 (std of per-fold diff 0.008; 17 wins / 8 losses) and precision@10% +0.007 (8 / 13 ties / 4), within noise. Keeping them leaves every earlier row comparable. If review timing turns out to be after the prediction date, drop both at no measured cost. Note: PerformanceRating = 4 exactly when PercentSalaryHike ≥ 20% on train (991/0, 0/185), so the two columns are redundant and SHAP in M4 may split credit between them.

## 2026-11-20 · 03: HR picks k from the capacity curve; the model doesn't
- Chose: report precision and recall for k = 10…60 per ~235 employees (`attrition.capacity`) instead of a single k. k = 30 stays the default for headline numbers (2026-10-28).
- Alternatives: fix k = 30 only; pick k where precision drops below a threshold.
- Why: extra leavers per extra 10 calls fall 5.6 → 4.6 → 3.4 → 2.7 → 1.5; calls 51–60 find no more than random (1.6 per 10), so going past 50 calls isn't worth it. Between 30 and 50 the calls are still 1.7–2× random, so the stopping point depends on cost per call vs value of retaining one person, which is HR's call. At k = 30: 17.0 of ~38 leavers, precision 0.565 ± 0.083, recall 0.446.
