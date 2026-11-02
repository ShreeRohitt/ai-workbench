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
