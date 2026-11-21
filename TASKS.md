# Tasks

Status: `todo` · `in-progress` · `review` · `done`

## 00 · Workbench setup — `done` (2026-10-25)
- Goal: a repo that runs from a clean clone, with lint, tests, CI, and logged LLM costs.
- Acceptance:
  - [x] `uv sync` + `uv run pytest` pass locally
  - [x] `ruff check` / `ruff format --check` clean
  - [x] `shared/llm.py` logs model, tokens, latency, cost to `logs/llm_calls.jsonl` (unit-tested, no live call yet)
  - [ ] CI green on GitHub (needs `gh` + a remote)
  - [ ] Docker Desktop installed (needed for pgvector / 01-M6)

## 01-M1 · Ingestion + chunking — `todo`
- Goal: load the GitLab handbook's People/HR policy pages and chunk them two ways.
- You implement (core logic): `load_docs`, `chunk_fixed`, `chunk_by_heading`.
- Acceptance:
  - [ ] Raw pages fetched by a script into `data/raw/` (not committed)
  - [ ] Tests: no empty chunks; exact overlap; no text lost; every chunk <= max_tokens; heading path present
  - [ ] Stats table per strategy: chunk count, token length p50/p95/max
  - [ ] DECISIONS.md entry on chunk size/overlap; LEARNINGS.md entry in your own words
- **The bet** (fill in before writing code):
  - Your prediction: heading-aware beats fixed-size (512 tok, 64 overlap) on recall@5 by ___ points.
  - Claude's prediction: < 5 points; the M2 reranker moves recall more than chunking does.
  - Settled at: 01-M5.

## 03 · Attrition model — milestones
Data: IBM HR Analytics Employee Attrition (fictional data created by IBM), 1,470 rows, 237 leavers (16%).

### 03-M1 · Data, locked test split, EDA — `done` (merged 2026-10-29)
- Goal: a checksum-pinned dataset, a stratified test set locked before EDA, and an EDA notebook with written findings.
- Acceptance:
  - [x] `scripts/fetch_data.sh` downloads + verifies SHA-256
  - [x] `attrition/split.py::make_split` (Rohit) — stratified 80/20, seeded; 4 tests pass
  - [x] EDA notebook on **train only**, with findings written as claims + numbers (deferred: comparison group + n in most findings)
  - [x] Leakage audit: every column marked known-at-prediction-time / ID-or-constant / suspicious (`docs/column_audit.md`)
  - [x] Business framing written down: HR capacity-limited → precision@k (Rohit's pick) — `docs/business_framing.md` (draft: Q4 reason + ceiling line missing)
  - [x] DECISIONS entries
  - [x] LEARNINGS entry (Rohit)
  - [x] 5-question interview (2026-10-29): Q1 gap, Q2 OK, Q3 OK, Q4 not answered, Q5 weak → re-practise Q1 + Q3
- **The bet** (fill in before M2): LightGBM beats logistic regression on PR-AUC by ___ points. Rohit (2026-10-29): **+20 points** — reason: LR can't capture combinations like overtime + Level 1. Claude: < 3 points, LR may win.

### 03-M2 · Validation harness + baselines — `done` (merged 2026-11-05)
- Goal: one fair way to score any model on train only (cross-validation), then score random, the rule, and logistic regression with it. Test set stays locked.
- Decided: k = top 10% of whoever is being scored (≈24 per CV fold, 30 on the test set).
- Acceptance:
  - [x] `attrition/metrics.py::precision_at_k` (Rohit) — 13 tests pass; stable tie-breaking
  - [x] CV harness `attrition/cv.py::cross_validate`: 5×5 repeated stratified folds, eval labels hidden from the model; 9 tests pass. Written with Claude's help (another session); Rohit explained `.iloc` vs `.loc` and `divmod`. 2026-11-04: Rohit explained `np.zeros` as X (split only uses len(X) and y); how stratified dealing works was taught.
  - [x] Preprocessing inside a sklearn Pipeline (Rohit, with the LR baseline)
  - [x] Baselines (Rohit): random, rule (OverTime=Yes & JobLevel=1; 51.7% on train, n=116), plain LR. Measured so far (2026-10-30, exploratory run): random precision@10% = 0.147 mean over 25 folds, range 0.04–0.25 (k=24).
  - [x] Experiment: LR with vs without the 5 survey columns, both reported (`attrition.report`; paired: `notebooks/02_paired_comparison.py`)
  - [x] Optional: LR + `overtime_and_level1` column (tests the bet's reasoning). Predictions (2026-11-04), PR-AUC change vs plain LR (0.533): Rohit **+3 points** (LR already gets close on that group: 45.9% predicted vs 51.7% actual); Claude **< 1 point** (JobRole, income, stock level already describe that group). **Result (2026-11-04):** PR-AUC 0.533 → 0.541 (+0.8 pts; wins 16/25 folds, 9 losses), precision@10% 0.603 → 0.605 (8 wins / 9 ties / 8 losses). Claude's call won; gain is within fold noise. Plain LR stays the baseline.
  - [x] One command prints PR-AUC and precision@10% as mean ± spread across folds, next to the random baseline (Claude: CLI + table)
  - [x] DECISIONS + LEARNINGS entries; 5-question interview (2026-11-05): Q1 metrics partial (no sizes; "25 folds = confident"), Q2 CV design partial (thought CV prevents overfitting), Q3 failure modes weak-partial (no "how to tell"; missed intervention effect), Q4 scaling partial (1 of ~5 issues), Q5 biggest weakness OK → re-practise Q2 + Q3
### 03-M3 · LightGBM, imbalance handling, operating threshold — `done` (merged 2026-11-21; LEARNINGS last bullet pending)
- Goal: does a tree model beat LR on the same folds (settles the bet), and how many people should HR call per month, by measured numbers. Test set stays locked. Decided 2026-11-21 (Rohit): score it once at the start of M5, after M4 SHAP.
- Baseline to beat: LR precision@10% 0.603 ± 0.091, PR-AUC 0.533 ± 0.069.
- Acceptance:
  - [x] Step 1: tree + boosting explained on our data (Claude). Check Qs 2026-11-05: right direction on both, but no numbers and no "how would you tell"
  - [x] `baselines.py::lightgbm_model` (Rohit) with small-data defaults fixed before looking at scores; row in `attrition.report` — done 2026-11-06: PR-AUC 0.554 ± 0.037, precision@10% 0.607 ± 0.068 (25 folds)
    - Defaults, fixed 2026-11-05 before any LightGBM score. Categoricals one-hot encoded, same inputs as LR (NUMERIC + CATEGORICAL), fixed rounds, no early stopping.

      | param | Rohit (used) | Claude (for comparison) | Rohit's reason |
      |---|---|---|---|
      | num_leaves | 6 | 8 (max_depth 3) | small data, keep trees simple |
      | min_child_samples | 40 | 30 | each group needs ~6 leavers to be trustworthy |
      | n_estimators × learning_rate | 200 × 0.05 | 300 × 0.03 | many small nudges, not too many rounds |
      | subsample (freq 1) / colsample_bytree | 0.8 / 0.8 | 0.8 / 0.8 | randomness stops memorising |
      | reg_lambda | 5 | 1 | brake on small groups |
      | random_state / verbose | 0 / -1 | 0 / -1 | |
  - [x] Paired LightGBM vs LR comparison on PR-AUC and precision@10% → bet settled with numbers
    - Measured 2026-11-06 (`notebooks/03_lightgbm_vs_lr.py`): PR-AUC +0.020 (18 wins / 7 losses), precision@10% +0.003 (13 / 5 ties / 7). Fit-vs-eval PR-AUC gap: LR +0.056, LightGBM +0.258. Conclusion: notebook last cell (bet wrong: +0.020 vs +0.20 bet; limit is information, not model).
    - Exploratory, chosen after seeing CV scores (not a model choice; LGBM_DEFAULTS stays 40): min_child_samples=100 → fit 0.812→0.744, eval 0.554→0.554, gap +0.258→+0.190; 100 vs 40 PR-AUC +0.000 (11 wins / 14 losses). Rohit predicted eval 0.6, fit 0.7, gap 0.1.
  - [x] Class-weight experiment for LR and LightGBM (Rohit's prediction written first)
    - Predictions (2026-11-09, before any run), PR-AUC change from class_weight="balanced": Rohit **LR +0.001, LightGBM +0.020** ("mistakes on leavers now cost more"); Claude **LR ±0.01, LightGBM −0.02 to +0.01** (ranking metrics ignore the probability shift; 5× weight on 190 leavers may add noise for trees).
    - **Result (2026-11-09, `notebooks/03_lightgbm_vs_lr.py`):** LR PR-AUC −0.024 (2 wins / 23 losses), precision@10% −0.030 (3 / 7 ties / 15). LightGBM PR-AUC −0.007 (9 / 16), precision@10% +0.003 (9 / 8 / 8). Balanced weights cut effective rows 1,176 → 637. Both LR predictions wrong; Claude's LightGBM range held. Decision: no class weights.
  - [x] Any tuning only inside each fold's fit rows (Rohit), with a test that eval rows never reach tuning
    - Predictions (2026-11-06, before the run), nested-CV PR-AUC vs untuned 0.554: Rohit **+0.10 or more**; Claude **±0.01**.
    - **Result (2026-11-06, 12-setting grid, 3 inner folds):** PR-AUC 0.556 ± 0.036, precision@10% 0.620 ± 0.056. vs untuned LightGBM: PR-AUC +0.002 (12 wins / 3 ties / 10 losses). vs LR: PR-AUC +0.022 (19 / 6), precision@10% +0.017 (10 / 9 ties / 6). num_leaves=4 chosen in 21/25 folds; 7 different settings won somewhere. Spy test confirms eval rows never reach tuning.
    - Shortcut optimism (2026-11-21, added after the M3 interview Q2). Predictions, best-of-12 plain-CV PR-AUC: Rohit **0.570**; Claude **0.558–0.565**. **Result:** 12 plain-CV averages 0.552–0.562; best of 12 = 0.562 (num_leaves 4, min_child_samples 100, reg_lambda 5) vs nested 0.556 → the shortcut overstates by **+0.007**. Claude's range held; Rohit's bet too big again. The gap also includes nested CV's noisier per-fold choice (3 inner folds), so it is not pure luck.
  - [x] Sensitivity row: LR without PercentSalaryHike and PerformanceRating (from M2 interview Q5). Code written by Claude at Rohit's request.
    - Train-only check (2026-11-20, `notebooks/pandas_practice.py`): PerformanceRating 16.0% (n=991) vs 16.8% (n=185); PercentSalaryHike quartiles 17.8 / 13.9 / 17.5 / 15.4% (n 247–326), no trend. Rating 4 ⇔ hike ≥ 20% exactly (991/0, 0/185).
    - Predictions (2026-11-20, before the run), PR-AUC change vs plain LR (0.533): Rohit **−0.01** ("no signal on their own"); Claude **±0.005**.
    - **Result (2026-11-20, `notebooks/02_paired_comparison.py`):** PR-AUC 0.533 → 0.536, diff +0.003 (std of per-fold diff 0.008; 17 wins / 8 losses). precision@10% 0.603 → 0.610, diff +0.007 (std 0.029; 8 / 13 ties / 4). Dropping them costs nothing; Rohit's direction was wrong, Claude's range held. NUMERIC unchanged so earlier rows stay comparable; if review timing turns out to be after the prediction date, drop both at no measured cost.
  - [ ] Precision and recall at k = 10…60 per month: table + plot (Claude), interpretation (Rohit)
    - Table + plot done 2026-11-20 (`uv run python -m attrition.capacity`, `docs/figures/capacity_curve.png`; plain LR, 25 folds of ~235 people, ~38 leavers each). precision@k 0.680 → 0.409 and recall@k 0.179 → 0.646 from k=10 to 60. Interpretation (Rohit's answers, gaps filled in review 2026-11-20):
      - k=30: 17.0 of ~38 leavers (random 4.8, ~3.5×); precision 0.565 ± 0.083 ≈ 14–19 leavers per 30 calls; recall 0.446, so over half of leavers are missed.
      - Extra leavers per extra 10 calls: 5.6, 4.6, 3.4, 2.7, 1.5. Calls 51–60 are no better than random (1.6 per 10). Calls 31–40 are still ~2× random, so where to stop is a cost-per-call vs value-of-retention call for HR, not a model output.
      - The ± shrinks with k because each person is worth 1/k of precision (0.10 at k=10, 0.017 at k=60). Rohit's first answer explained the falling mean instead; got it after a worked example (k=5 → 0.20 per person, bigger ±).
  - [x] Tests pass (60), lint clean; DECISIONS entries (2026-11-20/21); LEARNINGS (Rohit; numbers corrected by Claude; "what I'd explain differently" still to write)
  - [x] 5-question interview (2026-11-21): Q1 ship LightGBM? partial (sized the gain in people, but mixed PR-AUC with precision@10% and ignored the 72% trap), Q2 nested CV partial (right idea, no direction/size/how-to-tell), Q3 failure modes weak-partial (3 causes, 2 checks without expected result; missed label-window mismatch and silent unknown-category bug), Q4 scaling partial (1 of ~5: per-plant; missed time-based validation, point-in-time features, monitoring), Q5 redo weak-partial (1 vague item) → re-practise Q3 + Q4
### 03-M4 · SHAP + 3 CHRO recommendations — `in-progress` (branch `03-m4-shap`)
- Goal: explain why plain LR ranks someone high, and turn it into 3 recommendations a CHRO could act on and test. Test set stays locked (scored at the start of M5).
- Plan approved 2026-11-21. Risks watched: SHAP explains the model, not causes; correlated columns split credit (hike/rating, the years columns); recommendations use levers only (no Age/Gender/MaritalStatus).
- Acceptance:
  - [x] Step 1: SHAP for a linear model taught (Claude; toy 3-column example, coefficients, sigmoid). Check Q (contribution at exactly average income): Rohit answered −0.15 × 6 (used the value, not value − average); correct answer 0.
  - [x] Step 2: bet before any SHAP output (2026-11-21), top 3 columns by mean |contribution| in plain LR:
    - Rohit: **OverTime** (seen the correlation), **MonthlyIncome**, **JobLevel or YearsAtCompany** (undecided; both scored)
    - Claude: **OverTime**, **JobRole** (9 one-hot columns summed, collects credit), **StockOptionLevel** (level 0 covers all 373 Single employees on train). MonthlyIncome and JobLevel are strongly related and will split credit, so neither makes the top 3.
  - [ ] Step 3: `attrition/explain.py::lr_contributions` (Rohit): per-employee contributions, one-hot summed per original column; tests: base + contributions = model log-odds for every row; matches `shap.LinearExplainer` (Claude: tests + shap wiring)
  - [ ] Step 4: global bar chart + beeswarm (Claude); top-5 stability across the 25 folds
  - [ ] Step 5: LightGBM SHAP cross-check: same top drivers? overtime × Level 1 interaction? (Claude wiring, Rohit interprets)
  - [ ] Step 6: 3 CHRO recommendations (Rohit): finding with size + comparison + n, lever, test plan (pilot with uncalled group), caveat
  - [ ] Step 7: DECISIONS, LEARNINGS (Rohit), 5-question interview
### 03-M5 · Model card, fairness checks, README — `todo`
- First step: score plain LR on the locked test set, once (k = 10% = 30 of 294 and the capacity table).
- README must state the time window: the 16% leaver rate covers the dataset's unknown label window, not one month. "17 of ~38 leavers at k=30" is per ~235 employees over that window; at ~16%/year, a month has only ~3 leavers per 235 (found in the M3 interview Q3, 2026-11-21).
