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

### 03-M2 · Validation harness + baselines — `in-progress` (branch `03-m2-baseline`)
- Goal: one fair way to score any model on train only (cross-validation), then score random, the rule, and logistic regression with it. Test set stays locked.
- Decided: k = top 10% of whoever is being scored (≈24 per CV fold, 30 on the test set).
- Acceptance:
  - [x] `attrition/metrics.py::precision_at_k` (Rohit) — 13 tests pass; stable tie-breaking
  - [x] CV harness `attrition/cv.py::cross_validate`: 5×5 repeated stratified folds, eval labels hidden from the model; 9 tests pass. Written with Claude's help (another session); Rohit explained `.iloc` vs `.loc` and `divmod`. **Next session first:** explain `np.zeros` as X in `.split()`, and why `.copy()`.
  - [x] Preprocessing inside a sklearn Pipeline (Rohit, with the LR baseline)
  - [x] Baselines (Rohit): random, rule (OverTime=Yes & JobLevel=1; 51.7% on train, n=116), plain LR. Measured so far (2026-10-30, exploratory run): random precision@10% = 0.147 mean over 25 folds, range 0.04–0.25 (k=24).
  - [x] Experiment: LR with vs without the 5 survey columns, both reported (`attrition.report`; paired: `notebooks/02_paired_comparison.py`)
  - [x] Optional: LR + `overtime_and_level1` column (tests the bet's reasoning). Predictions (2026-11-04), PR-AUC change vs plain LR (0.533): Rohit **+3 points** (LR already gets close on that group: 45.9% predicted vs 51.7% actual); Claude **< 1 point** (JobRole, income, stock level already describe that group). **Result (2026-11-04):** PR-AUC 0.533 → 0.541 (+0.8 pts; wins 16/25 folds, 9 losses), precision@10% 0.603 → 0.605 (8 wins / 9 ties / 8 losses). Claude's call won; gain is within fold noise. Plain LR stays the baseline.
  - [x] One command prints PR-AUC and precision@10% as mean ± spread across folds, next to the random baseline (Claude: CLI + table)
  - [ ] DECISIONS + LEARNINGS entries; 5-question interview
### 03-M3 · LightGBM, imbalance handling, operating threshold — `todo`
### 03-M4 · SHAP + 3 CHRO recommendations — `todo`
### 03-M5 · Model card, fairness checks, README — `todo`
