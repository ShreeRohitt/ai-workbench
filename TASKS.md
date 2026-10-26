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

### 03-M1 · Data, locked test split, EDA — `in-progress` (branch `03-m1-eda`)
- Goal: a checksum-pinned dataset, a stratified test set locked before EDA, and an EDA notebook with written findings.
- Acceptance:
  - [x] `scripts/fetch_data.sh` downloads + verifies SHA-256
  - [x] `attrition/split.py::make_split` (Rohit) — stratified 80/20, seeded; 4 tests pass
  - [ ] EDA notebook on **train only**, with findings written as claims + numbers
  - [x] Leakage audit: every column marked known-at-prediction-time / ID-or-constant / suspicious (`docs/column_audit.md`)
  - [ ] Business framing written down: HR capacity-limited → precision@k (Rohit's pick)
  - [ ] DECISIONS + LEARNINGS entries; 5-question interview
- **The bet** (fill in before M2): LightGBM beats logistic regression on PR-AUC by ___ points. Claude: < 3 points, LR may win.

### 03-M2 · Validation harness + logistic regression baseline — `todo`
- Planned experiment: train with vs without the 5 survey columns (timing unknown) and report both.
### 03-M3 · LightGBM, imbalance handling, operating threshold — `todo`
### 03-M4 · SHAP + 3 CHRO recommendations — `todo`
### 03-M5 · Model card, fairness checks, README — `todo`
