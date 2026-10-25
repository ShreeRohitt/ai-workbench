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

## 01-M1 · Ingestion + chunking — `paused` (2026-10-25, on branch `01-m1-chunking`)
- Goal: load the GitLab handbook's People/HR policy pages and chunk them two ways.
- You implement (core logic): `load_docs`, `chunk_fixed`, `chunk_by_heading`.
- Paused to start 03 (DS first). Scaffolding + spec tests are on the branch; tests fail with NotImplementedError until you implement.
- Acceptance:
  - [x] Raw pages fetched by a script into `data/raw/` (not committed) — `scripts/fetch_handbook.sh`, pinned commit, 93 pages
  - [ ] Tests: no empty chunks; exact overlap; no text lost; every chunk <= max_tokens; heading path present
  - [ ] Stats table per strategy: chunk count, token length p50/p95/max
  - [ ] DECISIONS.md entry on chunk size/overlap; LEARNINGS.md entry in your own words
- **The bet** (fill in before writing code):
  - Your prediction: heading-aware beats fixed-size (512 tok, 64 overlap) on recall@5 by **8** points.
  - Claude's prediction: < 5 points; the M2 reranker moves recall more than chunking does.
  - Settled at: 01-M5.
