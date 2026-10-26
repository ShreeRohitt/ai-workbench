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
