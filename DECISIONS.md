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

## 2026-10-25 · 01-M1: tokenizer and tiny sections
- Chose: count tokens with tiktoken `o200k_base`; merge heading sections < 50 tokens into following siblings (parent's heading path).
- Alternatives: the M2 embedder's own tokenizer (exact but blocks M1); keep tiny sections as their own chunks.
- Why: proxy counts are close enough to size chunks; re-count once the embedder is chosen. One-line sections make low-context chunks that retrieve badly.

## 2026-10-25 · Data science (03) before finishing RAG (01)
- Chose: park 01-M1 on its branch and start 03 (attrition model).
- Alternatives: finish curriculum in order.
- Why: Rohit's call — get a complete DS story first. 01-M1 resumes from the branch.
