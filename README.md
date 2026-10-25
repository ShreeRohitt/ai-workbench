# AI Workbench

Applied AI and data science projects, each with honest, measured results.

| # | Project | Status |
|---|---|---|
| 01 | RAG system with real evaluation | in progress |
| 02 | Document extraction pipeline | planned |
| 03 | Attrition prediction model | planned |
| 04 | Tool-calling agent | planned |
| 05 | Experimentation + SQL | planned |

## Setup
```bash
uv sync
cp .env.example .env   # add API keys
uv run pytest
```

All data is public or synthetic. Every LLM call is logged to `logs/llm_calls.jsonl` (model, tokens, latency, cost).
