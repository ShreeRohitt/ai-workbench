"""USD per 1M tokens, used to put a cost on every logged LLM call.

Anthropic rates come from the Claude API model table (cached 2026-10-06). Re-check them before
quoting costs in a README. OpenAI models are deliberately absent: add them from
https://openai.com/api/pricing when you first use one. Until then, their calls log cost_usd = None
rather than a made-up number.
"""

PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    # Haiku 5.5 rates apply to prompts up to 100K tokens ($0.50 / $2.50 beyond).
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    # Server-side refusal fallbacks can route a request to these models.
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}

# Defaults when a model has no explicit rate: 5-minute cache writes cost 1.25x input,
# cache reads ~0.1x input.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Cost of one call. `input_tokens` = uncached input only. None for unpriced models."""
    rates = PRICING.get(model)
    if rates is None:
        return None
    cache_read_rate = rates.get("cache_read", rates["input"] * CACHE_READ_MULTIPLIER)
    total = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * cache_read_rate
        + cache_write_tokens * rates["input"] * CACHE_WRITE_MULTIPLIER
    )
    return total / 1_000_000
