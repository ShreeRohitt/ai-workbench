"""Thin LLM client: one `call()` for Anthropic and OpenAI.

Every call (including failed ones) is appended to logs/llm_calls.jsonl with model, tokens,
latency and cost, so projects can report real cost/latency numbers.

    from shared.llm import call
    result = call("Summarise this policy: ...", tag="rag-answer")
    print(result.text, result.cost_usd)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from shared.pricing import cost_usd

load_dotenv()

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "llm_calls.jsonl"
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}

# Models that accept the server-side refusal fallback ("default" routing by refusal category).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class RawResponse:
    """Provider-neutral view of one response. `input_tokens` excludes cached tokens."""

    text: str
    model: str  # the model that actually served the request (may differ after a fallback)
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    stop_reason: str | None


@dataclass
class LLMResult(RawResponse):
    provider: str
    latency_s: float
    cost_usd: float | None  # None when the model has no entry in shared/pricing.py
    tag: str | None


@cache
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


@cache
def _openai_client():
    import openai

    return openai.OpenAI()


def _call_anthropic(model: str, system: str | None, messages: list[dict], max_tokens: int, **kw):
    params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages, **kw}
    if system:
        params["system"] = system
    client = _anthropic_client()
    if model in _FALLBACK_MODELS:
        resp = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
        )
    else:
        resp = client.messages.create(**params)
    u = resp.usage
    return RawResponse(
        text="".join(b.text for b in resp.content if b.type == "text"),
        model=resp.model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_tokens=u.cache_creation_input_tokens or 0,
        stop_reason=resp.stop_reason,  # check for "refusal" before trusting `text`
    )


def _call_openai(model: str, system: str | None, messages: list[dict], max_tokens: int, **kw):
    resp = _openai_client().responses.create(
        model=model, instructions=system, input=messages, max_output_tokens=max_tokens, **kw
    )
    u = resp.usage
    cached = u.input_tokens_details.cached_tokens if u.input_tokens_details else 0
    return RawResponse(
        text=resp.output_text,
        model=resp.model,
        input_tokens=u.input_tokens - cached,  # OpenAI counts cached tokens inside input_tokens
        output_tokens=u.output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=0,
        stop_reason=resp.status,
    )


_PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai}


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def call(
    prompt: str | None = None,
    *,
    messages: list[dict] | None = None,
    system: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    max_tokens: int = 4096,
    tag: str | None = None,
    **kwargs: Any,
) -> LLMResult:
    """Send one request. Pass either `prompt` (a single user turn) or a full `messages` list.

    `tag` labels the call in the log (e.g. "m4-judge") so costs can be grouped per experiment.
    Extra kwargs go straight to the provider SDK (e.g. output_config={"effort": "low"}).
    """
    if (prompt is None) == (messages is None):
        raise ValueError("pass exactly one of `prompt` or `messages`")
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(_PROVIDERS)}")
    model = model or DEFAULT_MODELS.get(provider)
    if model is None:
        raise ValueError(f"no default model for {provider!r}; pass model=...")
    messages = messages or [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    base = {"ts": datetime.now(UTC).isoformat(), "provider": provider, "tag": tag}
    try:
        raw = _PROVIDERS[provider](model, system, messages, max_tokens, **kwargs)
    except Exception as e:
        latency = round(time.perf_counter() - start, 3)
        _log({**base, "model": model, "latency_s": latency, "error": repr(e)})
        raise

    result = LLMResult(
        **asdict(raw),
        provider=provider,
        latency_s=round(time.perf_counter() - start, 3),
        cost_usd=cost_usd(
            raw.model,
            raw.input_tokens,
            raw.output_tokens,
            raw.cache_read_tokens,
            raw.cache_write_tokens,
        ),
        tag=tag,
    )
    _log({**base, **{k: v for k, v in asdict(result).items() if k != "text"}})
    return result
