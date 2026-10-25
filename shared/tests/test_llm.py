import json

import pytest

from shared import llm
from shared.pricing import cost_usd


def test_cost_uses_per_million_rates():
    # 1M uncached input + 1M output on Opus 5.5 = $4 + $20
    assert cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)


def test_cost_counts_cache_reads_and_writes():
    # Opus 5.5: 1M cache reads at $0.20, 1M cache writes at 1.25 x $4
    assert cost_usd("claude-opus-5-5", 0, 0, 1_000_000, 1_000_000) == pytest.approx(0.20 + 5.0)


def test_unpriced_model_returns_none():
    assert cost_usd("some-unpriced-model", 100, 100) is None


@pytest.fixture
def fake_provider(monkeypatch, tmp_path):
    log_path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(llm, "LOG_PATH", log_path)

    def fake(model, system, messages, max_tokens, **kw):
        return llm.RawResponse(
            text="hello",
            model=model,
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
            stop_reason="end_turn",
        )

    monkeypatch.setitem(llm._PROVIDERS, "anthropic", fake)
    return log_path


def test_call_logs_one_record_without_text(fake_provider):
    result = llm.call("hi", tag="unit-test")

    assert result.text == "hello"
    assert result.cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1_000_000)
    records = [json.loads(line) for line in fake_provider.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["tag"] == "unit-test"
    assert records[0]["model"] == "claude-opus-5-5"
    assert "text" not in records[0]


def test_failed_call_is_logged_then_raised(monkeypatch, tmp_path):
    log_path = tmp_path / "calls.jsonl"
    monkeypatch.setattr(llm, "LOG_PATH", log_path)

    def boom(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setitem(llm._PROVIDERS, "anthropic", boom)
    with pytest.raises(RuntimeError):
        llm.call("hi")
    assert "rate limited" in json.loads(log_path.read_text())["error"]


def test_requires_exactly_one_of_prompt_or_messages():
    with pytest.raises(ValueError):
        llm.call()
    with pytest.raises(ValueError):
        llm.call("hi", messages=[{"role": "user", "content": "hi"}])


def test_openai_needs_explicit_model():
    with pytest.raises(ValueError, match="no default model"):
        llm.call("hi", provider="openai")
