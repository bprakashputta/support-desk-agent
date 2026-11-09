"""Spend ledger, cost maths, response cache and the cap: no network."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from support_desk.llm.client import LLMClient, ResponseCache
from support_desk.llm.ledger import BudgetExceeded, SpendLedger
from support_desk.llm.pricing import cost_usd


def test_cost_maths():
    # 50K uncached input + 50K output on Haiku 5.5 ($0.10 / $0.50 per MTok)
    assert cost_usd("claude-haiku-5-5", input_tokens=50_000, output_tokens=50_000) == pytest.approx(0.03)
    # cache reads cost 10% of input, writes 125%
    assert cost_usd("claude-sonnet-5-5", input_tokens=0, output_tokens=0, cache_read=1_000_000) == pytest.approx(0.20)
    assert cost_usd("claude-sonnet-5-5", input_tokens=0, output_tokens=0, cache_write=1_000_000) == pytest.approx(2.5)
    # Haiku long-prompt rate above 100K prompt tokens
    assert cost_usd("claude-haiku-5-5", input_tokens=200_000, output_tokens=0) == pytest.approx(0.10)


def test_ledger_cap_and_cross_process_reads(tmp_path):
    path = tmp_path / "spend.jsonl"
    a = SpendLedger(path, cap_usd=0.01)
    b = SpendLedger(path, cap_usd=0.01)  # a second process sharing the file
    a.record({"cost_usd": 0.006})
    b.check()
    b.record({"cost_usd": 0.005})
    with pytest.raises(BudgetExceeded):
        a.check()
    assert a.total == pytest.approx(0.011)


class FakeChat:
    def __init__(self):
        self.calls = 0
        self.kwargs = []

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages, **kw):
        self.calls += 1
        self.kwargs.append((messages, kw))
        return AIMessage(
            content="hi",
            response_metadata={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                }
            },
        )


def _client(tmp_path, cap=1.0, cache=True):
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap)
    c = LLMClient(
        "claude-haiku-5-5", ledger, cache=ResponseCache(tmp_path / "c.sqlite") if cache else None, api_key="test"
    )
    c._chat = FakeChat()
    return c, ledger


def test_identical_calls_hit_the_cache_and_cost_nothing(tmp_path):
    c, ledger = _client(tmp_path)
    msgs = [SystemMessage("policy"), HumanMessage("hello")]
    c.invoke(msgs)
    c.invoke(msgs)
    assert c._chat.calls == 1
    assert c.stats.calls == 2 and c.stats.cached_calls == 1
    assert ledger.total == pytest.approx(
        cost_usd("claude-haiku-5-5", input_tokens=10, output_tokens=20, cache_read=1000)
    )


def test_prompt_caching_breakpoints_are_sent(tmp_path):
    c, _ = _client(tmp_path, cache=False)
    c.invoke([SystemMessage("policy"), HumanMessage("hello")])
    messages, kw = c._chat.kwargs[0]
    assert messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert kw["cache_control"] == {"type": "ephemeral"}


def test_cap_refuses_calls(tmp_path):
    c, ledger = _client(tmp_path, cap=0.0001, cache=False)
    ledger.record({"cost_usd": 0.0001})
    with pytest.raises(BudgetExceeded):
        c.invoke([HumanMessage("hello")])
    assert c._chat.calls == 0
