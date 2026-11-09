"""Shared fixtures: a tiny retail database and a scripted stand-in for the model.

No test calls the Anthropic API. `ScriptedLLM` replays a fixed list of AIMessages, so a
test spells out exactly which tool calls the "model" makes and checks what the graph,
the policy engine and the tools do with them.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from support_desk.agent.graph import Deps, build_graph
from support_desk.domain.store import MemoryStore
from support_desk.kb.retriever import HelpCenter
from support_desk.llm.client import CallStats
from support_desk.ops import OpsStore
from support_desk.policy.guards import PolicyEngine

ROOT = Path(__file__).resolve().parents[1]

MINI = {
    "users": {
        "ana_diaz_1111": {
            "user_id": "ana_diaz_1111",
            "name": {"first_name": "Ana", "last_name": "Diaz"},
            "address": {
                "address1": "12 Oak Street",
                "address2": "Apt 4",
                "city": "Austin",
                "country": "USA",
                "state": "TX",
                "zip": "73301",
            },
            "email": "ana.diaz@example.com",
            "payment_methods": {
                "credit_card_1": {"source": "credit_card", "brand": "visa", "last_four": "4242", "id": "credit_card_1"},
                "gift_card_1": {"source": "gift_card", "balance": 50.0, "id": "gift_card_1"},
            },
            "orders": ["#W0000001", "#W0000002"],
        },
        "bo_chen_2222": {
            "user_id": "bo_chen_2222",
            "name": {"first_name": "Bo", "last_name": "Chen"},
            "address": {
                "address1": "9 Elm Road",
                "address2": "",
                "city": "Denver",
                "country": "USA",
                "state": "CO",
                "zip": "80202",
            },
            "email": "bo.chen@example.com",
            "payment_methods": {"paypal_2": {"source": "paypal", "id": "paypal_2"}},
            "orders": ["#W0000003"],
        },
    },
    "products": {
        "100": {
            "name": "T-Shirt",
            "product_id": "100",
            "variants": {
                "1001": {"item_id": "1001", "options": {"color": "red", "size": "M"}, "available": True, "price": 20.0},
                "1002": {
                    "item_id": "1002",
                    "options": {"color": "blue", "size": "M"},
                    "available": True,
                    "price": 25.0,
                },
                "1003": {
                    "item_id": "1003",
                    "options": {"color": "green", "size": "M"},
                    "available": False,
                    "price": 22.0,
                },
            },
        },
        "200": {
            "name": "Laptop",
            "product_id": "200",
            "variants": {
                "2001": {"item_id": "2001", "options": {"ram": "16GB"}, "available": True, "price": 1500.0},
            },
        },
    },
    "orders": {
        "#W0000001": {
            "order_id": "#W0000001",
            "user_id": "ana_diaz_1111",
            "address": {
                "address1": "12 Oak Street",
                "address2": "Apt 4",
                "city": "Austin",
                "country": "USA",
                "state": "TX",
                "zip": "73301",
            },
            "items": [
                {
                    "name": "T-Shirt",
                    "product_id": "100",
                    "item_id": "1001",
                    "price": 20.0,
                    "options": {"color": "red", "size": "M"},
                }
            ],
            "fulfillments": [],
            "status": "pending",
            "payment_history": [{"transaction_type": "payment", "amount": 20.0, "payment_method_id": "gift_card_1"}],
        },
        "#W0000002": {
            "order_id": "#W0000002",
            "user_id": "ana_diaz_1111",
            "address": {
                "address1": "12 Oak Street",
                "address2": "Apt 4",
                "city": "Austin",
                "country": "USA",
                "state": "TX",
                "zip": "73301",
            },
            "items": [
                {"name": "Laptop", "product_id": "200", "item_id": "2001", "price": 1500.0, "options": {"ram": "16GB"}},
                {
                    "name": "T-Shirt",
                    "product_id": "100",
                    "item_id": "1001",
                    "price": 20.0,
                    "options": {"color": "red", "size": "M"},
                },
            ],
            "fulfillments": [],
            "status": "delivered",
            "payment_history": [
                {"transaction_type": "payment", "amount": 1520.0, "payment_method_id": "credit_card_1"}
            ],
        },
        "#W0000003": {
            "order_id": "#W0000003",
            "user_id": "bo_chen_2222",
            "address": {
                "address1": "9 Elm Road",
                "address2": "",
                "city": "Denver",
                "country": "USA",
                "state": "CO",
                "zip": "80202",
            },
            "items": [
                {
                    "name": "T-Shirt",
                    "product_id": "100",
                    "item_id": "1002",
                    "price": 25.0,
                    "options": {"color": "blue", "size": "M"},
                }
            ],
            "fulfillments": [],
            "status": "pending",
            "payment_history": [{"transaction_type": "payment", "amount": 25.0, "payment_method_id": "paypal_2"}],
        },
    },
}


@pytest.fixture
def mini_data() -> dict[str, Any]:
    return copy.deepcopy(MINI)


def call(name: str, idx: int = 0, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"toolu_{name}_{idx}"}])


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedLLM:
    """Stands in for LLMClient: returns the next scripted message on each call."""

    def __init__(self, script: list[AIMessage | Callable[[list[BaseMessage]], AIMessage]]):
        self.script = list(script)
        self.seen: list[list[BaseMessage]] = []
        self.stats = CallStats()

    def invoke(self, messages, *, tools=None, tag="agent", meta=None) -> AIMessage:
        self.seen.append(list(messages))
        if not self.script:
            return AIMessage(content="(script exhausted)")
        nxt = self.script.pop(0)
        return nxt(messages) if callable(nxt) else nxt


def make_graph(store, llm, *, mode: str = "enforce", threshold: float = 1000.0, ops: OpsStore | None = None):
    ops = ops or OpsStore()
    deps = Deps(
        store=store,
        ops=ops,
        llm=llm,
        kb=HelpCenter.from_dir(ROOT / "kb"),
        policy=PolicyEngine(mode=mode, approval_threshold=threshold),
        policy_text="(policy)",
    )
    return build_graph(deps, checkpointer=InMemorySaver()), ops


@pytest.fixture
def mem_store(mini_data) -> MemoryStore:
    return MemoryStore(mini_data)
