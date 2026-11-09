"""Drive one conversation turn through the graph and summarise the resulting state.

Shared by the HTTP API and the eval harness, so both exercise the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

RECURSION_LIMIT = 80  # graph steps per customer turn (about 25 tool calls)


@dataclass
class TurnResult:
    replies: list[str]
    status: str  # "idle" | "awaiting_approval" | "handed_off"
    approval: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _summarise(graph, thread_id: str, n_replies_before: int, n_trace_before: int) -> TurnResult:
    snap = graph.get_state(_config(thread_id))
    values = snap.values
    replies = values.get("replies", [])[n_replies_before:]
    trace = values.get("trace", [])[n_trace_before:]
    if values.get("handoff"):
        return TurnResult(replies, "handed_off", trace=trace)
    for task in snap.tasks:
        for intr in task.interrupts:
            return TurnResult(replies, "awaiting_approval", approval=intr.value, trace=trace)
    return TurnResult(replies, "idle", trace=trace)


def _counts(graph, thread_id: str) -> tuple[int, int]:
    values = graph.get_state(_config(thread_id)).values or {}
    return len(values.get("replies", [])), len(values.get("trace", []))


def send_message(graph, thread_id: str, text: str) -> TurnResult:
    if not text.strip():
        raise ValueError("empty customer message")
    n_r, n_t = _counts(graph, thread_id)
    graph.invoke({"messages": [HumanMessage(text)]}, _config(thread_id))
    return _summarise(graph, thread_id, n_r, n_t)


def resume_approval(graph, thread_id: str, approved: bool, by: str, note: str = "") -> TurnResult:
    n_r, n_t = _counts(graph, thread_id)
    graph.invoke(Command(resume={"approved": approved, "by": by, "note": note}), _config(thread_id))
    return _summarise(graph, thread_id, n_r, n_t)


def thread_state(graph, thread_id: str) -> dict[str, Any]:
    return graph.get_state(_config(thread_id)).values or {}
