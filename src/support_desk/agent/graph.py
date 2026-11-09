"""The support agent as a LangGraph state machine.

    START -> agent --(tool call)--> guard --allow--------------> tools --> agent
               |                      |----needs approval--> approval --approved--> tools
               |                      |                          `--rejected--> agent
               |                      `----blocked (policy)----------------------> agent
               |                                    tools --(transfer)--> handoff --> END
               `--(text reply)--> END   (wait for the customer's next message)

- agent:    one model call with the policy prompt and the tool list.
- guard:    validates arguments against the tool's pydantic model and runs the policy
            engine. Nothing reaches a tool without passing here.
- approval: `interrupt()`; the graph is checkpointed and parked until a human approves or
            rejects in the console (or the eval's auto-approver resumes it).
- tools:    runs the tool inside a store transaction, with an idempotency key for writes
            and an audit-log entry for every call.
- handoff:  opens a ticket with the model's summary plus facts the code collected
            (verified customer, actions taken, refusals), then ends the conversation.
"""

from __future__ import annotations

import json
import operator
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from ..domain.store import Store
from ..domain.tools import TOOLS, Kind, run_tool, validate_args
from ..kb.retriever import HelpCenter, validate_citations
from ..llm.client import LLMClient
from ..ops import OpsStore, idempotency_key
from ..pii import redact, redact_text
from ..policy.guards import AUTH_TOOLS, PolicyEngine, Rule, Verdict, target_of

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"


class SearchHelpCenter(BaseModel):
    """Search the store's help-center articles for policy questions (returns, exchanges, cancellations, refunds, account). Returns passages with [kb:...] markers to cite."""  # noqa: E501

    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="What the customer wants to know, in a few words.")


KB_TOOL = "search_help_center"


def _kb_spec() -> dict[str, Any]:
    schema = SearchHelpCenter.model_json_schema()
    for k in ("title", "description"):
        schema.pop(k, None)
    for p in schema["properties"].values():
        p.pop("title", None)
    return {"name": KB_TOOL, "description": SearchHelpCenter.__doc__, "input_schema": schema}


def tool_specs() -> list[dict[str, Any]]:
    return [t.spec() for t in TOOLS.values()] + [_kb_spec()]


def load_prompt(name: str, **fmt: str) -> str:
    text = (PROMPTS / f"{name}.md").read_text()
    lines = text.splitlines()
    if lines and lines[0].startswith("<!--"):
        text = "\n".join(lines[1:]).lstrip("\n")
    return text.format(**fmt) if fmt else text


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    auth_user_id: str | None
    pending_call: dict[str, Any] | None
    executed: Annotated[list[dict[str, Any]], operator.add]
    violations: Annotated[list[dict[str, Any]], operator.add]
    blocked: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    kb_ids: Annotated[list[str], operator.add]
    replies: Annotated[list[str], operator.add]
    handoff: dict[str, Any] | None


@dataclass
class Deps:
    store: Store
    ops: OpsStore
    llm: LLMClient
    kb: HelpCenter
    policy: PolicyEngine
    policy_text: str


def _tid(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def _ev(node: str, kind: str, **detail: Any) -> dict[str, Any]:
    return {"ts": round(time.time(), 3), "node": node, "kind": kind, "detail": redact(detail)}


def _preview(result: str, limit: int = 140) -> str:
    """Short, redacted view of a tool result for the console. JSON results get key-based
    redaction (names, addresses, emails) before truncation, so a cut can't expose a field."""
    try:
        return json.dumps(redact(json.loads(result)))[:limit]
    except ValueError:
        return redact_text(result)[:limit]


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


def build_graph(deps: Deps, checkpointer=None):
    system = SystemMessage(load_prompt("agent_v1", policy=deps.policy_text))
    specs = tool_specs()

    def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        ai = deps.llm.invoke([system, *state["messages"]], tools=specs, tag="agent", meta={"thread": _tid(config)})
        ms = round((time.perf_counter() - started) * 1000)
        update: dict[str, Any] = {"messages": [ai]}
        if ai.tool_calls:
            call = ai.tool_calls[0]
            update["trace"] = [_ev("agent", "tool_call", tool=call["name"], args=call["args"], ms=ms)]
        else:
            # The stored message stays byte-identical (thinking blocks are bound to it);
            # only the customer-facing copy has unverifiable citations removed.
            reply, kept, dropped = validate_citations(_text(ai), set(state.get("kb_ids", [])))
            update["replies"] = [reply]
            update["trace"] = [_ev("agent", "reply", text=reply, citations=kept, dropped_citations=dropped, ms=ms)]
        return update

    def route_agent(state: AgentState) -> Literal["guard", "__end__"]:
        last = state["messages"][-1]
        return "guard" if isinstance(last, AIMessage) and last.tool_calls else END

    def guard(state: AgentState, config: RunnableConfig) -> Command:
        tid = _tid(config)
        ai = state["messages"][-1]
        calls = ai.tool_calls
        extra = [
            ToolMessage("Error: make one tool call at a time; this call was not run.", tool_call_id=c["id"])
            for c in calls[1:]
        ]
        call = calls[0]
        name = call["name"]

        def reject(text: str, **ev: Any) -> Command:
            return Command(
                goto="agent",
                update={
                    "messages": [ToolMessage(text, tool_call_id=call["id"]), *extra],
                    "pending_call": None,
                    **ev,
                },
            )

        if name == KB_TOOL:
            return Command(goto="tools", update={"pending_call": {"call": call}, "messages": extra})

        args = validate_args(name, call["args"])
        if isinstance(args, str):
            return reject(args, trace=[_ev("guard", "invalid_args", tool=name, error=args)])
        if TOOLS[name].kind is not Kind.WRITE and name not in {"get_user_details", "get_order_details"}:
            return Command(goto="tools", update={"pending_call": {"call": call}, "messages": extra})

        with deps.store.transaction() as view:
            decision = deps.policy.check(
                name,
                args,
                data=view,
                auth_user_id=state.get("auth_user_id"),
                messages=state["messages"],
                executed=state.get("executed", []),
            )
        violations = [{"rule": r.value, "why": w, "tool": name} for r, w in decision.violations]
        pending = {
            "call": call,
            "confirm_idx": decision.confirm_idx,
            "refund": decision.refund,
            "violations": violations,
        }
        if decision.verdict is Verdict.BLOCK:
            deps.ops.audit(tid, "blocked", tool=name, target=target_of(name, args), detail={"why": decision.message})
            return reject(
                f"Policy: {decision.message}. The action was not performed.",
                blocked=[{"tool": name, "violations": violations}],
                trace=[_ev("guard", "blocked", tool=name, why=decision.message)],
            )
        if decision.verdict is Verdict.APPROVAL:
            approval_id = "A-" + uuid.uuid4().hex[:8].upper()
            pending["approval_id"] = approval_id
            deps.ops.approval_open(approval_id, tid, name, target_of(name, args), decision.refund, call["args"])
            deps.ops.audit(
                tid,
                "approval_requested",
                tool=name,
                target=target_of(name, args),
                detail={"approval_id": approval_id, "refund": decision.refund},
            )
            return Command(
                goto="approval",
                update={
                    "pending_call": pending,
                    "messages": extra,
                    "trace": [
                        _ev("guard", "needs_approval", tool=name, refund=decision.refund, approval_id=approval_id)
                    ],
                },
            )
        return Command(
            goto="tools",
            update={
                "pending_call": pending,
                "messages": extra,
                "trace": [_ev("guard", "allowed", tool=name, audit_only_violations=violations)],
            },
        )

    def approval(state: AgentState, config: RunnableConfig) -> Command:
        pending = state["pending_call"]
        call = pending["call"]
        answer = interrupt(
            {
                "approval_id": pending["approval_id"],
                "tool": call["name"],
                "refund": pending["refund"],
                "args": redact(call["args"]),
            }
        )
        approved = bool(answer.get("approved"))
        by = str(answer.get("by", "supervisor"))
        note = str(answer.get("note", ""))
        deps.ops.approval_decide(pending["approval_id"], "approved" if approved else "rejected", by, note)
        deps.ops.audit(
            _tid(config),
            "approval_" + ("granted" if approved else "rejected"),
            actor=by,
            tool=call["name"],
            detail={"approval_id": pending["approval_id"], "note": note},
        )
        ev = _ev("approval", "approved" if approved else "rejected", by=by, note=note)
        if approved:
            return Command(goto="tools", update={"trace": [ev]})
        msg = (
            "Policy: a supervisor rejected this refund"
            + (f" ({note})" if note else "")
            + ". It was not performed. Tell the customer, and offer a transfer to a human agent."
        )
        return Command(
            goto="agent",
            update={"messages": [ToolMessage(msg, tool_call_id=call["id"])], "pending_call": None, "trace": [ev]},
        )

    def tools(state: AgentState, config: RunnableConfig) -> Command:
        tid = _tid(config)
        pending = state["pending_call"]
        call = pending["call"]
        name = call["name"]
        update: dict[str, Any] = {"pending_call": None}

        if name == KB_TOOL:
            text, ids = deps.kb.render(str(call["args"].get("query", "")))
            update.update(
                messages=[ToolMessage(text, tool_call_id=call["id"])],
                kb_ids=ids,
                trace=[_ev("tools", "kb_search", query=call["args"].get("query"), hits=ids)],
            )
            return Command(goto="agent", update=update)

        args = validate_args(name, call["args"])
        assert not isinstance(args, str)
        kind = TOOLS[name].kind
        replay = False
        started = time.perf_counter()
        with deps.store.transaction() as view:
            if kind is Kind.WRITE:
                key = idempotency_key(tid, name, args.model_dump())
                prior = deps.ops.idem_get(key)
                if prior is not None:
                    result, replay = prior, True
                else:
                    result = run_tool(view, name, args)
                    if not result.startswith("Error"):
                        deps.ops.idem_put(key, tid, name, result)
            else:
                result = run_tool(view, name, args)
        ms = round((time.perf_counter() - started) * 1000, 1)
        ok = not result.startswith("Error")

        auth = state.get("auth_user_id")
        if name in AUTH_TOOLS and ok:
            if auth is None:
                update["auth_user_id"] = result
                deps.ops.audit(tid, "customer_verified", tool=name, target=result)
            elif result != auth:
                violation = {
                    "rule": Rule.SINGLE.value,
                    "why": "second customer identity in one conversation",
                    "tool": name,
                }
                if deps.policy.mode == "enforce":
                    result, ok = (
                        "Policy: R2_single_customer: this conversation is already verified as a different "
                        "customer. Only one customer can be helped per conversation.",
                        False,
                    )
                    update["blocked"] = [{"tool": name, "violations": [violation]}]
                else:
                    update["auth_user_id"] = result
                    update["violations"] = [violation]

        if kind is Kind.WRITE:
            target = target_of(name, args)
            deps.ops.audit(
                tid,
                "write_replayed" if replay else ("write_ok" if ok else "write_failed"),
                tool=name,
                target=target,
                detail={
                    "args": args.model_dump(),
                    "refund": pending.get("refund", 0.0),
                    "result": result[:300] if not ok else "ok",
                },
            )
            if ok:
                update["executed"] = [
                    {
                        "tool": name,
                        "target": target,
                        "confirm_idx": pending.get("confirm_idx"),
                        "args": args.model_dump(),
                        "refund": pending.get("refund", 0.0),
                        "replay": replay,
                    }
                ]
                if pending.get("violations"):
                    update["violations"] = update.get("violations", []) + pending["violations"]
        elif kind is Kind.HANDOFF:
            deps.ops.audit(tid, "handoff_requested", tool=name, detail={"summary": args.summary})
        else:
            deps.ops.audit(tid, "read", tool=name, detail={"ok": ok})

        update["messages"] = [ToolMessage(result, tool_call_id=call["id"])]
        update["trace"] = [_ev("tools", "result", tool=name, ok=ok, replay=replay, ms=ms, preview=_preview(result))]
        return Command(goto="handoff" if kind is Kind.HANDOFF else "agent", update=update)

    def handoff(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tid = _tid(config)
        call = next(
            c
            for m in reversed(state["messages"])
            if isinstance(m, AIMessage)
            for c in m.tool_calls
            if c["name"] == "transfer_to_human_agents"
        )
        last_customer = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        context = {
            "verified": state.get("auth_user_id") is not None,
            "actions_taken": [{"tool": e["tool"], "target": e["target"]} for e in state.get("executed", [])],
            "policy_blocks": [b["tool"] for b in state.get("blocked", [])],
            "last_customer_message": last_customer,
            "turns": sum(isinstance(m, HumanMessage) for m in state["messages"]),
        }
        ticket = deps.ops.ticket_create(tid, state.get("auth_user_id"), call["args"].get("summary", ""), context)
        deps.ops.audit(tid, "ticket_opened", detail={"ticket": ticket})
        text = load_prompt("handoff_v1", ticket_ref=f" (ticket {ticket})")
        return {
            "messages": [AIMessage(text)],
            "replies": [text],
            "handoff": {"ticket": ticket, "summary": call["args"].get("summary", ""), "context": context},
            "trace": [_ev("handoff", "ticket", ticket=ticket)],
        }

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("guard", guard, destinations=("agent", "tools", "approval"))
    g.add_node("approval", approval, destinations=("agent", "tools"))
    g.add_node("tools", tools, destinations=("agent", "handoff"))
    g.add_node("handoff", handoff)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_agent, {"guard": "guard", END: END})
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)
