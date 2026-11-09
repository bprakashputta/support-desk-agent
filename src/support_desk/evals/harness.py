"""Run the agent against an LLM-simulated customer and grade the outcome.

Grading is tau-bench's: a conversation passes (reward 1) when the final database hash
equals the hash obtained by replaying the task's ground-truth actions on a fresh copy of
the database, and every required output string appears in one of the agent's replies
(case-insensitive, commas removed). pass^1 is the mean reward over tasks with one trial
each.

The simulated customer uses tau-bench's user-simulator prompt verbatim. Each conversation
gets a fresh in-memory database, a fresh in-memory checkpointer and its own ops store.
Approvals are resolved by an auto-approver (always approve) so refunds above the threshold
complete, and how often an approval was needed is reported.
"""

from __future__ import annotations

import copy
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ..agent.graph import Deps, build_graph, load_prompt
from ..agent.session import resume_approval, send_message, thread_state
from ..domain.store import MemoryStore, data_hash, load_tau_json
from ..domain.tools import run_tool, validate_args
from ..kb.retriever import HelpCenter
from ..llm.client import LLMClient
from ..llm.ledger import BudgetExceeded, SpendLedger
from ..ops import OpsStore
from ..policy.guards import PolicyEngine

MAX_TURNS = 30
STOP = "###STOP###"


@dataclass
class RunConfig:
    name: str
    agent_model: str = "claude-haiku-5-5"
    user_model: str = "claude-haiku-5-5"
    guard_mode: str = "enforce"  # "enforce" = code-enforced guards, "audit" = prompt-only
    approval_threshold: float = 1000.0
    effort: str | None = None


class Env:
    """Loaded once per process: base database, ground-truth hashes, KB, policy text."""

    def __init__(self, tau_dir: Path, kb_dir: Path):
        self.tau_dir = tau_dir
        self.base = load_tau_json(tau_dir)
        self.policy = (tau_dir / "wiki.md").read_text()
        self.kb = HelpCenter.from_dir(kb_dir)
        self._gt: dict[tuple[str, int], str] = {}

    def tasks(self, split: str) -> list[dict[str, Any]]:
        return json.loads((self.tau_dir / f"tasks_{split}.json").read_text())

    def fresh_data(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.base)

    def gt_hash(self, split: str, idx: int, task: dict[str, Any]) -> str:
        key = (split, idx)
        if key not in self._gt:
            store = MemoryStore(self.fresh_data())
            for action in task["actions"]:
                if action["name"] == "transfer_to_human_agents":
                    continue
                args = validate_args(action["name"], action["kwargs"])
                if isinstance(args, str):
                    continue
                with store.transaction() as view:
                    run_tool(view, action["name"], args)
            self._gt[key] = store.hash()
        return self._gt[key]


def _sim_reply(sim: LLMClient, history: list[BaseMessage], task_id: str) -> str:
    """Next simulated-customer message. The simulator occasionally returns no text at all
    (seen after the agent's closing "anything else?"); it is retried once and otherwise
    treated as the customer ending the conversation, never sent to the agent as an empty
    message (the API rejects empty user turns)."""
    for _ in range(2):
        msg = sim.invoke(history, tag="user_sim", meta={"task": task_id})
        content = (
            msg.content
            if isinstance(msg.content, str)
            else "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")
        ).strip()
        if content:
            history.append(AIMessage(content))
            return content
    history.append(AIMessage(STOP))
    return STOP


def run_conversation(
    env: Env,
    cfg: RunConfig,
    ledger: SpendLedger,
    *,
    task_id: str,
    instruction: str,
    opening: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """One simulated conversation. Returns the trajectory summary and the final store."""
    store = MemoryStore(env.fresh_data())
    ops = OpsStore()
    agent_llm = LLMClient(cfg.agent_model, ledger, effort=cfg.effort, api_key=api_key)
    sim = LLMClient(cfg.user_model, ledger, max_tokens=1024, api_key=api_key)
    deps = Deps(
        store=store,
        ops=ops,
        llm=agent_llm,
        kb=env.kb,
        policy=PolicyEngine(mode=cfg.guard_mode, approval_threshold=cfg.approval_threshold),
        policy_text=env.policy,
    )
    graph = build_graph(deps, checkpointer=InMemorySaver())
    thread = f"{cfg.name}:{task_id}"
    sim_history: list[BaseMessage] = [
        SystemMessage(load_prompt("user_sim_v1", instruction=instruction)),
        HumanMessage("Hi! How can I help you today?"),
    ]
    transcript: list[dict[str, str]] = []
    approvals = 0
    status = "idle"
    error = None
    started = time.perf_counter()
    end_reason = "max_turns"
    try:
        if opening:
            customer = opening
            sim_history.append(AIMessage(opening))
        else:
            customer = _sim_reply(sim, sim_history, task_id)
        for _ in range(MAX_TURNS):
            if STOP in customer:
                end_reason = "user_stop"
                break
            transcript.append({"role": "customer", "text": customer})
            turn = send_message(graph, thread, customer)
            while turn.status == "awaiting_approval":
                approvals += 1
                turn = resume_approval(graph, thread, approved=True, by="eval-auto-approver")
            reply = "\n".join(r for r in turn.replies if r)
            transcript.append({"role": "agent", "text": reply})
            status = turn.status
            if status == "handed_off":
                end_reason = "handoff"
                break
            sim_history.append(HumanMessage(reply or "(no response)"))
            customer = _sim_reply(sim, sim_history, task_id)
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001 - a crashed conversation scores 0 and is reported
        error = f"{type(e).__name__}: {e}"
        end_reason = "error"
        traceback.print_exc()
    wall = time.perf_counter() - started
    state = thread_state(graph, thread)
    a, u = agent_llm.stats, sim.stats
    return {
        "store": store,
        "transcript": transcript,
        "replies": state.get("replies", []),
        "executed": state.get("executed", []),
        "violations": state.get("violations", []),
        "blocked": state.get("blocked", []),
        "handoff": bool(state.get("handoff")),
        "approvals": approvals,
        "end_reason": end_reason,
        "error": error,
        "wall_s": round(wall, 2),
        "agent_calls": a.calls,
        "agent_latency_s": round(a.latency_s, 2),
        "agent_cost_usd": round(a.cost_usd, 6),
        "user_cost_usd": round(u.cost_usd, 6),
        "agent_tokens": {
            "input": a.input_tokens,
            "cache_read": a.cache_read,
            "cache_write": a.cache_write,
            "output": a.output_tokens,
        },
        "audit_rows": len(ops.audit_rows(limit=10_000)),
    }


def grade_tau(env: Env, split: str, idx: int, task: dict[str, Any], conv: dict[str, Any]) -> dict[str, Any]:
    r_db = conv["store"].hash() == env.gt_hash(split, idx, task)
    outputs = {}
    for out in task.get("outputs", []):
        outputs[out] = any(out.lower() in r.lower().replace(",", "") for r in conv["replies"])
    r_out = all(outputs.values()) if outputs else True
    return {"reward": float(r_db and r_out), "r_db": r_db, "r_outputs": r_out, "outputs": outputs}


def db_changed(env: Env, store: MemoryStore) -> bool:
    return store.hash() != data_hash(env.base)
