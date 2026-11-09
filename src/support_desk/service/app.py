"""HTTP API and the two-pane UI (customer chat + agent console)."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from ..agent.graph import Deps, build_graph
from ..agent.session import resume_approval, send_message, thread_state
from ..config import Settings
from ..domain.store import SqliteStore, load_tau_json
from ..kb.retriever import HelpCenter
from ..llm.client import LLMClient, ResponseCache
from ..llm.ledger import BudgetExceeded, SpendLedger
from ..logging_setup import configure_logging
from ..ops import OpsStore
from ..pii import pseudonym, redact
from ..policy.guards import PolicyEngine

log = logging.getLogger("support_desk.api")
STATIC = Path(__file__).parent / "static"


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = SqliteStore(settings.db_path)
        if self.store.is_empty():
            if not (settings.tau_dir / "orders.json").exists():
                raise RuntimeError(f"no seed data in {settings.tau_dir}; run scripts/download_tau_bench.py")
            self.store.seed(load_tau_json(settings.tau_dir))
            log.info("seeded order database from tau-bench retail data")
        self.ops = OpsStore(self.store.conn, self.store._lock)
        self.ops.conn.execute(
            "CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, created REAL, updated REAL, status TEXT)"
        )
        self.ledger = SpendLedger(settings.data_dir / "spend.jsonl", settings.spend_cap_usd)
        self.llm = LLMClient(
            settings.agent_model,
            self.ledger,
            cache=ResponseCache(settings.data_dir / "llm_cache.sqlite"),
            api_key=settings.anthropic_api_key,
        )
        self.policy_text = (settings.tau_dir / "wiki.md").read_text()
        cp_conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
        self.checkpointer = SqliteSaver(cp_conn)
        self.graph = build_graph(
            Deps(
                store=self.store,
                ops=self.ops,
                llm=self.llm,
                kb=HelpCenter.from_dir(settings.kb_dir),
                policy=PolicyEngine(mode=settings.guard_mode, approval_threshold=settings.approval_threshold),
                policy_text=self.policy_text,
            ),
            checkpointer=self.checkpointer,
        )
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def lock(self, thread_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(thread_id, threading.Lock())

    def set_status(self, thread_id: str, status: str) -> None:
        now = time.time()
        with self.ops.lock:
            self.ops.conn.execute(
                "INSERT INTO threads (id, created, updated, status) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET updated = excluded.updated, status = excluded.status",
                (thread_id, now, now, status),
            )

    def status(self, thread_id: str) -> str | None:
        with self.ops.lock:
            row = self.ops.conn.execute("SELECT status FROM threads WHERE id = ?", (thread_id,)).fetchone()
        return row[0] if row else None


RT: Runtime | None = None


def rt() -> Runtime:
    if RT is None:
        raise HTTPException(503, "service starting")
    return RT


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RT
    settings = Settings.load()
    configure_logging(settings.log_level)
    RT = Runtime(settings)
    log.info("ready", extra={"event": {"model": settings.agent_model, "guard_mode": settings.guard_mode}})
    yield


app = FastAPI(title="support-desk-agent", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=500)
    by: str = Field(default="console", max_length=64)


def _transcript(values: dict[str, Any]) -> list[dict[str, str]]:
    replies = iter(values.get("replies", []))
    out = []
    for m in values.get("messages", []):
        if isinstance(m, HumanMessage):
            out.append({"role": "customer", "text": m.content if isinstance(m.content, str) else ""})
        elif isinstance(m, AIMessage) and not m.tool_calls:
            out.append({"role": "agent", "text": next(replies, "")})
    return out


def _view(thread_id: str) -> dict[str, Any]:
    r = rt()
    values = thread_state(r.graph, thread_id)
    auth = values.get("auth_user_id")
    return {
        "thread_id": thread_id,
        "status": r.status(thread_id) or "new",
        "customer": pseudonym(auth) if auth else None,
        "transcript": _transcript(values),
        "trace": values.get("trace", []),
        "executed": redact(values.get("executed", [])),
        "blocked": values.get("blocked", []),
        "handoff": redact(values.get("handoff")),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    r = rt()
    with r.store._lock:
        orders = r.store.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    return {
        "status": "ok",
        "model": r.settings.agent_model,
        "guard_mode": r.settings.guard_mode,
        "orders": orders,
        "spend_usd": round(r.ledger.total, 4),
        "spend_cap_usd": r.settings.spend_cap_usd,
        "api_key_configured": bool(r.settings.anthropic_api_key),
    }


@app.post("/api/threads")
def new_thread() -> dict[str, str]:
    tid = "th_" + uuid.uuid4().hex[:12]
    rt().set_status(tid, "idle")
    return {"thread_id": tid}


@app.get("/api/threads")
def list_threads() -> list[dict[str, Any]]:
    r = rt()
    with r.ops.lock:
        rows = r.ops.conn.execute(
            "SELECT id, created, updated, status FROM threads ORDER BY updated DESC LIMIT 50"
        ).fetchall()
    return [dict(zip(("thread_id", "created", "updated", "status"), row, strict=True)) for row in rows]


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str) -> dict[str, Any]:
    if rt().status(thread_id) is None:
        raise HTTPException(404, "unknown thread")
    return _view(thread_id)


@app.post("/api/threads/{thread_id}/messages")
def post_message(thread_id: str, body: MessageIn) -> dict[str, Any]:
    r = rt()
    status = r.status(thread_id)
    if status is None:
        raise HTTPException(404, "unknown thread")
    if status == "handed_off":
        raise HTTPException(409, "this conversation was handed to a human agent")
    if status == "awaiting_approval":
        raise HTTPException(409, "waiting for a supervisor to review a refund")
    lock = r.lock(thread_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "a message for this conversation is already being processed")
    try:
        log.info("customer_message", extra={"event": {"thread": thread_id, "text": body.text}})
        turn = send_message(r.graph, thread_id, body.text)
    except BudgetExceeded as e:
        raise HTTPException(503, str(e)) from e
    finally:
        lock.release()
    r.set_status(thread_id, turn.status)
    return {**_view(thread_id), "new_replies": turn.replies, "approval": turn.approval}


@app.get("/api/approvals")
def approvals(status: str | None = "pending") -> list[dict[str, Any]]:
    return rt().ops.approvals(status or None)


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: str, body: DecisionIn) -> dict[str, Any]:
    r = rt()
    appr = r.ops.approval(approval_id)
    if appr is None:
        raise HTTPException(404, "unknown approval")
    if appr["status"] != "pending":
        raise HTTPException(409, f"already {appr['status']}")
    thread_id = appr["thread_id"]
    with r.lock(thread_id):
        try:
            turn = resume_approval(r.graph, thread_id, body.approved, by=body.by, note=body.note)
        except BudgetExceeded as e:
            raise HTTPException(503, str(e)) from e
    r.set_status(thread_id, turn.status)
    return {**_view(thread_id), "new_replies": turn.replies}


@app.get("/api/tickets")
def tickets() -> list[dict[str, Any]]:
    return rt().ops.tickets()


@app.get("/api/audit")
def audit(thread_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    return rt().ops.audit_rows(thread_id, min(limit, 500))


@app.get("/api/examples")
def examples() -> list[dict[str, Any]]:
    """A few synthetic tau-bench customers to try the demo with."""
    r = rt()
    picks = ["mason_li_6934", "harper_khan_9597", "omar_johnson_2562"]
    out = []
    with r.store.transaction() as view:
        for uid in picks:
            if uid not in view.users:
                continue
            u = view.users[uid]
            orders = [
                {
                    "order_id": oid,
                    "status": view.orders[oid]["status"],
                    "total": round(sum(i["price"] for i in view.orders[oid]["items"]), 2),
                }
                for oid in u["orders"]
            ]
            out.append(
                {"name": f"{u['name']['first_name']} {u['name']['last_name']}", "email": u["email"], "orders": orders}
            )
    return out
