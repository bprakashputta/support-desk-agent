"""Operational records: audit log, idempotency keys, approval requests, handoff tickets.

In the service these tables live in the same SQLite file and connection as the orders, so
a write tool's database change and its idempotency record commit in one transaction:
either both are there or neither is. That is what makes a replayed tool call (a crash
between the tool and the checkpoint, a retried HTTP request, a model repeating itself) a
no-op instead of a second refund.

Everything stored here is redacted with `pii.redact` first.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from .pii import redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    thread_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    tool TEXT,
    target TEXT,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_thread ON audit_log(thread_id, id);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    result TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    target TEXT,
    amount REAL NOT NULL,
    args TEXT NOT NULL,
    status TEXT NOT NULL,
    decided_by TEXT,
    note TEXT,
    created REAL NOT NULL,
    decided REAL
);
CREATE INDEX IF NOT EXISTS approvals_status ON approvals(status, created);
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    customer TEXT,
    summary TEXT NOT NULL,
    context TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL NOT NULL
);
"""


def idempotency_key(thread_id: str, tool: str, args: dict[str, Any]) -> str:
    blob = json.dumps({"t": thread_id, "tool": tool, "args": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class OpsStore:
    def __init__(self, conn: sqlite3.Connection | None = None, lock: threading.RLock | None = None):
        self.conn = conn or sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.lock = lock or threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    # -- audit ---------------------------------------------------------------
    def audit(
        self,
        thread_id: str,
        event: str,
        *,
        actor: str = "agent",
        tool: str | None = None,
        target: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO audit_log (ts, thread_id, actor, event, tool, target, detail) VALUES (?,?,?,?,?,?,?)",
                (time.time(), thread_id, actor, event, tool, redact(target), json.dumps(redact(detail or {}))),
            )

    def audit_rows(self, thread_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        q = "SELECT id, ts, thread_id, actor, event, tool, target, detail FROM audit_log"
        params: tuple = ()
        if thread_id:
            q += " WHERE thread_id = ?"
            params = (thread_id,)
        q += " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = self.conn.execute(q, (*params, limit)).fetchall()
        cols = ["id", "ts", "thread_id", "actor", "event", "tool", "target", "detail"]
        out = [dict(zip(cols, r, strict=True)) for r in rows]
        for r in out:
            r["detail"] = json.loads(r["detail"])
        return out

    # -- idempotency ---------------------------------------------------------
    def idem_get(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT result FROM idempotency WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def idem_put(self, key: str, thread_id: str, tool: str, result: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO idempotency (key, thread_id, tool, result, ts) VALUES (?,?,?,?,?)",
                (key, thread_id, tool, result, time.time()),
            )

    # -- approvals -----------------------------------------------------------
    def approval_open(
        self, approval_id: str, thread_id: str, tool: str, target: str, amount: float, args: dict
    ) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO approvals (id, thread_id, tool, target, amount, args, status, created) "
                "VALUES (?,?,?,?,?,?, 'pending', ?)",
                (approval_id, thread_id, tool, target, amount, json.dumps(redact(args)), time.time()),
            )

    def approval_decide(self, approval_id: str, status: str, by: str, note: str = "") -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE approvals SET status = ?, decided_by = ?, note = ?, decided = ? WHERE id = ?",
                (status, by, note, time.time(), approval_id),
            )

    def approvals(self, status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, thread_id, tool, target, amount, args, status, decided_by, note, created, decided FROM approvals"
        params: tuple = ()
        if status:
            q += " WHERE status = ?"
            params = (status,)
        with self.lock:
            rows = self.conn.execute(q + " ORDER BY created DESC LIMIT 100", params).fetchall()
        cols = [
            "id",
            "thread_id",
            "tool",
            "target",
            "amount",
            "args",
            "status",
            "decided_by",
            "note",
            "created",
            "decided",
        ]
        out = [dict(zip(cols, r, strict=True)) for r in rows]
        for r in out:
            r["args"] = json.loads(r["args"])
        return out

    def approval(self, approval_id: str) -> dict[str, Any] | None:
        return next((a for a in self.approvals() if a["id"] == approval_id), None)

    # -- tickets -------------------------------------------------------------
    def ticket_create(self, thread_id: str, customer: str | None, summary: str, context: dict) -> str:
        tid = "T-" + uuid.uuid4().hex[:8].upper()
        with self.lock:
            self.conn.execute(
                "INSERT INTO tickets (id, thread_id, customer, summary, context, status, created) VALUES (?,?,?,?,?, 'open', ?)",
                (
                    tid,
                    thread_id,
                    redact(customer, "user_id") if customer else None,
                    redact(summary),
                    json.dumps(redact(context)),
                    time.time(),
                ),
            )
        return tid

    def tickets(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, thread_id, customer, summary, context, status, created FROM tickets ORDER BY created DESC LIMIT 100"
            ).fetchall()
        cols = ["id", "thread_id", "customer", "summary", "context", "status", "created"]
        out = [dict(zip(cols, r, strict=True)) for r in rows]
        for r in out:
            r["context"] = json.loads(r["context"])
        return out
