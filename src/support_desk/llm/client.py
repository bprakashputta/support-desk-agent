"""Model access: LangChain's ChatAnthropic behind a thin wrapper that adds what a
production call path needs and LangChain does not do on its own:

- a response cache for identical calls (SQLite, keyed on model + tools + messages);
- a hard spend cap checked before every billed call (see `ledger.py`);
- per-call token / cost / latency records;
- prompt caching: a breakpoint on the system prompt (which also covers the tool list) and
  Anthropic's top-level automatic breakpoint on the newest message, so each turn of a long
  conversation re-reads the earlier turns from cache instead of paying for them again.

Retries (429 / 5xx / connection errors) and timeouts are the SDK's, configured here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)

from .ledger import SpendLedger
from .pricing import cost_usd

log = logging.getLogger("support_desk.llm")


@dataclass
class CallStats:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    by_tag: dict[str, float] = field(default_factory=dict)

    def add(self, rec: dict[str, Any]) -> None:
        self.calls += 1
        self.cached_calls += int(rec["cached"])
        self.input_tokens += rec["input_tokens"]
        self.cache_read += rec["cache_read"]
        self.cache_write += rec["cache_write"]
        self.output_tokens += rec["output_tokens"]
        self.cost_usd += rec["cost_usd"]
        self.latency_s += rec["latency_ms"] / 1000
        self.by_tag[rec["tag"]] = self.by_tag.get(rec["tag"], 0.0) + rec["cost_usd"]


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, msg TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> AIMessage | None:
        with self._lock:
            row = self._conn.execute("SELECT msg FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        msg = messages_from_dict([json.loads(row[0])])[0]
        return msg if isinstance(msg, AIMessage) else None

    def put(self, key: str, msg: AIMessage) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, msg) VALUES (?, ?)",
                (key, json.dumps(message_to_dict(msg))),
            )


def _canonical(msg: BaseMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"type": msg.type, "content": msg.content}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        out["tool_calls"] = [{"name": t["name"], "args": t["args"], "id": t["id"]} for t in msg.tool_calls]
    if getattr(msg, "tool_call_id", None):
        out["tool_call_id"] = msg.tool_call_id
    return out


class LLMClient:
    def __init__(
        self,
        model: str,
        ledger: SpendLedger,
        *,
        cache: ResponseCache | None = None,
        max_tokens: int = 4096,
        effort: str | None = None,
        timeout_s: float = 90.0,
        max_retries: int = 4,
        api_key: str | None = None,
    ):
        self.model = model
        self.ledger = ledger
        self.cache = cache
        self.max_tokens = max_tokens
        self.effort = effort
        self.stats = CallStats()
        self._stats_lock = threading.Lock()
        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        self._chat = ChatAnthropic(
            model=model,
            max_tokens=max_tokens,
            timeout=timeout_s,
            max_retries=max_retries,
            api_key=api_key,
            **kwargs,
        )

    def _key(self, messages: Sequence[BaseMessage], tools: list[dict] | None) -> str:
        blob = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "effort": self.effort,
                "tools": tools or [],
                "messages": [_canonical(m) for m in messages],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _with_system_breakpoint(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        out = list(messages)
        if out and isinstance(out[0], SystemMessage) and isinstance(out[0].content, str):
            out[0] = SystemMessage(
                content=[{"type": "text", "text": out[0].content, "cache_control": {"type": "ephemeral"}}]
            )
        return out

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: list[dict] | None = None,
        tag: str = "agent",
        meta: dict[str, Any] | None = None,
    ) -> AIMessage:
        key = self._key(messages, tools)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self._record(tag, hit, 0.0, cached=True, meta=meta)
            return hit

        self.ledger.check()
        runnable = self._chat.bind_tools(tools, parallel_tool_calls=False) if tools else self._chat
        started = time.perf_counter()
        msg = runnable.invoke(self._with_system_breakpoint(messages), cache_control={"type": "ephemeral"})
        latency = time.perf_counter() - started
        assert isinstance(msg, AIMessage)
        self._record(tag, msg, latency, cached=False, meta=meta)
        if self.cache is not None:
            self.cache.put(key, msg)
        return msg

    def _record(self, tag: str, msg: AIMessage, latency: float, *, cached: bool, meta: dict[str, Any] | None) -> None:
        usage = (msg.response_metadata or {}).get("usage") or {}
        rec = {
            "model": self.model,
            "tag": tag,
            "cached": cached,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "cache_read": int(usage.get("cache_read_input_tokens") or 0),
            "cache_write": int(usage.get("cache_creation_input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "latency_ms": round(latency * 1000),
        }
        rec["cost_usd"] = (
            0.0
            if cached
            else round(
                cost_usd(
                    self.model,
                    input_tokens=rec["input_tokens"],
                    output_tokens=rec["output_tokens"],
                    cache_read=rec["cache_read"],
                    cache_write=rec["cache_write"],
                ),
                6,
            )
        )
        if meta:
            rec.update(meta)
        if not cached:
            self.ledger.record(rec)
        with self._stats_lock:
            self.stats.add(rec)
        log.info("llm_call", extra={"event": rec})
