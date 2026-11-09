"""Spend ledger with a hard cap.

Every model call appends one JSON line (tokens, cost, latency, tag). Before a call the
client asks `check()`, which re-reads lines other processes appended since the last check,
so the eval runner and the API server share one budget. Past the cap, calls are refused.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, path: str | Path, cap_usd: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._offset = 0
        self._total = 0.0
        self._refresh()

    def _refresh(self) -> None:
        with self.path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        if not chunk:
            return
        end = chunk.rfind(b"\n") + 1  # ignore a partially written last line
        for line in chunk[:end].splitlines():
            if line.strip():
                self._total += float(json.loads(line).get("cost_usd", 0.0))
        self._offset += end

    @property
    def total(self) -> float:
        with self._lock:
            self._refresh()
            return self._total

    def check(self, reserve_usd: float = 0.0) -> None:
        with self._lock:
            self._refresh()
            if self._total + reserve_usd >= self.cap_usd:
                raise BudgetExceeded(f"spend cap reached: ${self._total:.4f} spent of ${self.cap_usd:.2f}")

    def record(self, entry: dict[str, Any]) -> None:
        entry = {"ts": round(time.time(), 3), **entry}
        line = (json.dumps(entry) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(fd, line)  # single write of a short line: atomic on POSIX append
            finally:
                os.close(fd)
            self._refresh()
