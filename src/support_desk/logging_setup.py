"""JSON logs, one object per line, with PII redacted before anything is written."""

from __future__ import annotations

import json
import logging
import sys
import time

from .pii import redact, redact_text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": redact_text(record.getMessage()),
        }
        event = getattr(record, "event", None)
        if event is not None:
            out["event"] = redact(event)
        if record.exc_info:
            out["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
