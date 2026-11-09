"""PII redaction for logs, the audit trail and anything shown in the agent console.

Two layers:
- key-based: in structured data, values under known PII keys (email, address lines, zip,
  names, card last-four) are replaced outright;
- pattern-based: free text (chat messages, tool output, model summaries) is scrubbed of
  emails, phone numbers, street addresses, 5-digit zip codes, card numbers and
  tau-bench-style user ids (`first_last_1234`, which embed a real-looking name).

User ids become a stable pseudonym (`user:3f9a1c2b`) so a log reader can still follow one
customer through a conversation without learning who they are.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

PII_KEYS = {
    "email",
    "address1",
    "address2",
    "zip",
    "first_name",
    "last_name",
    "last_four",
    "phone",
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s){1,3}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct|Place|Pl|Park|Terrace)\b\.?"
)
_SUITE = re.compile(r"\b(?:Suite|Apt|Unit)\s+\d+\b", re.IGNORECASE)
_ZIP = re.compile(r"(?<![\d#])\d{5}(?![\d])")
_USER_ID = re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")


def pseudonym(user_id: str) -> str:
    return "user:" + hashlib.sha256(user_id.encode()).hexdigest()[:8]


def redact_text(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _USER_ID.sub(lambda m: pseudonym(m.group(0)), text)
    text = _STREET.sub("[street]", text)
    text = _SUITE.sub("[unit]", text)
    text = _PHONE.sub("[phone]", text)
    text = _CARD.sub(lambda m: m.group(0) if len(re.sub(r"\D", "", m.group(0))) == 10 else "[card]", text)
    text = _ZIP.sub("[zip]", text)
    return text


def redact(value: Any, key: str | None = None) -> Any:
    """Recursively redact a JSON-like value."""
    if key in PII_KEYS and value not in (None, ""):
        return "[redacted]"
    if key == "user_id" and isinstance(value, str):
        return pseudonym(value)
    if isinstance(value, dict):
        if key == "name" and {"first_name", "last_name"} & value.keys():
            return "[redacted]"
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
