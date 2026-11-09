"""Help-center retrieval with citations.

The knowledge base is a folder of markdown articles; each `##` section is one passage with
a stable id (`returns#where-the-refund-goes`). Retrieval is BM25 over those passages. The
corpus is a few dozen passages, so a lexical index is exact enough, runs in microseconds
and needs no embedding API (see docs/decisions/004).

Citations are checked in code: `validate_citations` keeps only `[kb:...]` markers whose id
was actually returned by a search earlier in the conversation and drops the rest, so the
customer never sees a citation the model made up.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "can",
        "i",
        "my",
        "me",
        "you",
        "your",
        "we",
        "our",
        "it",
        "this",
        "that",
        "with",
        "do",
        "does",
        "how",
        "what",
        "when",
        "if",
        "at",
        "by",
        "from",
        "as",
        "was",
        "will",
        "would",
    ]
)
CITATION = re.compile(r"\[kb:([a-z0-9-]+#[a-z0-9-]+)\]")


def _tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") else t)  # crude plural folding
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Passage:
    id: str
    article: str
    heading: str
    text: str


class HelpCenter:
    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self._docs = [Counter(_tokens(f"{p.article} {p.heading} {p.heading} {p.text}")) for p in passages]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(passages)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.by_id = {p.id: p for p in passages}

    @classmethod
    def from_dir(cls, root: Path) -> HelpCenter:
        passages: list[Passage] = []
        for path in sorted(root.glob("*.md")):
            article, heading, buf = path.stem, "", []
            title = path.stem
            for line in [*path.read_text().splitlines(), "## "]:
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## "):
                    if heading and buf:
                        passages.append(Passage(f"{article}#{_slug(heading)}", title, heading, " ".join(buf).strip()))
                    heading, buf = line[3:].strip(), []
                elif line.strip():
                    buf.append(line.strip())
        return cls(passages)

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        q = _tokens(query)
        scores = []
        for i, doc in enumerate(self._docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
            for t in q:
                f = doc.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.passages[i], round(s, 3)) for s, i in scores[:k]]

    def render(self, query: str, k: int = 3) -> tuple[str, list[str]]:
        hits = self.search(query, k)
        if not hits:
            return "No help-center articles matched. Do not answer from memory; say you are not sure.", []
        lines = [f"[kb:{p.id}] {p.article} > {p.heading}: {p.text}" for p, _ in hits]
        return "\n".join(lines), [p.id for p, _ in hits]


def validate_citations(text: str, allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Strip citations that were never retrieved. Returns (text, kept, dropped)."""
    kept, dropped = [], []

    def _sub(m: re.Match[str]) -> str:
        cid = m.group(1)
        if cid in allowed:
            kept.append(cid)
            return m.group(0)
        dropped.append(cid)
        return ""

    cleaned = CITATION.sub(_sub, text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), kept, dropped
