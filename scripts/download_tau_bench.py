"""Download the tau-bench retail domain (MIT, Sierra Research) into data/tau_bench/.

Pinned to one upstream commit so the database, policy and tasks never drift under the
eval. The task files are Python source; they are parsed with `ast` (never executed) and
written out as JSON.

    uv run python scripts/download_tau_bench.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import httpx

COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
BASE = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{COMMIT}/tau_bench/envs/retail"
FILES = {
    "orders.json": "data/orders.json",
    "products.json": "data/products.json",
    "users.json": "data/users.json",
    "wiki.md": "wiki.md",
    "tasks_test.py": "tasks_test.py",
    "tasks_train.py": "tasks_train.py",
    "LICENSE": None,  # fetched from the repo root below
}


def _literal(node: ast.AST):
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name not in {"Task", "Action"}:
            raise ValueError(f"unexpected call {name!r} in task file")
        return {kw.arg: _literal(kw.value) for kw in node.keywords}
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    return ast.literal_eval(node)


def parse_tasks(source: str) -> list[dict]:
    """Turn `TASKS_X = [Task(...), ...]` into plain dicts without executing the file."""
    tree = ast.parse(source)
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.List):
            return [_literal(e) for e in stmt.value.elts]
    raise ValueError("no task list found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tau_bench")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "github.com/sierra-research/tau-bench", "commit": COMMIT, "files": {}}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name, remote in FILES.items():
            url = (
                f"https://raw.githubusercontent.com/sierra-research/tau-bench/{COMMIT}/LICENSE"
                if remote is None
                else f"{BASE}/{remote}"
            )
            resp = client.get(url)
            resp.raise_for_status()
            body = resp.content
            manifest["files"][name] = hashlib.sha256(body).hexdigest()
            if name.endswith(".py"):
                tasks = parse_tasks(body.decode())
                split = name.removeprefix("tasks_").removesuffix(".py")
                (out / f"tasks_{split}.json").write_text(json.dumps(tasks, indent=1))
                print(f"{split}: {len(tasks)} tasks")
            else:
                (out / name).write_bytes(body)
                print(f"{name}: {len(body):,} bytes")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
