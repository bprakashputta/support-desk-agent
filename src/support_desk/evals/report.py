"""Aggregate eval/results/*.jsonl into eval/results/summary.json and eval/RESULTS.md."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from ..config import ROOT


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _boot_ci(values: list[float], n: int = 5000, seed: int = 7) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    cost = [r["agent_cost_usd"] + r["user_cost_usd"] for r in rows]
    wall = [r["wall_s"] for r in rows]
    per_call = [r["agent_latency_s"] / r["agent_calls"] for r in rows if r["agent_calls"]]
    out: dict[str, Any] = {
        "n": n,
        "agent_model": rows[0]["agent_model"],
        "user_model": rows[0]["user_model"],
        "guard_mode": rows[0]["guard_mode"],
        "policy_violation_rate": sum(1 for r in rows if r["violations"]) / n,
        "violating_writes": sum(len(r["violations"]) for r in rows),
        "writes": sum(len(r["executed"]) for r in rows),
        "guard_block_rate": sum(1 for r in rows if r["blocked"]) / n,
        "handoff_rate": sum(1 for r in rows if r["handoff"]) / n,
        "approval_rate": sum(1 for r in rows if r["approvals"]) / n,
        "errors": sum(1 for r in rows if r["error"]),
        "cost_per_conv_usd": statistics.fmean(cost),
        "agent_cost_per_conv_usd": statistics.fmean(r["agent_cost_usd"] for r in rows),
        "user_sim_cost_per_conv_usd": statistics.fmean(r["user_cost_usd"] for r in rows),
        "total_cost_usd": sum(cost),
        "wall_s_mean": statistics.fmean(wall),
        "wall_s_p50": _pct(wall, 0.5),
        "wall_s_p90": _pct(wall, 0.9),
        "agent_call_latency_s_mean": statistics.fmean(per_call) if per_call else 0.0,
        "agent_calls_per_conv": statistics.fmean(r["agent_calls"] for r in rows),
    }
    if "reward" in rows[0]:
        rewards = [r["reward"] for r in rows]
        out["pass_1"] = statistics.fmean(rewards)
        out["pass_1_ci95"] = _boot_ci(rewards)
        out["db_match_rate"] = statistics.fmean(float(r["r_db"]) for r in rows)
    if "defended" in rows[0]:
        out["defended_rate"] = statistics.fmean(r["defended"] for r in rows)
        out["attacks_succeeded"] = [r["task_id"] for r in rows if r["attack_succeeded"]]
    return out


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    ra = {r["task_id"]: r for r in a}
    rb = {r["task_id"]: r for r in b}
    common = sorted(ra.keys() & rb.keys())

    def side(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
        sel = [rows[t] for t in common]
        return {
            "pass_1": statistics.fmean(r["reward"] for r in sel),
            "cost_per_conv_usd": statistics.fmean(r["agent_cost_usd"] + r["user_cost_usd"] for r in sel),
            "agent_cost_per_conv_usd": statistics.fmean(r["agent_cost_usd"] for r in sel),
            "wall_s_mean": statistics.fmean(r["wall_s"] for r in sel),
            "agent_call_latency_s_mean": statistics.fmean(
                r["agent_latency_s"] / r["agent_calls"] for r in sel if r["agent_calls"]
            ),
        }

    return {
        "n": len(common),
        "a": side(ra),
        "b": side(rb),
        "a_only_pass": sum(1 for t in common if ra[t]["reward"] > rb[t]["reward"]),
        "b_only_pass": sum(1 for t in common if rb[t]["reward"] > ra[t]["reward"]),
    }


def _f(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(results: Path) -> None:
    runs = {p.stem: _rows(p) for p in sorted(results.glob("*.jsonl"))}
    summary = {name: summarise(rows) for name, rows in runs.items() if rows}
    pairs = {}
    for model in ("haiku", "sonnet"):
        a, b = f"tau-test-prompt_only-{model}", f"tau-test-guarded-{model}"
        if a in runs and b in runs:
            pairs[f"{a} vs {b}"] = paired(runs[a], runs[b])
    if "tau-test-guarded-haiku" in runs and "tau-test30-guarded-sonnet" in runs:
        pairs["tau-test-guarded-haiku vs tau-test30-guarded-sonnet"] = paired(
            runs["tau-test-guarded-haiku"], runs["tau-test30-guarded-sonnet"]
        )
    ledger = ROOT / "data" / "spend.jsonl"
    total_spend = (
        sum(json.loads(line).get("cost_usd", 0) for line in ledger.read_text().splitlines() if line.strip())
        if ledger.exists()
        else None
    )
    (results / "summary.json").write_text(
        json.dumps({"runs": summary, "paired": pairs, "total_spend_usd_all_runs": total_spend}, indent=2)
    )

    lines = ["# Eval results", "", "Generated by `uv run python -m support_desk.evals report`.", ""]
    tau = {k: v for k, v in summary.items() if "pass_1" in v}
    if tau:
        lines += [
            "## tau-bench retail",
            "",
            "| Run | N | pass^1 (95% CI) | Policy-violating convs | Guard blocks | Handoff | Needed approval "
            "| Cost / conv | Wall time p50 / p90 | Agent call latency |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in tau.items():
            lo, hi = s["pass_1_ci95"]
            lines.append(
                f"| `{name}` | {s['n']} | {_f(s['pass_1'])} ({_f(lo)}–{_f(hi)}) | {_f(s['policy_violation_rate'])} "
                f"({s['violating_writes']}/{s['writes']} writes) | {_f(s['guard_block_rate'])} "
                f"| {_f(s['handoff_rate'])} | {_f(s['approval_rate'])} | ${s['cost_per_conv_usd']:.4f} "
                f"| {s['wall_s_p50']:.0f}s / {s['wall_s_p90']:.0f}s | {s['agent_call_latency_s_mean']:.2f}s |"
            )
        lines.append("")
    if pairs:
        lines += [
            "### Paired comparisons (same tasks)",
            "",
            "| A vs B | Tasks | pass^1 A | pass^1 B | Only A passed | Only B passed | Cost / conv A | Cost / conv B "
            "| Agent call latency A / B |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k, p in pairs.items():
            a, b = p["a"], p["b"]
            lines.append(
                f"| `{k}` | {p['n']} | {_f(a['pass_1'])} | {_f(b['pass_1'])} | {p['a_only_pass']} | {p['b_only_pass']} "
                f"| ${a['cost_per_conv_usd']:.4f} | ${b['cost_per_conv_usd']:.4f} "
                f"| {a['agent_call_latency_s_mean']:.2f}s / {b['agent_call_latency_s_mean']:.2f}s |"
            )
    rt = {k: v for k, v in summary.items() if "defended_rate" in v}
    if rt:
        lines += [
            "",
            "## Red team (prompt injection / social engineering)",
            "",
            "| Run | Scenarios | Attacks defended | Attacks that succeeded | Cost / conv |",
            "|---|---|---|---|---|",
        ]
        for name, s in rt.items():
            lines.append(
                f"| `{name}` | {s['n']} | {_f(s['defended_rate'])} | "
                f"{', '.join(s['attacks_succeeded']) or 'none'} | ${s['cost_per_conv_usd']:.4f} |"
            )
    if total_spend is not None:
        lines += ["", f"Total API spend recorded in the ledger (all runs, including development): ${total_spend:.2f}."]
    (results.parent / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
