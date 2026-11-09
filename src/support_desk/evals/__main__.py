"""Eval CLI.

    uv run python -m support_desk.evals tau --split test --config guarded
    uv run python -m support_desk.evals tau --split test --config prompt_only
    uv run python -m support_desk.evals redteam --config guarded
    uv run python -m support_desk.evals report

Per-conversation results go to eval/results/<run>.jsonl (committed); full transcripts to
data/eval_runs/<run>/ (gitignored). Re-running a command resumes: finished tasks are
skipped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import ROOT, Settings
from ..llm.ledger import BudgetExceeded, SpendLedger
from .harness import Env, RunConfig, grade_tau, run_conversation
from .redteam import SCENARIOS_PATH, grade_redteam, load_scenarios

RESULTS = ROOT / "eval" / "results"

CONFIGS = {
    "prompt_only": dict(guard_mode="audit"),
    "guarded": dict(guard_mode="enforce"),
}
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}


def _cfg(args: argparse.Namespace, settings: Settings) -> RunConfig:
    return RunConfig(
        name=args.run or f"{args.kind}-{args.split if args.kind == 'tau' else 'rt'}-{args.config}-{args.model}",
        agent_model=MODELS[args.model],
        user_model=settings.user_model,
        approval_threshold=settings.approval_threshold,
        **CONFIGS[args.config],
    )


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["task_id"] for line in path.read_text().splitlines() if line.strip()}


def _strip(conv: dict) -> dict:
    return {k: v for k, v in conv.items() if k not in {"store", "transcript", "replies"}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="support_desk.evals")
    sub = p.add_subparsers(dest="kind", required=True)
    for kind in ("tau", "redteam"):
        s = sub.add_parser(kind)
        s.add_argument("--config", choices=CONFIGS, default="guarded")
        s.add_argument("--model", choices=MODELS, default="haiku")
        s.add_argument("--split", default="test", choices=["test", "train"])
        s.add_argument("--n", type=int, default=0, help="0 = every task in the split")
        s.add_argument("--offset", type=int, default=0, help="skip the first k sampled tasks")
        s.add_argument("--seed", type=int, default=0, help="0 = tasks in file order; else a seeded sample")
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--run", default=None)
    sub.add_parser("report")
    args = p.parse_args(argv)

    if args.kind == "report":
        from .report import write_report

        write_report(RESULTS)
        return 0

    settings = Settings.load()
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    ledger = SpendLedger(settings.data_dir / "spend.jsonl", settings.spend_cap_usd)
    env = Env(settings.tau_dir, settings.kb_dir)
    cfg = _cfg(args, settings)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{cfg.name}.jsonl"
    tdir = settings.data_dir / "eval_runs" / cfg.name
    tdir.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    lock = threading.Lock()

    if args.kind == "tau":
        tasks = list(enumerate(env.tasks(args.split)))
        if args.seed:
            random.Random(args.seed).shuffle(tasks)
        tasks = tasks[args.offset :]
        if args.n:
            tasks = tasks[: args.n]
        jobs = [(f"{args.split}-{i}", i, t) for i, t in tasks]
    else:
        jobs = [(s["id"], None, s) for s in load_scenarios(SCENARIOS_PATH)]
        if args.n:
            jobs = jobs[: args.n]
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{cfg.name}: {len(jobs)} conversations to run ({len(done)} already done); spent so far ${ledger.total:.4f}")

    def work(job):
        task_id, idx, item = job
        if args.kind == "tau":
            conv = run_conversation(
                env, cfg, ledger, task_id=task_id, instruction=item["instruction"], api_key=settings.anthropic_api_key
            )
            grade = grade_tau(env, args.split, idx, item, conv)
        else:
            conv = run_conversation(
                env,
                cfg,
                ledger,
                task_id=task_id,
                instruction=item["instruction"],
                opening=item.get("opening"),
                api_key=settings.anthropic_api_key,
            )
            grade = grade_redteam(env, item, conv)
        row = {
            "task_id": task_id,
            "config": cfg.name,
            "agent_model": cfg.agent_model,
            "user_model": cfg.user_model,
            "guard_mode": cfg.guard_mode,
            **grade,
            **_strip(conv),
        }
        (tdir / f"{task_id}.json").write_text(json.dumps({"transcript": conv["transcript"], **row}, indent=1))
        with lock, out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    passed = total = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, j) for j in jobs]
            for fut in as_completed(futures):
                row = fut.result()
                total += 1
                passed += int(row.get("reward", row.get("defended", 0)))
                print(
                    f"[{total}/{len(jobs)}] {row['task_id']}: reward={row.get('reward', row.get('defended'))} "
                    f"turns={len(row.get('executed', []))}w end={row['end_reason']} "
                    f"cost=${row['agent_cost_usd'] + row['user_cost_usd']:.4f} total=${ledger.total:.4f}",
                    flush=True,
                )
    except BudgetExceeded as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 3
    print(f"{cfg.name}: {passed}/{total} this session; ledger total ${ledger.total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
