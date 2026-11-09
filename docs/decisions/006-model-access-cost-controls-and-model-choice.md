# 006: Model access, cost controls and the Haiku-vs-Sonnet choice

**Status:** accepted

## Context
The owner's convention is to reach models through LangChain interfaces. The brief requires a hard spend cap, caching of identical calls, per-call token, cost and latency logging, retries and timeouts.

## Decision
- `LLMClient` wraps `langchain_anthropic.ChatAnthropic`. Retries (4, with backoff on 429/5xx/connection errors) and timeouts (90 s) come from the SDK.
- **Hard cap:** an append-only JSONL ledger is checked before every billed call, and calls are refused at `SPEND_CAP_USD` ($5). The ledger re-reads lines other processes appended, so the eval runner and the API server share one budget.
- **Identical-call cache:** SQLite, keyed on model, effort, tools and the canonical messages. A hit returns the stored `AIMessage` and is logged at $0. It is on in the service and off in the eval, so conversations stay independent samples.
- **Prompt caching:** a breakpoint on the system prompt (which also covers the tool list), plus Anthropic's top-level automatic breakpoint on the newest message. Each turn of a long tool loop then re-reads the earlier turns from cache.
- **Thinking blocks are never edited:** Claude 5.5 models bind thinking blocks to the exact message. The citation filter (ADR 004) works on a copy for the customer and never rewrites history.
- **Models:** `claude-haiku-5-5` is the agent by default and is always the user simulator.

## Evidence
- Prompt caching: across all runs, 95.5% of the Haiku agent's prompt tokens were cache reads. The agent's spend was $0.97, against an estimated $4.31 at uncached rates (4.4x cheaper). Without caching, this eval would not have fit in the $5 budget.
- On the same 30 test tasks (seeded sample), guarded configuration:

| | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| pass^1 | 70.0% | 76.7% |
| only this model passed | 2 tasks | 4 tasks |
| cost / conversation (agent + simulator) | $0.0038 | $0.0580 |
| mean model-call latency | 1.27 s | 1.56 s |

Sonnet is 15x the cost for +6.7 points on 30 tasks. With 2-vs-4 discordant tasks, that gap cannot be told apart from noise. Haiku stays the default. Sonnet is one env var away (`AGENT_MODEL`) for a deployment that values the last few points over cost.

## Consequences
The user simulator is Haiku, not the GPT-4o tau-bench used originally, and the Claude 5.5 API has no temperature setting. Absolute pass^1 is therefore not directly comparable to published tau-bench leaderboards (see README, Limitations).
