# 001: Port the tau-bench retail domain instead of importing the package

**Status:** accepted

## Context
The eval is tau-bench retail (Sierra Research, MIT): a 500-user / 1,000-order database, 15 tools, a written policy and 115 test tasks graded by final database hash. The `tau_bench` package pulls in litellm, openai, mistralai and google-generativeai, runs its own agent loop, and its tools are untyped static methods over a dict.

The service needs the same tools behind a policy layer, against SQLite, with typed arguments.

## Decision
- `scripts/download_tau_bench.py` fetches the data, policy and task files from one pinned commit (`59a200c`) into the gitignored `data/` directory. Task files are Python source; they are parsed with `ast` and never executed (a test checks that a planted call is rejected).
- The tool logic is ported into `domain/tools.py` with a pydantic model per tool. The same class generates the JSON schema the model sees and validates the arguments it sends.
- Return strings and database mutations are kept identical to upstream so the benchmark's grading still applies.

## Evidence
Replaying every ground-truth action of all 615 tasks (test + train) through the ported tools and through upstream's own code gave **0 hash mismatches and 0 output differences**. The 115 upstream test-split hashes are committed as a fixture, and CI checks them on every push.

## Consequences
- No dependency on litellm or the other provider SDKs. The agent uses one model client (`langchain-anthropic`).
- If upstream changes its tools, the pin keeps this repo on the old version until someone updates the commit and the fixture.
