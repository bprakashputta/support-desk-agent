# 003: Enforce policy in code, between the model's tool call and the tool

**Status:** accepted

## Context
The policy is in the system prompt, but a prompt is a request, not a control. A customer can talk the model out of it, or the model can just slip. Some rules are already enforced by the tau-bench tools (order status, refund destination, gift-card balance, "exchange once"). Others are not checked anywhere: who the customer is, whether the order is theirs, and whether they actually said yes.

## Decision
A `guard` node sits between `agent` and `tools` in the graph. Every tool call passes through it. It does two things:
1. Validates the arguments against the tool's pydantic model. Invalid calls go back to the model as an error it can fix.
2. Runs `PolicyEngine.check`:
   - **R1 auth_required:** account and order reads, and every write, need a customer verified by email or name + zip in this conversation.
   - **R2 single_customer:** only that customer's profile and orders. A second identity is refused.
   - **R3 explicit_confirmation:** a write needs an affirmative latest customer message, given right after an agent message that named the order id (or the new zip, for a profile address change). One "yes" covers each (tool, target) once.
   - **R4 refund_approval:** see ADR 005.

Blocked calls return `Policy: <rule>: <why>` as the tool result, so the model can recover by asking for confirmation or verification. There is also a mode switch: `enforce` (block) or `audit` (allow, but record the violation). Audit mode is the "prompt-only" baseline in the eval, and it is how violations are counted when nothing stops them.

## Evidence (tau-bench retail test split, N=115 each, Claude Haiku 5.5)
| | prompt-only (audit) | code-enforced |
|---|---|---|
| pass^1 | 75.7% | 73.9% |
| conversations with a policy-violating write | 6.1% (8 of 165 writes) | **0%** (0 of 158) |
| conversations where a guard blocked a call | n/a | 1.7% |
| cost / conversation | $0.0041 | $0.0042 |

All 8 violations in the prompt-only run were R3: the agent acted on a "yes" to a message that never restated the order id. None were unverified or cross-customer writes. The success difference (12 tasks passed only without guards, 10 only with) is within noise; the 95% CIs overlap almost entirely.

## Consequences
- The guard is the floor that holds no matter what the model does. It did not make the agent better at the benchmark, and it was never meant to.
- R3 is a heuristic: a regex for "yes" plus the order id in the agent's previous message. On the train split it produced one false block ("No longer needed. Yes, cancel it." was read as a refusal), which was fixed before the test runs. Its strictness costs an extra confirmation turn when the agent asks for confirmation without restating the order.
- The eval's violation counter uses the same rules as the guard. In enforce mode it is therefore 0 by construction, and the meaningful number is the audit-mode rate.
