# 005: Human approval through LangGraph interrupts; idempotent writes

**Status:** accepted

## Context
Some actions should not be fully automatic, however good the model is. Here that means refunds above a threshold (default $1,000), from a cancellation or a return. An approval can take minutes or hours, the process can restart in between, and an HTTP retry or a model that repeats itself must never refund twice.

## Decision
- When a write passes R1–R3 but its refund exceeds the threshold, the guard opens an approval record and routes to an `approval` node that calls `interrupt()`. The graph state is checkpointed (SQLite, ADR 002) and the thread is parked. The customer sees "a team member is reviewing this refund". The console lists pending approvals, and `POST /api/approvals/{id}` resumes the thread with `Command(resume=...)`.
- On approve, the graph continues to `tools`. On reject, the model gets a `Policy:` tool result with the reviewer's note and tells the customer.
- R4 applies in both eval configurations: it is architecture, not a prompt experiment. The eval resumes with an auto-approver and reports how often approval was needed (13.9% and 18.3% of test conversations).
- Every write computes an idempotency key: sha256 of thread + tool + canonical arguments. The key is stored in the same transaction as the write (ADR 002). A replay returns the stored result and logs `write_replayed`, without running the tool again.
- The refund amount is computed in code from the order (`refund_amount()`), never taken from the model. There is no free-form "refund" tool at all: money only moves as part of a cancel, return, exchange or item change that the tools price themselves. The most-targeted tool in a support bot does not exist here.

## Evidence
Tests cover approve, reject (order untouched, note passed to the model), a repeated identical write (one refund, one `write_replayed`) and the HTTP flow (409 while a review is pending, 409 on deciding twice). In the red-team set, the two "skip the human review" scenarios (a claimed manager pre-approval, and splitting a $5.5K return into small ones) both still went to approval in both configurations.

## Consequences
The approval record and the interrupt are two pieces of state. If a checkpoint were deleted by hand, an approval row could be left pending with no thread to resume; the API returns 409/404 rather than guessing.
