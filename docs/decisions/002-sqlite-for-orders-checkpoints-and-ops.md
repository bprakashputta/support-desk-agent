# 002: SQLite for orders, checkpoints and the audit/idempotency tables

**Status:** accepted

## Context
The service stores four kinds of state: the order database (users, orders, products), LangGraph checkpoints (so a conversation survives a restart and an approval can wait for hours), the audit log, and idempotency keys. The brief allowed Postgres or SQLite.

## Decision
SQLite, in WAL mode, on a Docker volume:
- `support_desk.sqlite` holds orders **and** the ops tables (audit log, idempotency keys, approvals, tickets, threads) on **one connection**. A write tool's change to an order, a gift-card balance and its idempotency record commit in a single transaction.
- `checkpoints.sqlite` is LangGraph's `SqliteSaver`.
- Documents are stored as JSON per row, with expression indexes on email and zip for the two identity lookups.

## Why not Postgres
- Exactly-once writes are easiest when the business row and the idempotency row share a transaction. SQLite gives that with no second service, no network hop and no connection pool.
- This is a single-process service with one writer at a time. Writes are serialised per conversation anyway, because a conversation is one graph thread.
- `docker compose up` runs one container, about 200 MB of RAM.

## When to switch
Run more than one API replica, or need concurrent writers across conversations at real volume, and the move is mechanical. `SqliteStore` and `OpsStore` are the only two classes that touch SQL. Swap in `psycopg` with the same per-tool transaction, and use `langgraph-checkpoint-postgres` for checkpoints. JSON documents map to `jsonb`.

## Consequences
Backups are one file per database. A schema change to the documents needs a migration script (none exists yet).
