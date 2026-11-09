"""Order database: one interface, two backends.

`MemoryStore` holds the tau-bench JSON as plain dicts. The eval uses it because the
benchmark's ground truth is a hash of the whole database after the conversation.

`SqliteStore` is what the service runs on: one table per entity with the document as JSON,
an expression index for email lookup, and writes wrapped in a transaction so a tool that
touches an order *and* a gift-card balance commits both or neither.

Tools never talk to a backend directly. They get a `DataView`: mappings that load documents
on first access and remember them, so the ported tau-bench logic (`data["orders"][id]...`)
runs unchanged on either backend, and the store writes back only what the tool loaded.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

TABLES = ("users", "orders", "products")


def to_hashable(item: Any) -> Any:
    """Same canonicalisation tau-bench uses for its ground-truth database hash."""
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(e) for e in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(e) for e in item))
    return item


def data_hash(data: Mapping[str, Mapping[str, Any]]) -> str:
    return sha256(str(to_hashable({k: dict(v) for k, v in data.items()})).encode()).hexdigest()


def load_tau_json(root: Path) -> dict[str, dict[str, Any]]:
    return {t: json.loads((root / f"{t}.json").read_text()) for t in TABLES}


class DataView(Protocol):
    users: Mapping[str, dict]
    orders: Mapping[str, dict]
    products: Mapping[str, dict]

    def __getitem__(self, table: str) -> Mapping[str, dict]: ...
    def find_user_by_email(self, email: str) -> str | None: ...
    def find_user_by_name_zip(self, first: str, last: str, zip_code: str) -> str | None: ...
    def product_index(self) -> dict[str, str]: ...


class Store(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[DataView]: ...


# --------------------------------------------------------------------------- memory


class _CopyOnReadTable(Mapping[str, dict]):
    """Hands tools private copies of documents; `commit()` writes them back."""

    def __init__(self, backing: dict[str, dict]):
        self._backing = backing
        self.loaded: dict[str, dict] = {}

    def __getitem__(self, key: str) -> dict:
        if key not in self.loaded:
            self.loaded[key] = copy.deepcopy(self._backing[key])
        return self.loaded[key]

    def __contains__(self, key: object) -> bool:
        return key in self._backing

    def __iter__(self) -> Iterator[str]:
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def raw(self) -> dict[str, dict]:
        return self._backing

    def commit(self) -> None:
        self._backing.update(self.loaded)


class _MemoryView:
    def __init__(self, data: dict[str, dict[str, Any]]):
        self.users = _CopyOnReadTable(data["users"])
        self.orders = _CopyOnReadTable(data["orders"])
        self.products = _CopyOnReadTable(data["products"])

    def __getitem__(self, table: str) -> _CopyOnReadTable:
        return getattr(self, table)

    def find_user_by_email(self, email: str) -> str | None:
        for uid, p in self.users.raw().items():
            if p["email"].lower() == email.lower():
                return uid
        return None

    def find_user_by_name_zip(self, first: str, last: str, zip_code: str) -> str | None:
        for uid, p in self.users.raw().items():
            if (
                p["name"]["first_name"].lower() == first.lower()
                and p["name"]["last_name"].lower() == last.lower()
                and p["address"]["zip"] == zip_code
            ):
                return uid
        return None

    def product_index(self) -> dict[str, str]:
        return {p["name"]: p["product_id"] for p in self.products.raw().values()}


class MemoryStore:
    """In-process store. Tools mutate private copies; nothing lands unless the tool returns."""

    def __init__(self, data: dict[str, dict[str, Any]]):
        self.data = data
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, root: Path) -> MemoryStore:
        return cls(load_tau_json(root))

    @contextmanager
    def transaction(self) -> Iterator[DataView]:
        with self._lock:
            view = _MemoryView(self.data)
            yield view
            view.users.commit()
            view.orders.commit()

    def hash(self) -> str:
        return data_hash(self.data)


# --------------------------------------------------------------------------- sqlite


class _LazyTable(Mapping[str, dict]):
    def __init__(self, conn: sqlite3.Connection, table: str):
        self._conn, self._table = conn, table
        self.loaded: dict[str, dict] = {}
        self.original: dict[str, str] = {}

    def _load(self, key: str) -> dict | None:
        if key in self.loaded:
            return self.loaded[key]
        row = self._conn.execute(f"SELECT doc FROM {self._table} WHERE id = ?", (key,)).fetchone()
        if row is None:
            return None
        self.original[key] = row[0]
        self.loaded[key] = json.loads(row[0])
        return self.loaded[key]

    def __getitem__(self, key: str) -> dict:
        doc = self._load(key)
        if doc is None:
            raise KeyError(key)
        return doc

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._load(key) is not None

    def __iter__(self) -> Iterator[str]:
        for (key,) in self._conn.execute(f"SELECT id FROM {self._table} ORDER BY rowid"):
            yield key

    def __len__(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def dirty(self) -> Iterator[tuple[str, str]]:
        for key, doc in self.loaded.items():
            encoded = json.dumps(doc)
            if encoded != self.original.get(key):
                yield key, encoded


class _SqliteView:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.users = _LazyTable(conn, "users")
        self.orders = _LazyTable(conn, "orders")
        self.products = _LazyTable(conn, "products")

    def __getitem__(self, table: str) -> _LazyTable:
        return getattr(self, table)

    def find_user_by_email(self, email: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM users WHERE lower(json_extract(doc, '$.email')) = lower(?)", (email,)
        ).fetchone()
        return row[0] if row else None

    def find_user_by_name_zip(self, first: str, last: str, zip_code: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM users WHERE json_extract(doc, '$.address.zip') = ? "
            "AND lower(json_extract(doc, '$.name.first_name')) = lower(?) "
            "AND lower(json_extract(doc, '$.name.last_name')) = lower(?)",
            (zip_code, first, last),
        ).fetchone()
        return row[0] if row else None

    def product_index(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT json_extract(doc, '$.name'), id FROM products ORDER BY rowid").fetchall()
        return dict(rows)


class SqliteStore:
    """SQLite-backed store for the service. One connection, serialised by a lock."""

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()
        for t in TABLES:
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id TEXT PRIMARY KEY, doc TEXT)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS users_email ON users(lower(json_extract(doc, '$.email')))")
        self.conn.execute("CREATE INDEX IF NOT EXISTS users_zip ON users(json_extract(doc, '$.address.zip'))")

    def is_empty(self) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0

    def seed(self, data: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            for t in TABLES:
                self.conn.execute(f"DELETE FROM {t}")
                self.conn.executemany(
                    f"INSERT INTO {t} (id, doc) VALUES (?, ?)",
                    ((k, json.dumps(v)) for k, v in data[t].items()),
                )
            self.conn.execute("COMMIT")

    @contextmanager
    def transaction(self) -> Iterator[DataView]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            view = _SqliteView(self.conn)
            try:
                yield view
                for t in ("users", "orders"):
                    rows = list(view[t].dirty())
                    self.conn.executemany(f"UPDATE {t} SET doc = ? WHERE id = ?", ((d, k) for k, d in rows))
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                t: {k: json.loads(d) for k, d in self.conn.execute(f"SELECT id, doc FROM {t} ORDER BY rowid")}
                for t in TABLES
            }
