"""Store backends agree with each other, roll back on failure, and the ported tools
reproduce upstream tau-bench's ground-truth database hashes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import ROOT

from support_desk.domain.store import MemoryStore, SqliteStore, load_tau_json
from support_desk.domain.tools import TOOLS, refund_amount, run_tool, validate_args

TAU = ROOT / "data" / "tau_bench"


def _apply(store, name, **kw):
    args = validate_args(name, kw)
    assert not isinstance(args, str), args
    with store.transaction() as view:
        return run_tool(view, name, args)


def test_sqlite_and_memory_backends_agree(tmp_path: Path, mini_data):
    mem = MemoryStore(json.loads(json.dumps(mini_data)))
    sql = SqliteStore(tmp_path / "db.sqlite")
    sql.seed(json.loads(json.dumps(mini_data)))
    steps = [
        ("find_user_id_by_email", {"email": "ANA.DIAZ@example.com"}),
        ("find_user_id_by_name_zip", {"first_name": "bo", "last_name": "chen", "zip": "80202"}),
        ("cancel_pending_order", {"order_id": "#W0000001", "reason": "ordered by mistake"}),
        (
            "exchange_delivered_order_items",
            {
                "order_id": "#W0000002",
                "item_ids": ["1001"],
                "new_item_ids": ["1002"],
                "payment_method_id": "gift_card_1",
            },
        ),
        ("modify_pending_order_payment", {"order_id": "#W0000003", "payment_method_id": "paypal_2"}),
        ("list_all_product_types", {}),
    ]
    for name, kw in steps:
        assert _apply(mem, name, **kw) == _apply(sql, name, **kw)
    assert sql.snapshot() == mem.data


def test_sqlite_transaction_rolls_back_on_error(tmp_path: Path, mini_data):
    sql = SqliteStore(tmp_path / "db.sqlite")
    sql.seed(mini_data)
    with pytest.raises(RuntimeError), sql.transaction() as view:
        view.orders["#W0000001"]["status"] = "cancelled"
        raise RuntimeError("crash mid-write")
    assert sql.snapshot()["orders"]["#W0000001"]["status"] == "pending"


def test_memory_store_only_commits_on_success(mini_data):
    mem = MemoryStore(mini_data)
    with pytest.raises(RuntimeError), mem.transaction() as view:
        view.orders["#W0000001"]["status"] = "cancelled"
        raise RuntimeError("boom")
    assert mem.data["orders"]["#W0000001"]["status"] == "pending"


def test_tool_preconditions(mem_store):
    assert _apply(mem_store, "cancel_pending_order", order_id="#W0000002", reason="no longer needed") == (
        "Error: non-pending order cannot be cancelled"
    )
    assert (
        _apply(
            mem_store,
            "return_delivered_order_items",
            order_id="#W0000002",
            item_ids=["2001"],
            payment_method_id="paypal_2",
        )
        == "Error: payment method not found"
    )
    assert (
        _apply(
            mem_store,
            "exchange_delivered_order_items",
            order_id="#W0000002",
            item_ids=["1001"],
            new_item_ids=["1003"],
            payment_method_id="credit_card_1",
        )
        == "Error: new item 1003 not found or available"
    )


def test_refund_amount(mem_store):
    with mem_store.transaction() as view:
        ret = validate_args(
            "return_delivered_order_items",
            {"order_id": "#W0000002", "item_ids": ["2001", "1001"], "payment_method_id": "credit_card_1"},
        )
        assert refund_amount(view, "return_delivered_order_items", ret) == 1520.0
        cancel = validate_args("cancel_pending_order", {"order_id": "#W0000001", "reason": "no longer needed"})
        assert refund_amount(view, "cancel_pending_order", cancel) == 20.0


def test_tool_specs_are_valid_anthropic_definitions():
    for tool in TOOLS.values():
        spec = tool.spec()
        assert spec["name"] == tool.name and spec["description"]
        assert spec["input_schema"]["type"] == "object"
        assert "title" not in spec["input_schema"]


def test_validate_args_rejects_extra_and_missing():
    assert validate_args("cancel_pending_order", {"order_id": "#W1"}).startswith("Error: invalid arguments")
    assert "extra" in validate_args("get_order_details", {"order_id": "#W1", "force": True}).lower()
    assert validate_args("nope", {}).startswith("Error: unknown tool")


@pytest.mark.skipif(not (TAU / "orders.json").exists(), reason="run scripts/download_tau_bench.py")
def test_ported_tools_reproduce_upstream_ground_truth_hashes():
    """Hashes in the fixture were produced by tau-bench's own tool code (commit 59a200c)."""
    expected = json.loads((ROOT / "tests" / "fixtures" / "upstream_gt_hashes_test.json").read_text())
    tasks = json.loads((TAU / "tasks_test.json").read_text())
    base = load_tau_json(TAU)
    for i, task in enumerate(tasks):
        store = MemoryStore(json.loads(json.dumps(base)))
        for action in task["actions"]:
            if action["name"] != "transfer_to_human_agents":
                _apply(store, action["name"], **action["kwargs"])
        assert store.hash() == expected[f"test-{i}"], f"task {i}"
