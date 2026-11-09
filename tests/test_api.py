"""HTTP API on a temp SQLite database with a scripted model: chat, approval, audit, health."""

from __future__ import annotations

import json

import pytest
from conftest import MINI, ScriptedLLM, call, say
from fastapi.testclient import TestClient

import support_desk.service.app as appmod


@pytest.fixture
def client(tmp_path, monkeypatch):
    tau = tmp_path / "tau"
    tau.mkdir()
    for t in ("users", "orders", "products"):
        (tau / f"{t}.json").write_text(json.dumps(MINI[t]))
    (tau / "wiki.md").write_text("# policy")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TAU_DIR", str(tau))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    script = ScriptedLLM(
        [
            call("find_user_id_by_email", 0, email="ana.diaz@example.com"),
            say("Return the laptop from order #W0000002 to credit_card_1? Reply yes."),
            call(
                "return_delivered_order_items",
                1,
                order_id="#W0000002",
                item_ids=["2001"],
                payment_method_id="credit_card_1",
            ),
            say("Your return is in progress."),
        ]
    )
    monkeypatch.setattr(appmod, "LLMClient", lambda *a, **k: script)
    with TestClient(appmod.app) as c:
        yield c


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["orders"] == 3 and h["spend_cap_usd"] == 5.0


def test_chat_approval_flow(client):
    tid = client.post("/api/threads").json()["thread_id"]
    r = client.post(f"/api/threads/{tid}/messages", json={"text": "I'm ana.diaz@example.com, return my laptop"})
    assert r.status_code == 200 and r.json()["status"] == "idle"
    r = client.post(f"/api/threads/{tid}/messages", json={"text": "yes"}).json()
    assert r["status"] == "awaiting_approval"
    # the customer cannot push more messages while a refund is under review
    assert client.post(f"/api/threads/{tid}/messages", json={"text": "hurry"}).status_code == 409
    pending = client.get("/api/approvals").json()
    assert len(pending) == 1 and pending[0]["amount"] == 1500.0
    done = client.post(f"/api/approvals/{pending[0]['id']}", json={"approved": True, "by": "lead"}).json()
    assert done["status"] == "idle" and done["new_replies"] == ["Your return is in progress."]
    assert client.post(f"/api/approvals/{pending[0]['id']}", json={"approved": True}).status_code == 409
    view = client.get(f"/api/threads/{tid}").json()
    assert view["customer"].startswith("user:")
    assert [m["role"] for m in view["transcript"]] == ["customer", "agent", "customer", "agent"]
    audit = client.get("/api/audit", params={"thread_id": tid}).json()
    events = {a["event"] for a in audit}
    assert {"customer_verified", "approval_requested", "approval_granted", "write_ok"} <= events
    assert "ana.diaz@example.com" not in json.dumps(audit)


def test_unknown_thread_404(client):
    assert client.get("/api/threads/nope").status_code == 404
    assert client.post("/api/threads/nope/messages", json={"text": "hi"}).status_code == 404
