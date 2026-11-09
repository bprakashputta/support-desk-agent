from __future__ import annotations

from conftest import ROOT

from support_desk.kb.retriever import HelpCenter, validate_citations
from support_desk.pii import pseudonym, redact, redact_text


def test_redact_text():
    s = (
        "Email ana.diaz@example.com, call 512-555-0199, ship to 12 Oak Street Apt 4, Austin 73301, "
        "card 4242 4242 4242 4242, user ana_diaz_1111, order #W0000001, item 1008292230"
    )
    out = redact_text(s)
    for leaked in ("ana.diaz@", "555-0199", "Oak Street", "73301", "4242 4242", "ana_diaz_1111"):
        assert leaked not in out
    # business identifiers survive
    assert "#W0000001" in out and "1008292230" in out
    assert pseudonym("ana_diaz_1111") in out


def test_redact_structured():
    doc = {
        "user_id": "ana_diaz_1111",
        "name": {"first_name": "Ana", "last_name": "Diaz"},
        "address": {"address1": "12 Oak Street", "zip": "73301", "city": "Austin"},
        "payment_methods": {"credit_card_1": {"last_four": "4242"}},
        "order_id": "#W1",
    }
    r = redact(doc)
    assert r["user_id"].startswith("user:")
    assert r["name"] == "[redacted]"
    assert r["address"]["address1"] == "[redacted]" and r["address"]["zip"] == "[redacted]"
    assert r["address"]["city"] == "Austin"
    assert r["payment_methods"]["credit_card_1"]["last_four"] == "[redacted]"
    assert r["order_id"] == "#W1"


def test_pseudonym_is_stable():
    assert pseudonym("x_y_1234") == pseudonym("x_y_1234") != pseudonym("x_y_1235")


KB = HelpCenter.from_dir(ROOT / "kb")


def test_kb_retrieval_hits_the_right_article():
    cases = {
        "where does my refund go when I return something": "returns",
        "can I cancel an order that was already processed": "cancellations",
        "swap my shirt for a different size": "exchanges",
        "change shipping address on a pending order": "modify-order",
        "why do you need my zip code to verify me": "account",
        "talk to a real person": "human-support",
    }
    for q, article in cases.items():
        top = KB.search(q, k=1)[0][0]
        assert top.id.startswith(article + "#"), (q, top.id)


def test_citation_validation():
    text, kept, dropped = validate_citations("A [kb:returns#eligibility] B [kb:fake#x].", {"returns#eligibility"})
    assert kept == ["returns#eligibility"] and dropped == ["fake#x"]
    assert "[kb:fake#x]" not in text and "[kb:returns#eligibility]" in text


def test_console_preview_redacts_json_tool_results():
    import json

    from support_desk.agent.graph import _preview

    user = json.dumps(
        {"name": {"first_name": "Harper", "last_name": "Khan"}, "email": "h@x.com", "user_id": "harper_khan_9597"}
    )
    out = _preview(user)
    assert "Harper" not in out and "Khan" not in out and "h@x.com" not in out and "harper_khan" not in out


def test_zip_inside_identifier_is_not_redacted():
    from support_desk.pii import redact_text

    assert redact_text("approval_id=A-FBB12345 ok") == "approval_id=A-FBB12345 ok"
    assert redact_text("order #W8073958") == "order #W8073958"
    assert redact_text("my zip is 19029.") == "my zip is [zip]."
