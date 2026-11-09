from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from support_desk.domain.tools import validate_args
from support_desk.policy.guards import PolicyEngine, Rule, Verdict, is_affirmative


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, please go ahead", True),
        ("yes", True),
        ("No longer needed. Yes, cancel it.", True),
        ("Yes, I don't need it anymore", True),
        ("Sounds good", True),
        ("No.", False),
        ("No, wait, the other order", False),
        ("Hold on, let me check", False),
        ("Yes but don't proceed yet", False),
        ("What would the refund be?", False),
    ],
)
def test_affirmative(text, expected):
    assert is_affirmative(text) is expected


def _check(engine, view, name, args, msgs, auth="ana_diaz_1111", executed=()):
    a = validate_args(name, args)
    return engine.check(name, a, data=view, auth_user_id=auth, messages=msgs, executed=list(executed))


CANCEL = ("cancel_pending_order", {"order_id": "#W0000001", "reason": "no longer needed"})
CONFIRMED = [AIMessage("Cancel order #W0000001 (no longer needed)? Reply yes."), HumanMessage("yes")]


def test_confirmed_write_is_allowed(mem_store):
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(), view, *CANCEL, CONFIRMED)
    assert d.verdict is Verdict.ALLOW and not d.violations


def test_confirmation_must_name_the_order(mem_store):
    msgs = [AIMessage("Shall I go ahead with that?"), HumanMessage("yes")]
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(), view, *CANCEL, msgs)
    assert d.verdict is Verdict.BLOCK and d.violations[0][0] is Rule.CONFIRM


def test_confirmation_is_single_use(mem_store):
    used = [{"tool": "cancel_pending_order", "target": "#W0000001", "confirm_idx": 1}]
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(), view, *CANCEL, CONFIRMED, executed=used)
    assert d.verdict is Verdict.BLOCK


def test_unverified_write_is_blocked(mem_store):
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(), view, *CANCEL, CONFIRMED, auth=None)
    assert Rule.AUTH in [r for r, _ in d.violations]


def test_other_customers_order_is_blocked(mem_store):
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(), view, "get_order_details", {"order_id": "#W0000003"}, [])
    assert d.verdict is Verdict.BLOCK and d.violations[0][0] is Rule.SINGLE


def test_profile_address_change_needs_zip_in_confirmation(mem_store):
    args = {
        "user_id": "ana_diaz_1111",
        "address1": "1 New St",
        "address2": "",
        "city": "Austin",
        "state": "TX",
        "country": "USA",
        "zip": "73344",
    }
    ok = [AIMessage("Update your default address to 1 New St, Austin TX 73344?"), HumanMessage("yes")]
    with mem_store.transaction() as view:
        assert _check(PolicyEngine(), view, "modify_user_address", args, ok).verdict is Verdict.ALLOW
        other = dict(args, user_id="bo_chen_2222")
        assert _check(PolicyEngine(), view, "modify_user_address", other, ok).verdict is Verdict.BLOCK


def test_refund_over_threshold_needs_approval_in_both_modes(mem_store):
    ret = (
        "return_delivered_order_items",
        {"order_id": "#W0000002", "item_ids": ["2001"], "payment_method_id": "credit_card_1"},
    )
    msgs = [AIMessage("Return the laptop from #W0000002 to your card? yes?"), HumanMessage("yes")]
    with mem_store.transaction() as view:
        for mode in ("enforce", "audit"):
            d = _check(PolicyEngine(mode=mode, approval_threshold=1000), view, *ret, msgs)
            assert d.verdict is Verdict.APPROVAL and d.refund == 1500.0
        assert _check(PolicyEngine(approval_threshold=2000), view, *ret, msgs).verdict is Verdict.ALLOW


def test_audit_mode_records_but_allows(mem_store):
    with mem_store.transaction() as view:
        d = _check(PolicyEngine(mode="audit"), view, *CANCEL, [HumanMessage("cancel it")], auth=None)
    assert d.verdict is Verdict.ALLOW and len(d.violations) == 2
