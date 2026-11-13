"""Harness v2 — the rule layer ported from the VAmoS Pro benchmark's R5v2 arm.

Exercises ``rory_tools.harness`` directly (no vendor clients involved: the
rules only read call history and arguments) plus one pass through
``dispatch`` itself, to pin that the gate actually runs before the tool
implementation — not just that the rule functions are correct in isolation.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from rory_tools import CallSession, dispatch, harness, impl


def _verified_session() -> CallSession:
    session = CallSession()
    session.account = {"customer_id": "cus_1", "name": "Jordan Sample"}
    return session


# ---------------------------------------------------------------------------
# transfer_to_human: held until get_account has been fetched
# ---------------------------------------------------------------------------

def test_unverified_transfer_is_never_held_up():
    session = CallSession()
    assert harness.check(session, "transfer_to_human", {"reason": "supervisor"}) is None


def test_verified_transfer_blocked_before_get_account():
    session = _verified_session()
    blocked = harness.check(session, "transfer_to_human", {"reason": "shutoff"})
    assert blocked is not None
    assert "get_account" in blocked["error"]


def test_verified_transfer_allowed_after_get_account():
    session = _verified_session()
    harness.record(session, "get_account", {}, {"in_shutoff_process": True})
    assert harness.check(session, "transfer_to_human", {"reason": "shutoff"}) is None


# ---------------------------------------------------------------------------
# account-change tools: held until their read-back precondition succeeded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool,args,precondition",
    [
        ("make_payment", {"bill_id": "b1", "payment_method_id": "pm1", "amount_cents": 500}, "get_account"),
        ("create_payment_arrangement", {"installments": 6}, "quote_payment_arrangement"),
        ("modify_payment_arrangement", {"amount_cents": 500}, "get_payment_arrangement"),
        ("request_due_date_extension", {"bill_id": "b1", "days": 10}, "get_bill"),
        ("set_paperless_billing", {"enabled": True}, "get_account"),
    ],
)
def test_account_change_blocked_without_its_read_back(tool, args, precondition):
    session = _verified_session()
    blocked = harness.check(session, tool, args)
    assert blocked is not None
    assert precondition in blocked["error"]


def test_make_payment_allowed_after_get_account():
    session = _verified_session()
    harness.record(session, "get_account", {}, {"name": "Jordan Sample"})
    args = {"bill_id": "b1", "payment_method_id": "pm1", "amount_cents": 500}
    assert harness.check(session, "make_payment", args) is None


def test_create_payment_arrangement_requires_matching_quoted_installments():
    session = _verified_session()
    harness.record(
        session, "quote_payment_arrangement", {"installments": 6},
        {"allowed": True, "installments": 6},
    )
    # Different installments than the one actually quoted: still blocked.
    blocked = harness.check(session, "create_payment_arrangement", {"installments": 12})
    assert blocked is not None
    assert "quote_payment_arrangement" in blocked["error"]
    # The exact quoted terms: allowed.
    assert harness.check(session, "create_payment_arrangement", {"installments": 6}) is None


def test_a_refused_quote_satisfies_the_attempt_but_leaves_the_policy_check_to_run():
    """A refused quote has no terms to mismatch against, so the generic
    "a read-back was attempted" precondition is satisfied and the call falls
    through to create_payment_arrangement's own policy refusal — harness v2
    does not duplicate that logic, only that the read-back step happened."""
    session = _verified_session()
    harness.record(
        session, "quote_payment_arrangement", {"installments": 6},
        {"allowed": False, "reason": "already has an arrangement"},
    )
    assert harness.check(session, "create_payment_arrangement", {"installments": 6}) is None


def test_identical_change_is_not_repeated():
    session = _verified_session()
    harness.record(session, "get_account", {}, {"name": "Jordan Sample"})
    args = {"bill_id": "b1", "payment_method_id": "pm1", "amount_cents": 500}
    assert harness.check(session, "make_payment", args) is None
    harness.record(session, "make_payment", args, {"paid": True})
    blocked = harness.check(session, "make_payment", args)
    assert blocked is not None
    assert "already" in blocked["error"]


def test_a_different_amount_is_a_different_change_not_a_repeat():
    session = _verified_session()
    harness.record(session, "get_account", {}, {"name": "Jordan Sample"})
    first = {"bill_id": "b1", "payment_method_id": "pm1", "amount_cents": 500}
    second = {"bill_id": "b2", "payment_method_id": "pm1", "amount_cents": 700}
    harness.record(session, "make_payment", first, {"paid": True})
    assert harness.check(session, "make_payment", second) is None


def test_a_failed_call_never_satisfies_a_precondition_or_records_a_change():
    session = _verified_session()
    harness.record(session, "get_account", {}, {"error": "not found"})
    assert "get_account" not in session.harness.calls_seen

    args = {"bill_id": "b1", "payment_method_id": "pm1", "amount_cents": 500}
    harness.record(session, "make_payment", args, {"error": "card declined"})
    assert session.harness.applied_changes == set()


# ---------------------------------------------------------------------------
# Through dispatch(): the gate runs before the implementation, not instead of it
# ---------------------------------------------------------------------------

def test_dispatch_blocks_before_reaching_the_implementation(monkeypatch):
    called = []

    def _boom(_session: CallSession, _args: Dict[str, Any]) -> Dict[str, Any]:
        called.append(True)
        return {"transferred": True}

    monkeypatch.setitem(impl.IMPLEMENTATIONS, "transfer_to_human", _boom)
    session = _verified_session()

    result = dispatch(session, "transfer_to_human", {"reason": "shutoff"})
    assert "error" in result
    assert called == [], "the harness must block before the implementation runs"


def test_dispatch_lets_the_call_through_once_the_precondition_is_met(monkeypatch):
    monkeypatch.setitem(
        impl.IMPLEMENTATIONS, "get_account", lambda _s, _a: {"in_shutoff_process": True}
    )
    monkeypatch.setitem(
        impl.IMPLEMENTATIONS, "transfer_to_human", lambda _s, a: {"transferred": True, "reason": a["reason"]}
    )
    session = _verified_session()

    assert dispatch(session, "get_account", {}) == {"in_shutoff_process": True}
    result = dispatch(session, "transfer_to_human", {"reason": "shutoff"})
    assert result == {"transferred": True, "reason": "shutoff"}
