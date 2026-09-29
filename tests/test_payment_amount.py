from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rory_tools import CallSession, dispatch, impl


def _verified_session():
    session = CallSession()
    session.account = {"customer_id": "cus_payer"}
    session.fineract_client_id = 7
    return session


@pytest.mark.parametrize("amount", [2948, 7000])
def test_a_different_confirmed_amount_does_not_charge_the_bill(monkeypatch, amount):
    pay = Mock()
    monkeypatch.setattr(impl, "clients", SimpleNamespace(stripe=SimpleNamespace(
        get_bill=lambda _id: {"customer_id": "cus_payer", "amount_due_cents": 6007},
        pay_bill=pay,
    )))

    result = dispatch(_verified_session(), "make_payment", {
        "bill_id": "in_bill", "payment_method_id": "pm_card", "amount_cents": amount,
    })

    assert "no payment was made" in result["error"]
    pay.assert_not_called()


def test_a_confirmed_full_balance_is_charged_and_reported(monkeypatch):
    paid_bill = {"customer_id": "cus_payer", "status": "paid", "amount_due_cents": 0}
    pay = Mock(return_value=paid_bill)
    monkeypatch.setattr(impl, "clients", SimpleNamespace(stripe=SimpleNamespace(
        get_bill=lambda _id: {"customer_id": "cus_payer", "amount_due_cents": 6007},
        pay_bill=pay,
    )))

    result = dispatch(_verified_session(), "make_payment", {
        "bill_id": "in_bill", "payment_method_id": "pm_card", "amount_cents": 6007,
    })

    pay.assert_called_once_with("in_bill", "pm_card")
    assert result["paid"] is True
    assert result["amount_paid_cents"] == 6007


def test_a_plan_requiring_an_uncollected_down_payment_is_not_opened(monkeypatch):
    monkeypatch.setenv("RORY_REFERENCE_TIME", "2026-09-08T11:14:25Z")
    open_plan = Mock()
    monkeypatch.setattr(impl, "clients", SimpleNamespace(
        stripe=SimpleNamespace(arrears=lambda _id: {"arrears_cents": 11790, "days_past_due": 33}),
        fineract=SimpleNamespace(open_arrangement=open_plan),
    ))
    session = _verified_session()
    session.account["prior_arrangement_default_on"] = "2026-08-01"

    result = dispatch(session, "create_payment_arrangement", {"installments": 2})

    assert "No arrangement was opened" in result["error"]
    open_plan.assert_not_called()
