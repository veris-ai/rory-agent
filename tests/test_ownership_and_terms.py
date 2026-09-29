from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rory_tools import CallSession, dispatch, impl
from rory_tools.services.stripe_client import StripeClient


@pytest.fixture
def verified(monkeypatch):
    session = CallSession()
    session.account = {"customer_id": "cus_own"}
    session.fineract_client_id = 7
    vendor = SimpleNamespace(
        get_bill=lambda _: {"customer_id": "cus_other"},
        arrears=lambda _: {"arrears_cents": 12000, "days_past_due": 30},
        pay_bill=Mock(),
    )
    fineract = SimpleNamespace(open_arrangement=Mock())
    monkeypatch.setattr(impl, "clients", SimpleNamespace(stripe=vendor, fineract=fineract))
    return session, vendor, fineract


@pytest.mark.parametrize("tool,args", [
    ("get_bill", {"bill_id": "in_other"}),
    ("explain_bill", {"bill_id": "in_other"}),
    ("compare_bills", {"bill_id": "in_other", "compare_to_bill_id": "in_prior"}),
    ("make_payment", {"bill_id": "in_other", "amount_cents": 12000, "payment_method_id": "pm_card"}),
    ("request_due_date_extension", {"bill_id": "in_other", "days": 5}),
])
def test_other_customer_bill_is_an_ownership_refusal(verified, tool, args):
    session, stripe, _ = verified
    assert dispatch(session, tool, args) == {"error": "that bill is not on this account"}
    stripe.pay_bill.assert_not_called()


@pytest.mark.parametrize("term", [1, 12])
@pytest.mark.parametrize("tool", ["quote_payment_arrangement", "create_payment_arrangement"])
def test_invalid_term_is_a_policy_refusal_without_vendor_write(verified, tool, term):
    session, _, fineract = verified
    result = dispatch(session, tool, {"installments": term})
    if tool == "quote_payment_arrangement":
        assert result["allowed"] is False
        reason = result["reason"]
    else:
        reason = result["error"]
    assert "at least 2" in reason if term == 1 else "at most 6" in reason
    assert "lookup failed" not in reason
    fineract.open_arrangement.assert_not_called()


@pytest.mark.parametrize("missing", ["total", "amount_remaining"])
def test_missing_authoritative_invoice_amount_fails(missing):
    invoice = {"total": 10000, "amount_remaining": 7000, "amount_paid": 3000, "amount_due": 10000}
    del invoice[missing]
    with pytest.raises(KeyError, match=missing):
        StripeClient._bill_summary(invoice)


def test_paperless_write_round_trips_string_metadata():
    customer = {"id": "cus_own", "metadata": {"account_number": "7301"}}
    modify = Mock(side_effect=lambda _, metadata: {**customer, "metadata": metadata})
    client = object.__new__(StripeClient)
    client._stripe = SimpleNamespace(Customer=SimpleNamespace(retrieve=lambda _: customer, modify=modify))
    result = client.set_customer_metadata("cus_own", {"paperless": True})
    assert result["paperless"] is True
    modify.assert_called_once_with("cus_own", metadata={"account_number": "7301", "paperless": "true"})
