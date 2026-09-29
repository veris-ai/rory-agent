import pytest

from rory_tools.services.stripe_client import StripeClient


@pytest.mark.parametrize("overdue_days, expected_cents", [(0, 0), (3, 1200)])
def test_only_overdue_open_balances_count_as_arrears(monkeypatch, overdue_days, expected_cents):
    bills = [
        {"status": "open", "amount_due_cents": 1200, "days_past_due": overdue_days},
        {"status": "open", "amount_due_cents": 9000, "days_past_due": 0},
        {"status": "paid", "amount_due_cents": 0, "days_past_due": 60},
    ]
    client = object.__new__(StripeClient)
    monkeypatch.setattr(client, "list_bills", lambda _customer: bills)

    result = client.arrears("cus_current")

    assert result["arrears_cents"] == expected_cents
    assert result["open_bill_count"] == 2
    assert result["open_bills"] == bills[:2]
    assert result["days_past_due"] == overdue_days
