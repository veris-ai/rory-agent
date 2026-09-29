from types import SimpleNamespace

import pytest

from rory_tools import CallSession, dispatch
from rory_tools import impl


@pytest.fixture
def account(monkeypatch):
    customer = {
        "customer_id": "cus_postal", "name": "Jordan Sample",
        "mailing_address": {"line1": "742 Example St", "postal_code": "19406"},
    }
    monkeypatch.setattr(impl, "clients", SimpleNamespace(
        fineract=SimpleNamespace(find_client_by_account_number=lambda _number: {"client_id": 86}),
        stripe=SimpleNamespace(
            find_customer_by_account_number=lambda _number: customer,
            list_payment_methods=lambda _id: [{"last4": "4242"}],
            list_bills=lambda _id, **_kwargs: [{"total_cents": 12345}],
        ),
    ))
    return customer


@pytest.mark.parametrize("factor", ["406", "9406", "742", "74219406", "99999"])
def test_address_fragments_do_not_verify_or_unlock_account_tools(account, factor):
    session = CallSession()
    result = dispatch(session, "verify_caller", {
        "account_number": "0000-00002", "name_on_account": account["name"], "second_factor": factor,
    })
    assert result == {"verified": False, "attempts_remaining": 2}
    assert session.account is None
    assert "error" in dispatch(session, "get_account", {})


@pytest.mark.parametrize("factor", ["19406", "4242", "123.45"])
def test_complete_supported_factors_verify(account, factor):
    session = CallSession()
    result = dispatch(session, "verify_caller", {
        "account_number": "0000-00002", "name_on_account": account["name"], "second_factor": factor,
    })
    assert result == {"verified": True}
    assert session.account == account
