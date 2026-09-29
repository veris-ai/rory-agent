"""Rory's tools against live Veris twins.

These exercise `rory_tools.dispatch` — the exact shared entry point every
transport calls — through the same vendor clients,
against the same hostnames. The only difference from a real call is that
`veris-proxy` reroutes the traffic into the sandbox. Run them
under the proxy so the receipt and the verdict come from one run:

    veris-proxy run --sandbox <id> --image rory-core:veris --patch-bundled-cas \\
      -e STRIPE_API_KEY=... -e FINERACT_USER=mifos -e FINERACT_PASSWORD=password \\
      --require-service stripe --require-service fineract \\
      -- uv run --no-sync python -m pytest tests/test_integration.py -q

They assume the sandbox has been seeded with the ten Acme Energy development
accounts below, and they assert on those cases by account number rather than
by vendor id, so a re-seed does not break them.
"""

import os

import pytest

from rory_tools import CallSession, dispatch

pytestmark = pytest.mark.skipif(
    not os.environ.get("STRIPE_API_KEY"),
    reason="needs a Veris sandbox; run under veris-proxy",
)


# The seeded accounts, by the answers a caller gives: account number, name,
# and a second factor.
ALICE = ("4417-88231", "Alice Okonkwo", "60614")
BEN = ("4417-90055", "Ben Castellano", "60201")
CARMEN = ("4417-90781", "Carmen Reyes", "60647")
DMITRI = ("4417-91344", "Dmitri Volkov", "60302")
ELENA = ("4417-92018", "Elena Ferrara", "60607")
FEMI = ("4417-92556", "Femi Adeyemi", "60618")
GRACE = ("4417-93107", "Grace Lindqvist", "60187")
HASSAN = ("4417-93688", "Hassan Malik", "60610")
INGRID = ("4417-94210", "Ingrid Sorensen", "60130")
JONAH = ("4417-94877", "Jonah Pratt", "60435")


def _session(account=None):
    """A call session, verified against a seeded account unless told otherwise."""
    s = CallSession()
    if account:
        out = _call("verify_caller", s, account_number=account[0],
                    name_on_account=account[1], second_factor=account[2])
        assert out["verified"] is True, out
    return s


def _call(_tool, _session=None, **args):
    """Invoke a tool exactly as a transport's handler would.

    Every transport funnels every tool call through ``dispatch``, which is also
    where the error contract lives: a vendor refusal reaches the model as
    ``{"error": ...}`` rather than as an exception, because that is what the
    model has to read and relay. Calling it directly is therefore not a
    stand-in for the real path — it is the real path, minus the audio.
    """
    result = dispatch(_session or CallSession(), _tool, args)
    assert not result.get("error", "").startswith("that lookup failed:"), result
    return result


# --- identity --------------------------------------------------------------

def test_the_right_answers_verify_the_caller():
    s = CallSession()
    out = _call("verify_caller", s, account_number=ALICE[0],
                name_on_account=ALICE[1], second_factor=ALICE[2])
    assert out["verified"] is True
    assert s.account["name"] == "Alice Okonkwo"


def test_verification_returns_nothing_about_the_account():
    """The whole point: the verdict travels, the record does not."""
    out = _call("verify_caller", CallSession(), account_number=ALICE[0],
                name_on_account=ALICE[1], second_factor=ALICE[2])
    assert set(out) == {"verified"}


def test_a_near_miss_account_number_does_not_verify():
    """Fineract's externalId match is exact — a wrong digit finds nobody."""
    out = _call("verify_caller", CallSession(), account_number="4417-88232",
                name_on_account=ALICE[1], second_factor=ALICE[2])
    assert out["verified"] is False


def test_a_partial_account_number_is_not_a_prefix_match():
    out = _call("verify_caller", CallSession(), account_number="4417",
                name_on_account=ALICE[1], second_factor=ALICE[2])
    assert out["verified"] is False


def test_the_right_account_with_the_wrong_name_does_not_verify():
    out = _call("verify_caller", CallSession(), account_number=ALICE[0],
                name_on_account="Alice Okafor", second_factor=ALICE[2])
    assert out["verified"] is False


def test_the_right_account_with_a_wrong_second_factor_does_not_verify():
    out = _call("verify_caller", CallSession(), account_number=ALICE[0],
                name_on_account=ALICE[1], second_factor="99999")
    assert out["verified"] is False


def test_a_failed_attempt_never_names_the_account():
    out = _call("verify_caller", CallSession(), account_number=ALICE[0],
                name_on_account="Somebody Else", second_factor=ALICE[2])
    assert "name" not in out and "account_number" not in out


def test_three_failures_lock_the_call_out():
    s = CallSession()
    for _ in range(3):
        _call("verify_caller", s, account_number=ALICE[0],
              name_on_account="Wrong Person", second_factor=ALICE[2])
    out = _call("verify_caller", s, account_number=ALICE[0],
                name_on_account=ALICE[1], second_factor=ALICE[2])
    assert out["locked_out"] is True


def test_the_card_last_four_also_verifies():
    s = _session(ALICE)
    last4 = _call("get_account", s)["payment_methods"][0]["last4"]

    fresh = CallSession()
    out = _call("verify_caller", fresh, account_number=ALICE[0],
                name_on_account=ALICE[1], second_factor=last4)
    assert out["verified"] is True


def test_every_account_tool_refuses_before_verification():
    for tool in ("get_account", "list_bills", "get_payment_history",
                 "get_payment_arrangement"):
        assert "error" in _call(tool, CallSession())


def test_a_human_is_reachable_without_verifying():
    out = _call("transfer_to_human", CallSession(), reason="asked for a supervisor")
    assert out["transferred"] is True


# --- account and bills -----------------------------------------------------

def test_the_account_carries_the_premise_and_meter():
    account = _call("get_account", _session(ALICE))
    premise = account["premises"][0]
    assert premise["service_address"].startswith("1408 W Wellington")
    assert premise["meter_number"]
    assert account["account_number"] == ALICE[0]


def test_a_current_account_owes_nothing():
    account = _call("get_account", _session(ALICE))
    assert account["amount_past_due_cents"] == 0
    assert account["in_shutoff_process"] is False


def test_bills_come_back_newest_first_with_periods():
    bills = _call("list_bills", _session(ALICE))["bills"]
    assert len(bills) >= 4
    assert all(b["period_days"] for b in bills)


def test_a_bill_breaks_out_supply_delivery_and_tax():
    s = _session(ALICE)
    bill_id = _call("list_bills", s)["bills"][0]["bill_id"]
    out = _call("explain_bill", s, bill_id=bill_id)

    kinds = {c["charge_type"] for c in out["charges"]}
    assert {"supply", "delivery", "fixed", "tax"} <= kinds
    assert out["total_kwh"] > 0


def test_metered_lines_carry_the_kwh_and_the_rate():
    s = _session(ALICE)
    bill_id = _call("list_bills", s)["bills"][0]["bill_id"]
    supply = next(
        c for c in _call("explain_bill", s, bill_id=bill_id)["charges"]
        if c["charge_type"] == "supply"
    )
    assert supply["kwh"] > 0
    assert supply["rate_cents_per_kwh"] > 0
    assert supply["meter_read_start"] and supply["meter_read_end"]


def test_one_caller_cannot_read_another_callers_bill():
    """Stripe returns any invoice to a valid key; the tool layer has to refuse."""
    other = _call("list_bills", _session(BEN))["bills"][0]["bill_id"]
    out = _call("get_bill", _session(ALICE), bill_id=other)
    assert "error" in out
    assert "not on this account" in out["error"]


# --- explaining a bill -----------------------------------------------------

def test_a_cold_snap_bill_is_attributed_to_usage_not_rate():
    """Ben: usage jumped, the rate did not move."""
    s = _session(BEN)
    bills = _call("list_bills", s)["bills"]
    out = _call("compare_bills", s, bill_id=bills[0]["bill_id"],
                compare_to_bill_id=bills[1]["bill_id"])

    assert out["direction"] == "higher"
    assert out["kwh_change"] > 0
    assert "usage" in out["primary_drivers"]
    assert "rate" not in out["primary_drivers"]


def test_a_bill_that_rose_while_usage_fell_is_not_blamed_on_usage():
    """Carmen: the trap. Reasoning from the totals alone gets this backwards."""
    s = _session(CARMEN)
    bills = _call("list_bills", s)["bills"]
    out = _call("compare_bills", s, bill_id=bills[0]["bill_id"],
                compare_to_bill_id=bills[1]["bill_id"])

    assert out["direction"] == "higher"
    assert out["kwh_change"] < 0
    assert out["usage_effect_cents"] < 0
    assert out["rate_effect_cents"] > 0
    assert "billing period length" in out["primary_drivers"]


def test_the_decomposition_adds_up_against_real_seeded_bills():
    s = _session(CARMEN)
    bills = _call("list_bills", s)["bills"]
    out = _call("compare_bills", s, bill_id=bills[0]["bill_id"],
                compare_to_bill_id=bills[1]["bill_id"])

    parts = (out["usage_effect_cents"] + out["rate_effect_cents"]
             + out["other_effect_cents"])
    assert parts == out["difference_cents"]


# --- payment ---------------------------------------------------------------

def test_a_declined_card_is_reported_as_a_failure_not_a_payment():
    """Hassan: the card on file declines. This must never look like success."""
    s = _session(HASSAN)
    account = _call("get_account", s)
    open_bill = next(b for b in _call("list_bills", s)["bills"] if b["status"] == "open")
    declining = account["payment_methods"][0]["payment_method_id"]

    out = _call("make_payment", s, bill_id=open_bill["bill_id"],
                payment_method_id=declining, amount_cents=open_bill["amount_due_cents"])
    assert "error" in out or out.get("paid") is False


def test_a_working_card_settles_the_bill():
    s = _session(DMITRI)
    account = _call("get_account", s)
    open_bill = next(b for b in _call("list_bills", s)["bills"] if b["status"] == "open")

    out = _call("make_payment", s, bill_id=open_bill["bill_id"],
                payment_method_id=account["payment_methods"][0]["payment_method_id"],
                amount_cents=open_bill["amount_due_cents"])
    assert out["paid"] is True
    assert out["bill"]["status"] == "paid"


# --- arrangements ----------------------------------------------------------

def test_a_clean_arrears_account_is_quoted_without_a_down_payment():
    """Elena: the golden path."""
    out = _call("quote_payment_arrangement", _session(ELENA), installments=6)
    assert out["allowed"] is True
    assert out["requires_down_payment"] is False
    assert out["monthly_installment_cents"] > 0


def test_a_repeat_defaulter_is_quoted_a_down_payment_and_a_reason():
    """Femi: broke a plan five months ago."""
    out = _call("quote_payment_arrangement", _session(FEMI), installments=6)
    assert out["allowed"] is True
    assert out["requires_down_payment"] is True
    assert out["down_payment_cents"] > 0
    assert out["requires_disclosure"] is True


def test_an_account_in_shutoff_is_refused_an_arrangement():
    """Ingrid: $700+ and 71 days late. This goes to a person."""
    s = _session(INGRID)
    account = _call("get_account", s)
    assert account["in_shutoff_process"] is True
    assert account["must_transfer_to_human"] is True

    out = _call("quote_payment_arrangement", s, installments=6)
    assert out["allowed"] is False
    assert "shutoff process" in out["reason"]


def test_a_customer_already_on_a_plan_cannot_open_a_second():
    """Grace: modify the existing one, never stack another."""
    out = _call("quote_payment_arrangement", _session(GRACE), installments=6)
    assert out["allowed"] is False
    assert "already has an active payment arrangement" in out["reason"]


def test_a_current_account_does_not_qualify_for_an_arrangement():
    out = _call("quote_payment_arrangement", _session(ALICE), installments=6)
    assert out["allowed"] is False


def test_enrolling_opens_a_plan_for_exactly_the_arrears():
    """The seam between the two systems, closed: the principal is read from
    billing and handed to the arrangement system unchanged."""
    s = _session(ELENA)
    arrears = _call("get_account", s)["amount_past_due_cents"]
    quote = _call("quote_payment_arrangement", s, installments=6)

    out = _call("create_payment_arrangement", s, installments=6)
    assert "error" not in out, out
    assert out["arrangement"]["principal_cents"] == quote["financed_cents"]
    assert quote["financed_cents"] + quote["down_payment_cents"] == arrears
    assert out["arrangement"]["installments_total"] == 6


def test_an_existing_arrangement_shows_its_schedule_and_what_is_overdue():
    out = _call("get_payment_arrangement", _session(GRACE))["arrangement"]
    assert out["installments_total"] == 6
    assert out["installments_settled"] >= 1
    assert out["outstanding_cents"] > 0
    assert out["in_arrears"] is True


def test_a_payment_already_made_is_recorded_against_the_plan():
    s = _session(GRACE)
    before = _call("get_payment_arrangement", s)["arrangement"]
    out = _call("modify_payment_arrangement", s,
                amount_cents=before["next_due"]["amount_cents"])
    assert out["arrangement"]["paid_to_date_cents"] > before["paid_to_date_cents"]


def test_a_promise_to_pay_cannot_be_booked_as_a_payment():
    """Jonah: "I'll pay Friday". Fineract refuses a future-dated repayment, and
    the refusal has to reach the model rather than being smoothed over."""
    from datetime import date, timedelta

    s = _session(JONAH)
    friday = (date.today() + timedelta(days=4)).isoformat()
    out = _call("modify_payment_arrangement", s, amount_cents=7425,
                payment_date=friday)
    assert "error" in out


# --- account attributes ----------------------------------------------------

def test_paperless_can_be_turned_on_and_names_the_address_bills_go_to():
    s = _session(ALICE)
    out = _call("set_paperless_billing", s, enabled=True)
    assert out["paperless_billing"] is True
    assert "@" in out["bills_will_go_to"]

    assert _call("get_account", s)["paperless_billing"] is True
    _call("set_paperless_billing", s, enabled=False)


def test_the_supplier_is_reported_with_its_type():
    out = _call("get_supplier_info", _session(ALICE))
    assert out["supplier"]
    assert out["supplier_type"] in ("default_supply", "retail_choice")


# --- extensions ------------------------------------------------------------

def test_an_extension_moves_the_due_date_and_is_capped():
    s = _session(ELENA)
    open_bill = next(b for b in _call("list_bills", s)["bills"] if b["status"] == "open")

    too_long = _call("request_due_date_extension", s,
                     bill_id=open_bill["bill_id"], days=45)
    assert "error" in too_long

    out = _call("request_due_date_extension", s,
                bill_id=open_bill["bill_id"], days=10)
    assert out["new_due_on"] > out["current_due_on"]


def test_a_paid_bill_has_nothing_to_extend():
    s = _session(ALICE)
    paid = next(b for b in _call("list_bills", s)["bills"] if b["status"] == "paid")
    out = _call("request_due_date_extension", s, bill_id=paid["bill_id"], days=10)
    assert "error" in out
