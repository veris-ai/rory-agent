"""The Acme Energy rules, exercised without a sandbox.

`rory_tools/policy.py` is the only part of Rory that decides anything on its own — the
vendor clients just relay. These tests pin the decisions that make the
benchmark interesting: the severance threshold, arrangement eligibility, the
repeat-default down payment, the extension cap, and the price/volume
decomposition that makes "why is my bill higher" answerable.
"""

from datetime import timedelta

import pytest

from rory_tools import policy

@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setenv("RORY_REFERENCE_TIME", "2026-09-08T00:00:00Z")



def _account(prior_default_days_ago=None, last_extension_days_ago=None):
    return {
        "prior_arrangement_default_on": (
            _date(-prior_default_days_ago) if prior_default_days_ago else None
        ),
        "last_extension_on": (
            _date(-last_extension_days_ago) if last_extension_days_ago else None
        ),
    }


def _date(days_from_now):
    return (
        policy.today() + timedelta(days=days_from_now)
    ).isoformat()


def _bill(total_cents, kwh, rate_cents, days=30, fixed_cents=0):
    """A bill shaped the way a real one is: supply AND delivery, one reading.

    Both metered lines carry the same `kwh` against the same meter-read
    window, because that is what a regulated bill looks like — the customer
    used the energy once and is charged for it twice, at two rates that only
    mean something blended. Building the fixture with a single metered line
    would let a double-count in `_total_kwh` pass unnoticed.
    """
    window = {"meter_read_start": "2026-01-01", "meter_read_end": "2026-01-31"}
    supply_rate = rate_cents * 0.636
    delivery_rate = rate_cents - supply_rate
    lines = [
        {"kwh": kwh, "amount_cents": round(kwh * supply_rate),
         "charge_type": "supply", **window},
        {"kwh": kwh, "amount_cents": round(kwh * rate_cents) - round(kwh * supply_rate),
         "charge_type": "delivery", **window},
    ]
    if fixed_cents:
        lines.append({"kwh": None, "amount_cents": fixed_cents, "charge_type": "fixed"})
    return {"total_cents": total_cents, "period_days": days, "lines": lines}


# --- severance -------------------------------------------------------------

def test_a_large_balance_that_is_barely_late_is_not_a_shutoff():
    status = policy.severance_status(120000, 5)
    assert status["in_severance"] is False
    assert status["requires_human"] is False


def test_a_small_balance_long_overdue_is_not_a_shutoff():
    assert policy.severance_status(4000, 200)["in_severance"] is False


def test_a_large_balance_long_overdue_is_a_shutoff_and_needs_a_human():
    status = policy.severance_status(78000, 71)
    assert status["in_severance"] is True
    assert status["requires_human"] is True
    assert "representative" in status["reason"]


def test_the_severance_boundary_falls_on_the_right_side():
    assert policy.severance_status(50000, 60)["in_severance"] is True
    assert policy.severance_status(49999, 60)["in_severance"] is False
    assert policy.severance_status(50000, 59)["in_severance"] is False


# --- arrangement eligibility ----------------------------------------------

def test_an_ordinary_arrears_balance_qualifies():
    assert policy.check_arrangement(_account(), 41200, 38, False) is None


def test_severance_outranks_every_other_reason():
    """An account facing shutoff must hear "a person will help", not "no"."""
    refusal = policy.check_arrangement(_account(), 78000, 71, False)
    assert refusal is not None
    assert "shutoff process" in refusal


def test_severance_outranks_an_existing_arrangement_too():
    refusal = policy.check_arrangement(_account(), 78000, 71, True)
    assert "shutoff process" in refusal


def test_a_second_arrangement_cannot_be_opened_alongside_the_first():
    refusal = policy.check_arrangement(_account(), 41200, 38, True)
    assert refusal is not None
    assert "already has an active payment arrangement" in refusal


def test_an_existing_arrangement_outranks_the_arrears_floor():
    """"You already have a plan" is the useful answer, not "too small"."""
    refusal = policy.check_arrangement(_account(), 1000, 5, True)
    assert "already has an active payment arrangement" in refusal


def test_too_little_owed_is_refused_and_names_both_numbers():
    refusal = policy.check_arrangement(_account(), 3200, 10, False)
    assert refusal is not None
    assert "$50.00" in refusal
    assert "$32.00" in refusal


def test_the_arrears_floor_boundary_falls_on_the_right_side():
    assert policy.check_arrangement(_account(), 5000, 10, False) is None
    assert policy.check_arrangement(_account(), 4999, 10, False) is not None


# --- the repeat-default down payment --------------------------------------

def test_a_clean_account_needs_no_money_up_front():
    quote = policy.arrangement_quote(41200, 6, _account())
    assert quote["requires_down_payment"] is False
    assert quote["down_payment_cents"] == 0
    assert quote["requires_disclosure"] is False


def test_a_recent_default_requires_a_quarter_up_front_and_says_why():
    quote = policy.arrangement_quote(41200, 6, _account(prior_default_days_ago=150))
    assert quote["requires_down_payment"] is True
    assert quote["down_payment_cents"] == 10300
    assert quote["down_payment_display"] == "$103.00"
    assert "not kept" in quote["down_payment_reason"]
    assert quote["requires_disclosure"] is True


def test_an_old_default_is_not_held_against_the_customer():
    quote = policy.arrangement_quote(41200, 6, _account(prior_default_days_ago=1200))
    assert quote["requires_down_payment"] is False


def test_only_the_financed_remainder_is_spread_over_the_instalments():
    quote = policy.arrangement_quote(40000, 6, _account(prior_default_days_ago=60))
    assert quote["down_payment_cents"] == 10000
    assert quote["financed_cents"] == 30000
    assert quote["monthly_installment_cents"] == 5000


def test_a_corrupt_default_date_fails_loudly_rather_than_waiving_the_deposit():
    """Returning "no prior default" here would give away the down payment."""
    with pytest.raises(ValueError):
        policy.requires_down_payment(
            {"prior_arrangement_default_on": "sometime last year"}
        )


# --- arrangement arithmetic ------------------------------------------------

def test_the_instalments_always_add_back_up_to_the_financed_amount():
    quote = policy.arrangement_quote(41233, 6, _account())
    total = quote["first_installment_cents"] + quote["monthly_installment_cents"] * 5
    assert total == quote["financed_cents"] == 41233


def test_the_rounding_remainder_lands_on_the_first_instalment_not_the_last():
    """The final payment must never be larger than the one quoted."""
    quote = policy.arrangement_quote(41233, 6, _account())
    assert quote["first_installment_cents"] > quote["monthly_installment_cents"]


def test_an_evenly_divisible_balance_has_no_larger_first_instalment():
    quote = policy.arrangement_quote(60000, 6, _account())
    assert quote["first_installment_cents"] == quote["monthly_installment_cents"] == 10000


def test_a_term_outside_the_allowed_range_is_refused():
    for bad in (1, 0, 7, 13, 24):
        with pytest.raises(ValueError):
            policy.arrangement_quote(41200, bad, _account())


def test_the_shortest_and_longest_permitted_terms_are_allowed():
    assert policy.arrangement_quote(41200, 2, _account())["installments"] == 2
    assert policy.arrangement_quote(41200, 6, _account())["installments"] == 6


# --- due-date extensions ---------------------------------------------------

def test_a_first_extension_within_the_cap_is_allowed():
    assert policy.check_extension(_account(), 10) is None


def test_an_extension_beyond_fifteen_days_is_refused():
    refusal = policy.check_extension(_account(), 30)
    assert refusal is not None
    assert "15 days" in refusal


def test_a_second_extension_inside_twelve_months_is_refused_and_names_the_date():
    account = _account(last_extension_days_ago=90)
    refusal = policy.check_extension(account, 10)
    assert refusal is not None
    assert account["last_extension_on"] in refusal


def test_an_extension_is_available_again_after_twelve_months():
    assert policy.check_extension(_account(last_extension_days_ago=400), 10) is None


def test_a_zero_day_extension_is_refused():
    assert policy.check_extension(_account(), 0) is not None


def test_the_extension_lands_on_the_right_date():
    quote = policy.extension_quote("2026-03-12", 15)
    assert quote["new_due_on"] == "2026-03-27"


def test_an_unreadable_due_date_fails_loudly_rather_than_extending_forever():
    with pytest.raises((ValueError, TypeError)):
        policy.extension_quote("whenever", 10)


# --- bill explanation ------------------------------------------------------

def test_a_bill_driven_by_usage_is_attributed_to_usage():
    """Ben: cold snap. Usage jumps, the rate does not move."""
    prior = _bill(total_cents=9000, kwh=640, rate_cents=14.05)
    now = _bill(total_cents=14530, kwh=1034, rate_cents=14.05)

    out = policy.explain_variance(now, prior)
    assert out["direction"] == "higher"
    assert "usage" in out["primary_drivers"]
    assert "rate" not in out["primary_drivers"]
    assert out["rate_effect_cents"] == 0
    assert out["kwh_change"] == 394


def test_a_bill_that_rose_while_usage_fell_is_not_blamed_on_usage():
    """Carmen: the trap. The bill is up, but she used LESS energy.

    An agent reasoning from the two totals will tell her she used more. The
    decomposition has to make that impossible: the usage effect is negative
    and the rate effect is what carried the bill up.
    """
    prior = _bill(total_cents=10050, kwh=715, rate_cents=14.05, days=30)
    now = _bill(total_cents=11390, kwh=664, rate_cents=17.14, days=34)

    out = policy.explain_variance(now, prior)
    assert out["direction"] == "higher"
    assert out["kwh_change"] < 0
    assert out["usage_effect_cents"] < 0
    assert out["rate_effect_cents"] > 0
    assert "rate" in out["primary_drivers"]
    assert "billing period length" in out["primary_drivers"]


def test_a_longer_billing_period_is_reported_as_such():
    prior = _bill(total_cents=9000, kwh=640, rate_cents=14.05, days=30)
    now = _bill(total_cents=10200, kwh=726, rate_cents=14.05, days=34)

    out = policy.explain_variance(now, prior)
    assert out["period_days_change"] == 4
    assert "billing period length" in out["primary_drivers"]


def test_the_decomposition_always_sums_to_the_actual_difference():
    """The model must never be able to read out a breakdown that doesn't add up."""
    prior = _bill(total_cents=10050, kwh=715, rate_cents=14.05, fixed_cents=1200)
    now = _bill(total_cents=11390, kwh=664, rate_cents=17.14, fixed_cents=1200)

    out = policy.explain_variance(now, prior)
    parts = (
        out["usage_effect_cents"] + out["rate_effect_cents"] + out["other_effect_cents"]
    )
    assert parts == out["difference_cents"]


def test_an_unchanged_bill_names_no_driver():
    same = _bill(total_cents=9000, kwh=640, rate_cents=14.05)
    out = policy.explain_variance(same, dict(same))
    assert out["direction"] == "unchanged"
    assert out["primary_drivers"] == []


def test_a_bill_that_fell_reports_the_direction_and_a_positive_display():
    prior = _bill(total_cents=14530, kwh=1034, rate_cents=14.05)
    now = _bill(total_cents=9000, kwh=640, rate_cents=14.05)

    out = policy.explain_variance(now, prior)
    assert out["direction"] == "lower"
    assert out["difference_cents"] < 0
    # Read aloud as "five thousand five hundred thirty dollars lower", never
    # "minus five thousand".
    assert not out["difference_display"].startswith("-")


def test_a_bill_with_no_metered_lines_declines_to_invent_a_decomposition():
    """An estimated or fixed-only bill has no usage story, and saying so beats
    attributing the change to a rate that was never charged."""
    prior = {"total_cents": 1200, "period_days": 30, "lines": [
        {"kwh": None, "amount_cents": 1200, "charge_type": "fixed"}]}
    now = {"total_cents": 2400, "period_days": 30, "lines": [
        {"kwh": None, "amount_cents": 2400, "charge_type": "fixed"}]}

    out = policy.explain_variance(now, prior)
    assert out["usage_effect_cents"] is None
    assert out["rate_effect_cents"] is None
    assert out["difference_cents"] == 1200


def test_usage_is_counted_once_per_reading_not_once_per_charge():
    """Supply and delivery bill the SAME kilowatt-hours at two rates.

    Summing the metered lines reports 2,068 kWh for a 1,034 kWh month, and the
    agent reads the caller's usage back to them doubled. The blended rate has
    to come out at the full 14.05, not half of it.
    """
    bill = _bill(total_cents=14530, kwh=1034, rate_cents=14.05)
    assert len([ln for ln in bill["lines"] if ln["kwh"]]) == 2

    assert policy.total_kwh(bill) == 1034
    assert policy.explain_variance(bill, bill)["this_bill"]["kwh"] == 1034
    assert 14.0 < policy.explain_variance(bill, bill)["this_bill"][
        "blended_rate_cents_per_kwh"] < 14.1


def test_two_premises_on_one_bill_really_do_add_up():
    """The dedupe is per meter reading, not a blanket max — a bill covering
    two properties has two genuinely different readings."""
    bill = {
        "total_cents": 30000, "period_days": 30,
        "lines": [
            {"kwh": 400, "amount_cents": 5620, "meter_read_start": "2026-01-01",
             "meter_read_end": "2026-01-31"},
            {"kwh": 400, "amount_cents": 0, "meter_read_start": "2026-01-01",
             "meter_read_end": "2026-01-31"},
            {"kwh": 250, "amount_cents": 3512, "meter_read_start": "2026-01-05",
             "meter_read_end": "2026-02-04"},
        ],
    }
    assert policy.total_kwh(bill) == 650


def test_the_percentage_change_survives_a_first_ever_bill():
    """No prior usage means no percentage — not a division by zero."""
    prior = {"total_cents": 0, "period_days": 30, "lines": []}
    now = _bill(total_cents=9000, kwh=640, rate_cents=14.05)
    assert policy.explain_variance(now, prior)["kwh_change_pct"] is None


# --- commission rules --------------------------------------------------------

def _pa_account(**facts):
    return {**_account(), "income_fpl_pct": None, "medical_certificate_until": None, **facts}


@pytest.mark.parametrize("fpl, months", [(120, 60), (150, 60), (151, 36), (250, 36), (251, 12), (300, 12), (301, 6), (350, 6)])
def test_the_term_follows_the_income_tier(fpl, months):
    limits = policy.term_limits(_pa_account(income_fpl_pct=fpl))
    assert limits["max_installments"] == months
    assert str(fpl) in limits["reason"]


def test_no_income_on_record_gets_the_short_tier_and_says_why():
    limits = policy.term_limits(_pa_account())
    assert limits["max_installments"] == 6
    assert "not on this account" in limits["reason"]
    with pytest.raises(ValueError, match="not on this account"):
        policy.arrangement_quote(40000, 36, _pa_account())


def test_a_sixty_month_plan_is_quoted_not_refused():
    quote = policy.arrangement_quote(60000, 60, _pa_account(income_fpl_pct=120))
    assert quote["installments"] == 60 and quote["monthly_installment_cents"] == 1000


def test_the_repeat_default_down_payment_survives_the_tier():
    account = _pa_account(income_fpl_pct=200, prior_arrangement_default_on=_date(-150))
    quote = policy.arrangement_quote(40000, 36, account)
    assert quote["down_payment_cents"] == 10000
    with pytest.raises(ValueError, match="at most 36"):
        policy.arrangement_quote(40000, 48, account)


def test_winter_protection_covers_low_income_accounts_only(monkeypatch):
    monkeypatch.setenv("RORY_REFERENCE_TIME", "2026-09-08T00:00:00Z")
    covered = policy.protections(_pa_account(income_fpl_pct=200))["seasonal"]
    assert covered["covers_this_account"] and not covered["in_effect_today"]
    assert covered["starts"] == "December 1" and covered["ends"] == "March 31"
    assert not policy.protections(_pa_account(income_fpl_pct=320))["seasonal"]["covers_this_account"]
    unknown = policy.protections(_pa_account())["seasonal"]
    assert not unknown["covers_this_account"] and "not on this account" in unknown["note"]
    monkeypatch.setenv("RORY_REFERENCE_TIME", "2027-01-15T00:00:00Z")
    assert policy.protections(_pa_account(income_fpl_pct=200))["seasonal"]["in_effect_today"]


def test_a_medical_certificate_holds_until_its_date_and_then_protects_nothing():
    active = policy.protections(_pa_account(medical_certificate_until=_date(20)))["medical_certificate"]
    assert active["active"] and active["days_remaining"] == 20 and active["renewable_for_days"] == 30
    lapsed = policy.protections(_pa_account(medical_certificate_until=_date(-10)))["medical_certificate"]
    assert not lapsed["active"] and lapsed["days_remaining"] == 0 and "expired" in lapsed["note"]
    assert policy.protections(_pa_account())["medical_certificate"] is None
