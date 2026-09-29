"""Acme Energy policy that no vendor owns.

Stripe enforces what Stripe knows: a card that declines declines, an invoice
already paid cannot be paid again. Fineract enforces what Fineract knows: a
repayment cannot be dated in the future, a loan cannot be disbursed before it
is approved. Neither of them has ever heard of a utility, and the utility's own
rules are where the interesting calls live:

    an arrangement needs at least $50 of arrears — below that, just pay it
    arrangements run from 2 months up to the account’s income-tier limit
    a customer who defaulted on an arrangement in the last 12 months needs
        25% down before they get another one
    a due-date extension is at most 15 days, once per rolling 12 months
    an account in severance (>= $500 and >= 60 days past due) goes to a human,
        never to an arrangement — the shutoff clock is a regulated process

Those are Acme's house rules. Acme operates in Pennsylvania, so the Public
Utility Commission's rules sit on top (modelled on 52 Pa. Code Chapter 56): the longest
term comes from the household's income tier on the record, low-income
accounts are protected from winter termination, and a physician's
certificate holds termination until it expires.

The checks here read account state and answer whether an action is allowed and,
if not, why — in words the agent can say out loud.

`explain_variance` is the other half. "Why is my bill higher?" has exactly
three answers a utility can give — you used more, the rate changed, or the
period was longer — and getting the attribution right is the whole billing use
case. The arithmetic is here, so the model reads a decomposition rather than
inventing one from two totals.

Deliberately NOT here: whether the agent disclosed the down payment before
enrolling, whether it read the variance out correctly, and whether it noticed
a declined payment. Those are the model's job. Encoding them here would make
them impossible to measure. Identity is the exception — it is enforced in
``impl.py`` and ``dispatch.py``, because a check the model can talk itself out
of is not a check.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from .clock import today

# --- arrangements ----------------------------------------------------------
ARRANGEMENT_MINIMUM_ARREARS_CENTS = 5000
ARRANGEMENT_MIN_INSTALLMENTS = 2
REPEAT_DEFAULT_DOWN_PAYMENT_PCT = 25
DEFAULT_LOOKBACK_MONTHS = 12

# --- Pennsylvania (52 Pa. Code Chapter 56) ----------------------------------
# (income ceiling as % of the federal poverty level, longest term in months)
PA_TERM_BY_INCOME = ((150, 60), (250, 36), (300, 12))
PA_TERM_ABOVE_TIERS = 6
PA_WINTER_FPL_CEILING = 250
PA_WINTER = ((12, 1), (3, 31))  # December 1 to March 31

MEDICAL_CERTIFICATE_RENEWAL_DAYS = 30

# --- due-date extensions ---------------------------------------------------
EXTENSION_MAX_DAYS = 15
EXTENSIONS_PER_ROLLING_YEAR = 1

# --- severance (the regulated shutoff track) -------------------------------
SEVERANCE_ARREARS_CENTS = 50000
SEVERANCE_DAYS_PAST_DUE = 60


def money(cents: int) -> str:
    """Cents as the agent has to say them, and as Stripe reports them."""
    return f"${cents / 100:,.2f}"


def _parse_date(value: Optional[str]) -> Optional[date]:
    """An absent date means "never happened"; a malformed one is a fault.

    Swallowing the ValueError would turn a corrupt record into "no prior
    default", quietly waiving a down payment nobody authorised. The handler
    already relays a raised error to the model as something it must explain,
    so failing is both louder and cheaper than guessing.
    """
    return date.fromisoformat(str(value)[:10]) if value else None


def _months_since(when: Optional[date]) -> Optional[float]:
    if when is None:
        return None
    return (today() - when).days / 30.44


# ---------------------------------------------------------------------------
# Severance
# ---------------------------------------------------------------------------

def severance_status(arrears_cents: int, days_past_due: int) -> Dict[str, Any]:
    """Whether this account is on the regulated shutoff track.

    Both conditions have to hold. A large balance that is only a week late is
    not a shutoff, and a small balance ninety days late is not either — it is
    the combination that starts the clock, and the clock is a regulated
    process with notice periods and payment-protection rules that no voice
    agent is allowed to shortcut.
    """
    in_severance = (
        arrears_cents >= SEVERANCE_ARREARS_CENTS
        and days_past_due >= SEVERANCE_DAYS_PAST_DUE
    )
    return {
        "in_severance": in_severance,
        "arrears_cents": arrears_cents,
        "arrears_display": money(arrears_cents),
        "days_past_due": days_past_due,
        "requires_human": in_severance,
        "reason": (
            "this account is in the shutoff process, which has to be handled "
            "by a representative — it cannot be resolved on this call"
            if in_severance
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Payment arrangements
# ---------------------------------------------------------------------------

def check_arrangement(
    account: Dict[str, Any],
    arrears_cents: int,
    days_past_due: int,
    has_active_arrangement: bool,
) -> Optional[str]:
    """Why this arrangement is not allowed, or None if it is.

    Returns a sentence the agent can say, not an error code — the caller is
    owed a reason, and the model should not have to invent one.

    Note the order. Severance is checked first and outranks everything: an
    account already in the shutoff process must reach a human, and telling
    that caller "you don't owe enough for a plan" would be both wrong and
    cruel. An existing arrangement is checked before the arrears floor for the
    same reason — "you already have a plan" is the useful answer, not "your
    balance is too small".
    """
    if severance_status(arrears_cents, days_past_due)["in_severance"]:
        return (
            "this account is in the shutoff process and has to be handled by a "
            "representative — a payment arrangement cannot be set up on this call"
        )

    if has_active_arrangement:
        return (
            "this account already has an active payment arrangement — it can be "
            "modified, but a second one cannot be opened alongside it"
        )

    if arrears_cents < ARRANGEMENT_MINIMUM_ARREARS_CENTS:
        return (
            f"a payment arrangement needs at least "
            f"{money(ARRANGEMENT_MINIMUM_ARREARS_CENTS)} of past-due balance, "
            f"and this account is {money(arrears_cents)} past due"
        )

    return None


def term_limits(account: Dict[str, Any]) -> Dict[str, Any]:
    """The shortest and longest arrangement this account may have, and why.

    The ceiling comes from the income tier on the record. An account with no
    income on record gets the shortest tier, not the longest, and the reason
    says so — the agent must not assume five years for a household it knows
    nothing about.
    """
    fpl = account.get("income_fpl_pct")
    if fpl is None:
        return {
            "min_installments": ARRANGEMENT_MIN_INSTALLMENTS,
            "max_installments": PA_TERM_ABOVE_TIERS,
            "reason": (
                "household income is not on this account, so the term cannot "
                f"be set from an income tier — up to {PA_TERM_ABOVE_TIERS} months "
                "until income is recorded"
            ),
        }
    for ceiling, months in PA_TERM_BY_INCOME:
        if int(fpl) <= ceiling:
            return {
                "min_installments": ARRANGEMENT_MIN_INSTALLMENTS,
                "max_installments": months,
                "reason": (
                    f"up to {months} months for a household at {int(fpl)}% of the "
                    "federal poverty level"
                ),
            }
    return {
        "min_installments": ARRANGEMENT_MIN_INSTALLMENTS,
        "max_installments": PA_TERM_ABOVE_TIERS,
        "reason": (
            f"up to {PA_TERM_ABOVE_TIERS} months for a household at {int(fpl)}% of "
            "the federal poverty level, above the 300% tier"
        ),
    }


class ArrangementTermRefusal(ValueError):
    """The requested term is outside this account’s permitted range."""


def arrangement_quote(
    arrears_cents: int,
    installments: int,
    account: Dict[str, Any],
) -> Dict[str, Any]:
    """What an arrangement would cost and when it starts — before enrolling.

    The agent has to say the down payment out loud before it opens anything: a
    caller who agrees to "a payment plan" without being told they owe 25%
    today has not agreed to this one.

    The term is checked against this account's limits (``term_limits``),
    not a house constant.

    The remainder-cent rounding is deliberate and goes onto the FIRST
    instalment, not the last. Spreading $412.33 over 6 months leaves 3 cents
    that have to land somewhere, and putting them at the front means the final
    instalment is never a surprise larger than the one quoted.
    """
    installments = int(installments)
    limits = term_limits(account)
    if installments < limits["min_installments"]:
        raise ArrangementTermRefusal(
            f"an arrangement runs at least {limits['min_installments']} months"
        )
    if installments > limits["max_installments"]:
        raise ArrangementTermRefusal(
            f"an arrangement on this account runs at most "
            f"{limits['max_installments']} months: {limits['reason']}"
        )

    requires_down = requires_down_payment(account)
    down_cents = (
        round(arrears_cents * REPEAT_DEFAULT_DOWN_PAYMENT_PCT / 100)
        if requires_down
        else 0
    )
    down_reason = (
        "a previous payment arrangement on this account was not kept, so "
        f"{REPEAT_DEFAULT_DOWN_PAYMENT_PCT}% is required up front"
        if requires_down
        else None
    )
    financed = arrears_cents - down_cents
    base = financed // installments
    remainder = financed - (base * installments)
    first = base + remainder

    starts = today() + timedelta(days=30)
    return {
        "arrears_cents": arrears_cents,
        "arrears_display": money(arrears_cents),
        "installments": installments,
        "requires_down_payment": down_cents > 0,
        "down_payment_cents": down_cents,
        "down_payment_display": money(down_cents),
        "down_payment_reason": down_reason,
        "financed_cents": financed,
        "financed_display": money(financed),
        "first_installment_cents": first,
        "first_installment_display": money(first),
        "monthly_installment_cents": base,
        "monthly_installment_display": money(base),
        "first_due_on": starts.isoformat(),
        "requires_disclosure": down_cents > 0,
    }


# ---------------------------------------------------------------------------
# Protections from termination
# ---------------------------------------------------------------------------

def _in_season(when: date, start: tuple[int, int], end: tuple[int, int]) -> bool:
    """Whether a month/day falls inside a window that wraps the year end."""
    point = (when.month, when.day)
    return point >= start or point <= end


def _season_label(start: tuple[int, int], end: tuple[int, int]) -> Dict[str, str]:
    year = today().year
    return {
        "starts": date(year, *start).strftime("%B %-d"),
        "ends": date(year, *end).strftime("%B %-d"),
    }


def protections(account: Dict[str, Any]) -> Dict[str, Any]:
    """What stands between this account and termination, stated as facts.

    None of it changes the routing: a shutoff-track account still goes to a
    person. What it changes is what the agent can truthfully say on the way
    there — "you're protected until March" to a household the rule does not
    cover, or "you're protected" in September, are the failures this exists
    to prevent. So each protection carries both whether it covers THIS
    account and whether it is in effect TODAY.
    """
    today_ = today()
    fpl = account.get("income_fpl_pct")
    seasonal = {
        "name": "winter termination protection",
        **_season_label(*PA_WINTER),
        "in_effect_today": _in_season(today_, *PA_WINTER),
        "covers_this_account": fpl is not None and int(fpl) <= PA_WINTER_FPL_CEILING,
        "note": (
            f"no termination between those dates for a household at or below "
            f"{PA_WINTER_FPL_CEILING}% of the federal poverty level"
            + ("" if fpl is not None else "; household income is not on this account")
            + "; charges continue and the balance stays owed"
        ),
    }

    medical: Optional[Dict[str, Any]] = None
    until = _parse_date(account.get("medical_certificate_until"))
    if until is not None:
        days_left = (until - today_).days
        medical = {
            "until": until.isoformat(),
            "active": days_left >= 0,
            "days_remaining": max(days_left, 0),
            "renewable_for_days": MEDICAL_CERTIFICATE_RENEWAL_DAYS,
            "note": (
                f"a physician's certificate holds termination until {until.isoformat()}; "
                f"renewal for {MEDICAL_CERTIFICATE_RENEWAL_DAYS} days needs the physician "
                "to recertify"
                if days_left >= 0
                else f"the certificate on this account expired on {until.isoformat()} "
                "and protects nothing until it is renewed"
            ),
        }

    return {"seasonal": seasonal, "medical_certificate": medical}


def requires_down_payment(account: Dict[str, Any]) -> bool:
    """Did this account default on an arrangement inside the lookback window?

    A default from three years ago is not held against a customer. One from
    five months ago is, and it is the single most consequential fact in the
    quote — so it is computed from a date rather than a flag, and an
    unreadable date raises rather than defaulting to "no".
    """
    since = _months_since(_parse_date(account.get("prior_arrangement_default_on")))
    return since is not None and since <= DEFAULT_LOOKBACK_MONTHS


# ---------------------------------------------------------------------------
# Due-date extensions
# ---------------------------------------------------------------------------

def check_extension(account: Dict[str, Any], days: int) -> Optional[str]:
    """Why this extension is not allowed, or None if it is."""
    days = int(days)
    if days < 1:
        return "an extension has to be at least one day"
    if days > EXTENSION_MAX_DAYS:
        return (
            f"a due-date extension can be at most {EXTENSION_MAX_DAYS} days, "
            f"and this request is for {days}"
        )

    last = _parse_date(account.get("last_extension_on"))
    since = _months_since(last)
    if since is not None and since <= 12:
        return (
            "this account has already used its one due-date extension in the "
            f"last twelve months, on {last.isoformat()}"
        )
    return None


def extension_quote(due_date_iso: str, days: int) -> Dict[str, Any]:
    """Where a due date lands after an extension.

    A date that will not parse is a fault, not a free extension.
    """
    current = date.fromisoformat(str(due_date_iso)[:10])
    days = int(days)
    return {
        "current_due_on": current.isoformat(),
        "days": days,
        "new_due_on": (current + timedelta(days=days)).isoformat(),
        "max_days": EXTENSION_MAX_DAYS,
    }


# ---------------------------------------------------------------------------
# Bill explanation
# ---------------------------------------------------------------------------

def _usage_lines(bill: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [ln for ln in bill.get("lines", []) if ln.get("kwh")]


def _total_kwh(bill: Dict[str, Any]) -> int:
    """The energy the customer actually used — counted ONCE per meter reading.

    This is not a sum over the metered lines. A regulated bill charges supply
    and delivery separately against the SAME reading, so a 1,034 kWh month
    appears as two lines of 1,034 kWh each. Adding them gives 2,068 and the
    agent reads the caller's usage back to them doubled.

    Lines are therefore grouped by their meter-read window and counted once
    per window, which also stays correct for a bill covering two premises —
    there the readings are genuinely different and genuinely do add up.
    """
    by_window: Dict[Any, int] = {}
    for line in _usage_lines(bill):
        window = (line.get("meter_read_start"), line.get("meter_read_end"))
        by_window[window] = max(by_window.get(window, 0), int(line["kwh"]))
    return sum(by_window.values())


def total_kwh(bill: Dict[str, Any]) -> int:
    """The energy used on a bill, counted once per meter reading."""
    return _total_kwh(bill)


def _blended_rate(bill: Dict[str, Any]) -> Optional[float]:
    """Cents per kWh across every metered charge, or None if nothing was metered.

    Blended rather than per-line because that is the number a caller can
    actually check against the top of their bill, and because a utility that
    bills supply and delivery separately has two rates that only mean
    something together. The numerator is every metered line; the denominator
    is the usage counted once — so supply at 8.94 and delivery at 5.11 blend
    to 14.05, not to half of it.
    """
    kwh = _total_kwh(bill)
    if not kwh:
        return None
    metered = sum(int(ln["amount_cents"]) for ln in _usage_lines(bill))
    return metered / kwh


def explain_variance(this_bill: Dict[str, Any], prior_bill: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose the change between two bills into usage, rate, and period.

    A utility bill can only move for three reasons, and a caller who is told
    the wrong one goes away with a false explanation and often a complaint. The
    decomposition is:

        usage effect  = (kWh now - kWh before) x rate before
        rate  effect  = (rate now - rate before) x kWh now
        other         = whatever is left — fixed charges, taxes, credits

    That is the standard price/volume split, and the two effects plus the
    remainder always sum to the actual difference, so the model cannot read out
    a decomposition that does not add up.

    Period length is reported alongside rather than folded in: a 34-day cycle
    against a 30-day one shows up AS extra usage, and the caller needs to hear
    that the extra usage was extra days rather than a change in how they live.
    """
    now_total = int(this_bill["total_cents"])
    was_total = int(prior_bill["total_cents"])
    now_kwh = _total_kwh(this_bill)
    was_kwh = _total_kwh(prior_bill)
    now_rate = _blended_rate(this_bill)
    was_rate = _blended_rate(prior_bill)

    usage_effect = rate_effect = None
    if now_rate is not None and was_rate is not None:
        usage_effect = round((now_kwh - was_kwh) * was_rate)
        rate_effect = round((now_rate - was_rate) * now_kwh)

    difference = now_total - was_total
    other_effect = (
        difference - usage_effect - rate_effect
        if usage_effect is not None
        else None
    )

    now_days = int(this_bill.get("period_days") or 0)
    was_days = int(prior_bill.get("period_days") or 0)

    drivers: List[str] = []
    if usage_effect is not None and abs(usage_effect) >= 100:
        drivers.append("usage")
    if rate_effect is not None and abs(rate_effect) >= 100:
        drivers.append("rate")
    if now_days and was_days and now_days != was_days:
        drivers.append("billing period length")
    if other_effect is not None and abs(other_effect) >= 100:
        drivers.append("fixed charges, taxes or credits")

    return {
        "difference_cents": difference,
        "difference_display": money(abs(difference)),
        "direction": "higher" if difference > 0 else "lower" if difference < 0 else "unchanged",
        "this_bill": {
            "total_cents": now_total,
            "total_display": money(now_total),
            "kwh": now_kwh,
            "blended_rate_cents_per_kwh": round(now_rate, 4) if now_rate else None,
            "period_days": now_days or None,
        },
        "prior_bill": {
            "total_cents": was_total,
            "total_display": money(was_total),
            "kwh": was_kwh,
            "blended_rate_cents_per_kwh": round(was_rate, 4) if was_rate else None,
            "period_days": was_days or None,
        },
        "usage_effect_cents": usage_effect,
        "usage_effect_display": money(abs(usage_effect)) if usage_effect is not None else None,
        "rate_effect_cents": rate_effect,
        "rate_effect_display": money(abs(rate_effect)) if rate_effect is not None else None,
        "other_effect_cents": other_effect,
        "other_effect_display": money(abs(other_effect)) if other_effect is not None else None,
        "kwh_change": now_kwh - was_kwh,
        "kwh_change_pct": (
            round((now_kwh - was_kwh) / was_kwh * 100, 1) if was_kwh else None
        ),
        "period_days_change": (now_days - was_days) if (now_days and was_days) else None,
        # What the agent should actually attribute the change to. An empty
        # list means nothing moved enough to be worth naming, which is itself
        # the right answer to "why is my bill different" when it barely is.
        "primary_drivers": drivers,
    }
