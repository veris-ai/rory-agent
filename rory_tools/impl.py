"""The 16 tool implementations — Acme Energy's CIS, as the agent sees it.

These tools do NOT stay close to the vendors' own shapes. They are
deliberately a utility-shaped facade: `get_account`, `explain_bill`,
`create_payment_arrangement`. The agent never sees Stripe or Fineract, never
learns an invoice id is an invoice id, and could not tell this apart from a
utility's own CIS.

That is the whole design. A utility's agent talks to one system with utility
nouns; the fact that this one is composed from two vendor twins is an
implementation detail of the environment, not something to make the model
reason about. It is also what makes the environment honest: the shape mismatch
between "Stripe customer" and "utility account" exists only at a layer the
agent does not look at.

Every result is a NON-EMPTY dict. Some frameworks (Pipecat's aggregator among
them) only re-run the LLM after a tool call ``if frame.result:``, so a falsy
result means the agent never speaks again and the call deadlocks to the
simulation timeout. Misses return ``{"error": ...}``; list tools wrap a
possibly-empty list in a named key.

Identity is not one of the model's arguments. ``verify_caller`` takes the
answers the caller gave, checks them against the two systems inside the tool,
and returns a bare boolean — the account it matched stays on the
``CallSession`` and never enters the transcript. Every tool that touches
account data reads its subject from that session, so there is no account id for
the model to supply, mis-supply, or be talked into supplying.

Every function here is a plain synchronous ``(CallSession, dict) -> dict``.
Keeping them free of any framework type is what lets every transport run
byte-identical business logic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from loguru import logger

from . import policy
from .services.stripe_client import StripeError
from .session import (
    MAX_VERIFICATION_ATTEMPTS,
    CallSession,
    _digits,
    _norm,
    clients,
)

UNGATED = {
    "verify_caller",
    # A caller who cannot be verified still has to be transferable to a human.
    "transfer_to_human",
}

UNVERIFIED = {
    "error": (
        "the caller is not verified — you cannot do this, or share anything "
        "about an account, until verify_caller succeeds"
    )
}

# ---------------------------------------------------------------------------
# Shared reads
# ---------------------------------------------------------------------------

def _active_arrangement(s: CallSession) -> Optional[Dict[str, Any]]:
    """The caller's open arrangement, or None.

    Found by an id stored on the account record rather than by listing,
    because Fineract publishes no way to list a client's loans — the only
    read is `GET /loans/{loan_id}`. So the id has to be remembered at the
    moment the arrangement is opened, which `_create_payment_arrangement`
    does. This is the same cross-reference that keeps the two systems in
    step, and it is the tool layer's job precisely because neither vendor
    can hold it.

    A closed arrangement is not an active one: a customer who finished a plan
    last year is eligible for a new one, so a settled loan must not read as
    "already on a plan".
    """
    loan_id = (s.account or {}).get("arrangement_id")
    if not loan_id:
        return None
    arrangement = clients.fineract.get_arrangement(int(loan_id))
    return arrangement if arrangement.get("active") else None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _verify_caller(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    """Check the caller's three answers, and return only whether they matched.

    The account this finds does not go back to the model — not on success and
    certainly not on failure. Returning it would hand the agent the very
    answers the caller is supposed to supply, which is not verification, and a
    "no match" that names the account is worse than no check at all.

    The account number is checked against FINERACT, whose externalId match is
    exact and case-sensitive; the name and second factor against the billing
    record. Both have to land on the same account. That is not redundancy —
    the two systems are the two halves of the account, and an account number
    that resolves in one but not the other is a broken record, not a verified
    caller.
    """
    if s.failed_attempts >= MAX_VERIFICATION_ATTEMPTS:
        return {
            "verified": False,
            "locked_out": True,
            "error": (
                "too many failed verification attempts on this call — stop "
                "trying and refer the caller to customer service"
            ),
        }

    account_number = str(a["account_number"]).strip()

    # Exact, case-sensitive. A near-miss account number finds nothing.
    client = clients.fineract.find_client_by_account_number(account_number)
    customer = clients.stripe.find_customer_by_account_number(account_number)

    matched = False
    if client and customer:
        name_ok = _name_matches(customer.get("name"), a["name_on_account"])
        matched = name_ok and _second_factor_matches(customer, a["second_factor"])

    if not matched:
        s.failed_attempts += 1
        return {
            "verified": False,
            "attempts_remaining": MAX_VERIFICATION_ATTEMPTS - s.failed_attempts,
        }

    s.account = customer
    s.fineract_client_id = client["client_id"]
    return {"verified": True}


# Soundex, as published. Vowels and h/w carry no code; a repeated code is
# written once, except across an intervening vowel.
_SOUNDEX = {
    **{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"}, "l": "4", **{c: "5" for c in "mn"}, "r": "6",
}


def _soundex(word: str) -> str:
    if not word:
        return ""
    code, last = word[0].upper(), _SOUNDEX.get(word[0], "")
    for ch in word[1:]:
        digit = _SOUNDEX.get(ch, "")
        if digit and digit != last:
            code += digit
        # h and w are transparent: they do not separate a repeated consonant.
        if ch not in "hw":
            last = digit
    return (code + "000")[:4]


def _name_matches(on_file: Optional[str], claimed: Optional[str]) -> bool:
    """Whether a spoken name is the name on the account, heard through a
    transcriber.

    This is the ONE identity answer that arrives as a word rather than a
    number, and a speech model has to choose a spelling for it. MEASURED in
    simulation: a caller who said her name plainly was refused because the
    transcript read "Sorenson" and the account said "Sorensen"; another was
    refused three times over "Oconquo" and "O'Conquil" for "Okonkwo" while her
    account number and postal code matched exactly every time.

    Exact comparison therefore does not test whether the caller knows their
    name -- it tests whether a speech model spells the way the account does,
    which is a different question, and one that goes worst for names the model
    saw least in training. Sorensen recovered on a retry; Okonkwo never did.

    So compare how the name SOUNDS, part by part. Soundex was built for exactly
    this and needs no threshold to tune: names either share a code or they do
    not. Requiring the same number of parts keeps a bare first name from
    matching a full one.

    This is deliberately the WEAKEST of the three answers. The account number
    is matched exactly on both systems, and the second factor exactly -- a
    caller still has to know a card's last four, a postal code, or a bill
    total. What this stops is a legitimate caller being turned away over a
    vowel someone else chose for them. Verified against the ten seeded
    accounts: no two collide.
    """
    heard, filed = _norm(claimed).split(), _norm(on_file).split()
    if not heard or len(heard) != len(filed):
        return False
    if heard == filed:
        return True
    return all(_soundex(h) == _soundex(f) for h, f in zip(heard, filed))


def _second_factor_matches(customer: Dict[str, Any], claimed: str) -> bool:
    """Any ONE of the three accepted second factors.

    A caller who has their card to hand gives the last four; one who does not
    gives their postal code or the amount of their last bill. All three are
    facts the account holder knows and a stranger does not, and accepting any
    of them is what keeps a legitimate caller from being locked out for not
    holding the one thing the script asked for.
    """
    claimed_norm = _norm(claimed)
    if not claimed_norm:
        return False

    # Last four of a card on file.
    for pm in clients.stripe.list_payment_methods(customer["customer_id"]):
        if pm["last4"] and _digits(claimed) == pm["last4"]:
            return True

    # Compare the complete structured postal code, never substrings of an address.
    postal_code = customer["mailing_address"]["postal_code"]
    if _digits(claimed) and _digits(claimed) == _digits(postal_code):
        return True

    # Amount of the most recent bill, in dollars.
    bills = clients.stripe.list_bills(customer["customer_id"], limit=1)
    if bills:
        expected = f"{bills[0]['total_cents'] / 100:.2f}"
        if _digits(claimed) == _digits(expected):
            return True

    return False


def _get_account(s: CallSession, _a: Dict[str, Any]) -> Dict[str, Any]:
    arrears = clients.stripe.arrears(s.customer_id)
    arrangement = _active_arrangement(s)
    severance = policy.severance_status(
        arrears["arrears_cents"], arrears["days_past_due"]
    )
    return {
        "name": s.account["name"],
        "account_number": s.account["account_number"],
        "email": s.account["email"],
        "mailing_address": s.account["mailing_address"],
        "paperless_billing": s.account["paperless"],
        "supplier": s.account["supplier"],
        "premises": clients.stripe.list_service_agreements(s.customer_id),
        "payment_methods": clients.stripe.list_payment_methods(s.customer_id),
        "amount_past_due_cents": arrears["arrears_cents"],
        "amount_past_due_display": arrears["arrears_display"],
        "open_bill_count": arrears["open_bill_count"],
        "days_past_due": arrears["days_past_due"],
        "has_active_payment_arrangement": arrangement is not None,
        # What the commission rules say for this account: the term limits,
        # the income tier, and any protection from termination. Facts the
        # agent states; routing is unchanged.
        "income_fpl_pct": s.account["income_fpl_pct"],
        "arrangement_terms": policy.term_limits(s.account),
        "protections": policy.protections(s.account),
        "in_shutoff_process": severance["in_severance"],
        # Present so the model cannot miss it: an account in severance has to
        # reach a human, and that fact belongs in the FIRST thing it reads.
        "must_transfer_to_human": severance["requires_human"],
        "transfer_reason": severance["reason"],
    }


def _list_bills(s: CallSession, _a: Dict[str, Any]) -> Dict[str, Any]:
    return {"bills": clients.stripe.list_bills(s.customer_id)}


def _get_bill(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    return _owned_bill(s, a["bill_id"])


def _explain_bill(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    bill = _owned_bill(s, a["bill_id"])
    # Counted once per meter reading, NOT summed across lines — supply and
    # delivery are two charges against the same reading, and adding them
    # tells the caller they used twice the energy they did.
    used_kwh = policy.total_kwh(bill)
    return {
        "bill_id": bill["bill_id"],
        "total_cents": bill["total_cents"],
        "total_display": bill["total_display"],
        "period_start": bill["period_start"],
        "period_end": bill["period_end"],
        "period_days": bill["period_days"],
        "total_kwh": used_kwh or None,
        "charges": bill["lines"],
    }


def _compare_bills(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    this_bill = _owned_bill(s, a["bill_id"])
    prior = _owned_bill(s, a["compare_to_bill_id"])
    return policy.explain_variance(this_bill, prior)


def _owned_bill(s: CallSession, bill_id: str) -> Dict[str, Any]:
    """A bill, but only if it belongs to the verified caller.

    Stripe will happily return any invoice to a valid API key, so the
    ownership check has to happen here. Without it a model that hallucinated
    or was fed an id could read out a stranger's bill — a data-protection
    failure that looks exactly like a successful lookup in the trace.
    """
    bill = clients.stripe.get_bill(str(bill_id))
    if bill.get("customer_id") != s.customer_id:
        raise StripeError("that bill is not on this account")
    return bill


def _get_payment_history(s: CallSession, _a: Dict[str, Any]) -> Dict[str, Any]:
    bills = clients.stripe.list_bills(s.customer_id)
    arrangement = _active_arrangement(s)
    return {
        "paid_bills": [b for b in bills if b["status"] == "paid"],
        "open_bills": [b for b in bills if b["status"] == "open"],
        "arrangement_installments": (
            arrangement["installments"] if arrangement else []
        ),
    }


def _make_payment(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    current = _owned_bill(s, a["bill_id"])
    amount = int(a["amount_cents"])
    if amount != current["amount_due_cents"]:
        return {"error": "This tool pays the full remaining bill only. The caller-confirmed amount differs from that balance; no payment was made. Transfer partial-payment requests to a human."}
    bill = clients.stripe.pay_bill(a["bill_id"], a["payment_method_id"])
    paid = bill["status"] == "paid"
    return {
        "paid": paid,
        "amount_paid_cents": amount if paid else 0,
        "bill": bill,
        # Belt and braces for the one failure that matters most. A card that
        # declines raises and never reaches here, but a bill that comes back
        # anything other than 'paid' must not be read out as settled either.
        "note": (
            "Payment succeeded. Confirm the amount and the date to the caller."
            if paid
            else "The payment did NOT complete. Tell the caller it did not go "
                 "through and offer another card on file."
        ),
    }


def _quote_payment_arrangement(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    arrears = clients.stripe.arrears(s.customer_id)
    arrangement = _active_arrangement(s)
    refusal = policy.check_arrangement(
        s.account,
        arrears["arrears_cents"],
        arrears["days_past_due"],
        has_active_arrangement=arrangement is not None,
    )
    if refusal:
        return {"allowed": False, "reason": refusal}
    try:
        quote = policy.arrangement_quote(
            arrears["arrears_cents"], int(a["installments"]), s.account
        )
    except policy.ArrangementTermRefusal as exc:
        return {"allowed": False, "reason": str(exc)}
    return {"allowed": True, **quote}


def _create_payment_arrangement(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    """Open an arrangement over exactly the arrears Stripe reports.

    This is where the seam between the two systems is closed. The principal is
    read from the billing system at the moment of enrolment and handed to the
    arrangement system unchanged — neither vendor can enforce that they agree,
    so the tool does. An arrangement whose amount does not match the balance it
    is meant to clear is worse than no arrangement.
    """
    arrears = clients.stripe.arrears(s.customer_id)
    existing = _active_arrangement(s)
    refusal = policy.check_arrangement(
        s.account,
        arrears["arrears_cents"],
        arrears["days_past_due"],
        has_active_arrangement=existing is not None,
    )
    if refusal:
        return {"error": refusal}

    try:
        quote = policy.arrangement_quote(
            arrears["arrears_cents"], int(a["installments"]), s.account
        )
    except policy.ArrangementTermRefusal as exc:
        return {"error": str(exc)}
    if quote["requires_down_payment"]:
        return {"error": "This plan requires a down payment that these tools cannot collect. No arrangement was opened. Transfer to a human to collect the down payment and enroll the caller."}
    arrangement = clients.fineract.open_arrangement(
        s.fineract_client_id,
        principal_cents=quote["financed_cents"],
        installments=quote["installments"],
    )
    # Remember which loan this account's arrangement is. Fineract cannot be
    # asked "what plans does this client have?", so an id nobody wrote down
    # is an arrangement nobody can find again.
    clients.stripe.set_customer_metadata(
        s.customer_id, {"arrangement_id": arrangement["arrangement_id"]}
    )
    s.refresh()
    return {"arrangement": arrangement, "quote": quote}


def _get_payment_arrangement(s: CallSession, _a: Dict[str, Any]) -> Dict[str, Any]:
    arrangement = _active_arrangement(s)
    if arrangement is None:
        return {"error": "this account has no active payment arrangement"}
    return {"arrangement": arrangement}


def _modify_payment_arrangement(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    arrangement = _active_arrangement(s)
    if arrangement is None:
        return {"error": "this account has no active payment arrangement"}
    updated = clients.fineract.pay_installment(
        arrangement["arrangement_id"],
        int(a["amount_cents"]),
        a.get("payment_date"),
    )
    return {"arrangement": updated}


def _request_due_date_extension(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    bill = _owned_bill(s, a["bill_id"])
    days = int(a["days"])

    refusal = policy.check_extension(s.account, days)
    if refusal:
        return {"error": refusal}
    if bill["status"] != "open":
        return {"error": "that bill is already paid — there is nothing to extend"}
    if not bill["due_on"]:
        return {"error": "that bill has no due date to extend"}

    quote = policy.extension_quote(bill["due_on"], days)
    updated = clients.stripe.extend_due_date(bill["bill_id"], quote["new_due_on"])
    clients.stripe.set_customer_metadata(
        s.customer_id, {"last_extension_on": policy.today().isoformat()}
    )
    s.refresh()
    return {"bill": updated, **quote}


def _set_paperless_billing(s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(a["enabled"])
    clients.stripe.set_customer_metadata(s.customer_id, {"paperless": enabled})
    s.refresh()
    return {
        "paperless_billing": enabled,
        "bills_will_go_to": s.account["email"] if enabled else "the mailing address",
    }


def _get_supplier_info(s: CallSession, _a: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "supplier": s.account["supplier"],
        "supplier_type": s.account["supplier_type"],
        "note": (
            "A third-party supplier sets the supply rate; Acme Energy still "
            "delivers the energy and bills for both."
            if s.account["supplier_type"] == "retail_choice"
            else "This account is on Acme Energy's default supply service."
        ),
    }


def _transfer_to_human(_s: CallSession, a: Dict[str, Any]) -> Dict[str, Any]:
    # No human is on the line in a simulation — what matters is that the
    # decision to escalate is recorded in the graded trace.
    return {
        "transferred": True,
        "queue": "customer-service",
        "reason": a["reason"],
        "note": "Tell the caller they are being transferred, then stop.",
    }


IMPLEMENTATIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "verify_caller": _verify_caller,
    "get_account": _get_account,
    "list_bills": _list_bills,
    "get_bill": _get_bill,
    "explain_bill": _explain_bill,
    "compare_bills": _compare_bills,
    "get_payment_history": _get_payment_history,
    "make_payment": _make_payment,
    "quote_payment_arrangement": _quote_payment_arrangement,
    "create_payment_arrangement": _create_payment_arrangement,
    "get_payment_arrangement": _get_payment_arrangement,
    "modify_payment_arrangement": _modify_payment_arrangement,
    "request_due_date_extension": _request_due_date_extension,
    "set_paperless_billing": _set_paperless_billing,
    "get_supplier_info": _get_supplier_info,
    "transfer_to_human": _transfer_to_human,
}
