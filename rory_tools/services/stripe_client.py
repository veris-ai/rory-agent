"""Stripe — the billing system of record for Acme Energy.

Uses the real `stripe` SDK against `api.stripe.com`, or against the twin the
bench names in `STRIPE_API_BASE`.

The mapping from utility vocabulary to Stripe objects:

    customer account    Customer            (metadata carries the account number)
    service agreement   Subscription        (metadata carries premise + meter)
    monthly bill        Invoice
    bill segment        Invoice line item
    metered usage       line metadata: kwh, rate_cents_per_kwh, charge_type
    amount past due     sum of open invoices
    paying a bill       Invoice.pay
    due-date extension  Invoice.due_date
    credit/adjustment   Credit note
    card on file        PaymentMethod attached to the Customer

WHY USAGE LIVES IN LINE METADATA. The obvious modelling is a Price with a
per-kWh `unit_amount` and `quantity` = kWh. It does not survive contact with
the money: a residential supply rate is 8.94 cents per kWh, which is not an
integer number of cents, and the input that expresses fractional unit amounts
(`unit_amount_decimal`) is not one every Stripe-compatible backend supports. Passing both a flat
`amount` and a `quantity` is worse, because their interaction is undefined. So
each bill segment is a flat `amount`, with `kwh` and `rate_cents_per_kwh` in
`metadata` and a human-readable `description`. The agent gets structured
usage, the money stays exact, and nothing here depends on behavior a
Stripe-compatible backend might not reproduce.

Policy that Stripe itself enforces is left to Stripe. Policy that lives in
Acme Energy's own rules — arrangement eligibility, the extension cap, the
severance threshold — is in `rory_tools/policy.py`, not here.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import stripe

from ..clock import now


class StripeError(Exception):
    """An Acme Energy refusal that Stripe itself would not raise.

    Stripe's own refusals already arrive as ``stripe.StripeError`` and are
    relayed as they are — ``session.VENDOR_ERRORS`` catches that type directly,
    so there is nothing to translate. This is for the cases the vendor cannot
    know about: an account number that matches nothing, a bill belonging to
    somebody else.
    """


def _d(obj: Any) -> Any:
    """A Stripe object as a plain dict; anything already plain, untouched."""
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _iso(ts: Optional[int]) -> Optional[str]:
    return (
        datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
        if ts
        else None
    )


def _days_past_due(due_ts: Optional[int]) -> int:
    if not due_ts:
        return 0
    delta = now() - datetime.fromtimestamp(
        int(due_ts), tz=timezone.utc
    )
    return max(0, delta.days)


class StripeClient:
    def __init__(self) -> None:
        stripe.api_key = os.environ["STRIPE_API_KEY"]
        # The Veris bench runs this image with no DNS interception: it
        # injects each twin's base URL (STRIPE_API_BASE) and expects the
        # agent to send its Stripe traffic there. Unset in production, where
        # the SDK's own api.stripe.com stays in force.
        if direct_base := os.environ.get("STRIPE_API_BASE"):
            stripe.api_base = direct_base.rstrip("/")
        # The SDK passes its OWN bundled CA file as requests' `verify=`
        # (stripe/_http_client.py), so unlike an httpx client it ignores
        # SSL_CERT_FILE and never trusts a CA added to the host's store.
        stripe.ca_bundle_path = os.environ.get("SSL_CERT_FILE", stripe.ca_bundle_path)
        self._stripe = stripe

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def find_customer_by_account_number(self, account_number: str) -> Optional[Dict[str, Any]]:
        """The customer whose metadata carries this utility account number.

        Exact match on the stored value — no prefix, no fuzz. The account
        number is half of the caller's identity claim, so a near miss has to
        miss.
        """
        escaped = str(account_number).replace("'", "")
        found = self._stripe.Customer.search(
            query=f"metadata['account_number']:'{escaped}'", limit=2
        )
        rows = [_d(c) for c in found.auto_paging_iter()]
        return self._customer_summary(rows[0]) if rows else None

    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        return self._customer_summary(_d(self._stripe.Customer.retrieve(customer_id)))

    # ------------------------------------------------------------------
    # Service agreements (premises)
    # ------------------------------------------------------------------
    def list_service_agreements(self, customer_id: str) -> List[Dict[str, Any]]:
        """One per premise the customer takes service at.

        A customer with two properties has two of these, each with its own
        service address and meter — which is why "which address is this bill
        for?" is answerable at all.
        """
        subs = self._stripe.Subscription.list(customer=customer_id, limit=10)
        out = []
        for sub in subs.auto_paging_iter():
            s = _d(sub)
            meta = s.get("metadata") or {}
            out.append(
                {
                    "service_agreement_id": s.get("id"),
                    "status": s.get("status"),
                    "premise_id": meta.get("premise_id"),
                    "service_address": meta.get("service_address"),
                    "meter_number": meta.get("meter_number"),
                    "commodity": meta.get("commodity"),
                    "rate_plan": meta.get("rate_plan"),
                    "started_on": _iso(s.get("start_date")),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Bills
    # ------------------------------------------------------------------
    def list_bills(self, customer_id: str, limit: int = 13) -> List[Dict[str, Any]]:
        """Bills newest first. The default reaches back thirteen months.

        Thirteen rather than twelve deliberately: "the same month last year"
        is the comparison a caller asks for, and twelve rows makes that the
        oldest row rather than one with a neighbour.
        """
        invoices = self._stripe.Invoice.list(customer=customer_id, limit=limit)
        return [self._bill_summary(inv) for inv in invoices.auto_paging_iter()]

    def get_bill(self, invoice_id: str) -> Dict[str, Any]:
        """One bill, with its segments — the breakdown "explain my bill" needs."""
        invoice = _d(self._stripe.Invoice.retrieve(invoice_id))
        summary = self._bill_summary(invoice)
        summary["lines"] = self._bill_lines(invoice_id)
        return summary

    def _bill_lines(self, invoice_id: str) -> List[Dict[str, Any]]:
        lines = self._stripe.Invoice.list_lines(invoice_id, limit=50)
        out = []
        for line in lines.auto_paging_iter():
            ln = _d(line)
            meta = ln.get("metadata") or {}
            kwh = meta.get("kwh")
            rate = meta.get("rate_cents_per_kwh")
            out.append(
                {
                    "description": ln.get("description"),
                    "amount_cents": ln.get("amount"),
                    "amount_display": _money(ln.get("amount") or 0),
                    "charge_type": meta.get("charge_type"),
                    "kwh": int(kwh) if kwh else None,
                    "rate_cents_per_kwh": float(rate) if rate else None,
                    "meter_read_start": meta.get("read_start"),
                    "meter_read_end": meta.get("read_end"),
                    "read_type": meta.get("read_type"),
                }
            )
        return out

    def arrears(self, customer_id: str) -> Dict[str, Any]:
        """The past-due balance, all open bills, and the oldest overdue age.

        Both numbers matter and they are not interchangeable: the severance
        test needs the balance AND the age, and an account can be badly
        overdue on a small amount or barely late on a large one.
        """
        bills = self.list_bills(customer_id)
        open_bills = [b for b in bills if b["status"] == "open"]
        total = sum(int(b["amount_due_cents"] or 0) for b in open_bills if b["days_past_due"] > 0)
        worst = max((b["days_past_due"] for b in open_bills), default=0)
        return {
            "arrears_cents": total,
            "arrears_display": _money(total),
            "open_bill_count": len(open_bills),
            "days_past_due": worst,
            "open_bills": open_bills,
        }

    # ------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------
    def list_payment_methods(self, customer_id: str) -> List[Dict[str, Any]]:
        pms = self._stripe.PaymentMethod.list(customer=customer_id, type="card", limit=10)
        out = []
        for pm in pms.auto_paging_iter():
            d = _d(pm)
            card = d.get("card") or {}
            out.append(
                {
                    "payment_method_id": d.get("id"),
                    "brand": card.get("brand"),
                    "last4": card.get("last4"),
                    "exp_month": card.get("exp_month"),
                    "exp_year": card.get("exp_year"),
                }
            )
        return out

    def pay_bill(self, invoice_id: str, payment_method_id: str) -> Dict[str, Any]:
        """Pay a bill with a card ALREADY on the account.

        `payment_method_id` must be one already attached — Rory never takes a
        card number by phone, so there is no path here that accepts raw PAN.

        A decline arrives as ``stripe.CardError`` and is deliberately NOT
        caught: the handler turns it into an error the model has to read and
        relay. Swallowing it and returning a bill summary would let the agent
        tell a caller their payment went through when it did not, which is the
        single worst thing this agent can do and therefore the thing the trace
        most needs to be able to catch.
        """
        invoice = self._stripe.Invoice.pay(invoice_id, payment_method=payment_method_id)
        return self._bill_summary(invoice)

    def extend_due_date(self, invoice_id: str, new_due_date_iso: str) -> Dict[str, Any]:
        """Move a bill's due date out.

        Stripe only carries `due_date` on a `send_invoice` invoice, which is
        what an unpaid utility bill already is here — a bill that would be
        auto-charged has nothing to extend.
        """
        due = datetime.fromisoformat(f"{new_due_date_iso}T12:00:00+00:00")
        invoice = self._stripe.Invoice.modify(
            invoice_id, due_date=int(due.timestamp())
        )
        return self._bill_summary(invoice)

    # ------------------------------------------------------------------
    # Account attributes
    # ------------------------------------------------------------------
    def set_customer_metadata(self, customer_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Write account preferences back, preserving every other key.

        Read-modify-write rather than sending the changed key alone. Real
        Stripe merges a partial `metadata` map — individual keys are updated,
        the rest survive, and an empty string unsets one — so sending just
        `{"paperless": True}` would be correct there.

        A backend that replaces the map wholesale instead would silently drop
        `account_number` and `fineract_client_id` — the account's entire
        identity — the first time a caller turns paperless billing on.

        Merging here is correct under BOTH semantics, so the agent does not
        depend on which one it is talking to.
        """
        current = _d(self._stripe.Customer.retrieve(customer_id)).get("metadata") or {}
        merged = {**{k: str(v) for k, v in current.items()},
                  **{k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in updates.items()}}
        customer = self._stripe.Customer.modify(customer_id, metadata=merged)
        return self._customer_summary(_d(customer))

    # ------------------------------------------------------------------
    # Shaping
    # ------------------------------------------------------------------
    @staticmethod
    def _customer_summary(cust: Dict[str, Any]) -> Dict[str, Any]:
        cust = _d(cust)
        meta = cust.get("metadata") or {}
        address = cust.get("address") or {}
        return {
            "customer_id": cust.get("id"),
            "name": cust.get("name"),
            "email": cust.get("email"),
            "phone": cust.get("phone"),
            "account_number": meta.get("account_number"),
            "mailing_address": address,
            "mailing_postal_code": address.get("postal_code"),
            "paperless": meta.get("paperless") == "true",
            "supplier": meta.get("supplier"),
            "supplier_type": meta.get("supplier_type"),
            "prior_arrangement_default_on": meta.get("prior_arrangement_default_on") or None,
            "last_extension_on": meta.get("last_extension_on") or None,
            "fineract_client_id": meta.get("fineract_client_id"),
            # The facts the commission rules turn on. Each is on the record
            # only when the utility recorded it, and absence is itself a state.
            "income_fpl_pct": (
                int(meta["income_fpl_pct"]) if meta.get("income_fpl_pct") not in (None, "") else None
            ),
            "medical_certificate_until": meta.get("medical_certificate_until") or None,
            # The account's open arrangement, if it has one. Carried here
            # because Fineract has no loans-listing endpoint — see
            # fineract_client for why this cross-reference has to exist.
            "arrangement_id": meta.get("arrangement_id") or None,
        }

    @staticmethod
    def _service_period(inv: Dict[str, Any]) -> tuple[Any, Any]:
        """The span of service this bill covers, from its LINE ITEMS.

        Not `invoice.period_start`/`period_end`: on a one-off invoice --
        which is what a utility bill is -- Stripe sets both of those to the
        moment the invoice was drawn, so reading them makes every bill a
        zero-day period ending today. MEASURED against real Stripe: line
        period 2026-07-29..2026-09-01 (34 days) came back as
        invoice.period_start == invoice.period_end == 2026-09-01.

        That is not cosmetic. "Your bill went up because the cycle ran 34 days
        instead of 30" is an explanation this agent exists to give, and a
        zero-length period erases it -- leaving a caller told the rate rose
        when the truth was the rate rose AND they were billed for four extra
        days.

        The line periods are the real ones and Stripe inlines them on
        `Invoice.list`, so this costs no extra call. Fixed charges (the
        monthly customer charge, tax) carry a period too, so the span is the
        outermost edge across every line. Falls back to the invoice fields
        when a bill has no lines with periods -- wrong is better than absent
        for a bill this agent did not create.
        """
        periods = []
        for line in ((inv.get("lines") or {}).get("data") or []):
            period = (_d(line) or {}).get("period") or {}
            start, end = period.get("start"), period.get("end")
            if start and end:
                periods.append((int(start), int(end)))
        if periods:
            return min(p[0] for p in periods), max(p[1] for p in periods)
        return inv.get("period_start"), inv.get("period_end")

    @staticmethod
    def _bill_summary(inv) -> Dict[str, Any]:
        inv = _d(inv)
        period_start, period_end = StripeClient._service_period(inv)
        days = None
        if period_start and period_end:
            days = round((int(period_end) - int(period_start)) / 86400)
        # Stripe's `amount_due` is the original finalized amount; it does not
        # become zero after payment. `total` is therefore the bill amount and
        # `amount_remaining` is what the customer still owes. Require these
        # authoritative amounts instead of guessing from invoice status.
        billed = inv["total"]
        remaining = inv["amount_remaining"]
        # The owning customer travels with every bill so the tool layer can
        # check ownership without a second round trip. Stripe returns any
        # invoice to a valid key, so somebody has to check, and doing it on
        # data already in hand keeps the check free enough that there is no
        # reason to skip it.
        customer = inv.get("customer")
        if isinstance(customer, dict):
            customer = customer.get("id")
        return {
            "bill_id": inv.get("id"),
            "customer_id": customer,
            "number": inv.get("number"),
            "status": inv.get("status"),
            "total_cents": billed,
            "total_display": _money(billed),
            "amount_due_cents": remaining,
            "amount_due_display": _money(remaining or 0),
            "due_on": _iso(inv.get("due_date")),
            "days_past_due": (
                _days_past_due(inv.get("due_date"))
                if inv.get("status") == "open"
                else 0
            ),
            "issued_on": _iso(inv.get("created")),
            "period_start": _iso(period_start),
            "period_end": _iso(period_end),
            "period_days": days,
        }
