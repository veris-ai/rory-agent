"""Apache Fineract — the payment-arrangement system of record for Acme Energy.

Plain httpx against the Mifos-hosted Fineract host (or the twin the bench names
in `FINERACT_API_BASE`), with the platform's static HTTP Basic credentials and
the tenant header every call requires.

The mapping from utility vocabulary to Fineract objects:

    customer account        Client          (externalId IS the account number)
    payment arrangement     Loan            (principal = the arrears financed)
    arrangement terms       Loan schedule   (numberOfRepayments = instalments)
    enrolling               apply -> approve -> disburse
    paying an instalment    Loan transaction, command=repayment
    what is left to pay     Loan summary + repayment schedule

WHY A LOAN. An arrangement is arrears turned into a fixed schedule of dated
obligations that reduce a balance — which is a loan, and modelling it as one
buys behaviour that would otherwise have to be faked. Three of Fineract's
measured refusals do real work here:

  * a repayment dated in the future is REFUSED, so a caller's "I'll pay
    Tuesday" cannot be recorded as a payment. An agent that tries to book a
    promise as a payment gets an error it has to read out, instead of a
    caller who believes they have paid;
  * a command against the wrong state is a 400, so an arrangement nobody
    approved cannot be disbursed;
  * `externalId` matching is EXACT and CASE-SENSITIVE, so a near-miss account
    number genuinely fails to find an account rather than fuzzily succeeding.

Interest is zero. A payment arrangement is a schedule, not credit, and a
utility that charged interest on one would be in front of its regulator.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ._http import vendor_request
from ..clock import today as clock_today

BASE_PATH = "/fineract-provider/api/v1"

# Fineract takes and returns dates in a format the CALLER declares on every
# request. This is the platform's own default and the one every measured
# scenario uses; changing it changes the meaning of every date field.
DATE_FORMAT = "dd MMMM yyyy"
LOCALE = "en"

# Fineract renders dates back as [year, month, day] arrays, not strings.
ARRANGEMENT_PRODUCT_NAME = "Payment Arrangement"


class FineractError(Exception):
    """A refusal from Fineract, or an Acme Energy rule enforced on top of it."""


def _fmt(when: date) -> str:
    """A date in the format declared on the request.

    `%-d` rather than `%d`: Fineract's `dd MMMM yyyy` accepts an unpadded day
    and renders one, and sending "04 July 2026" where the platform expects
    "4 July 2026" is a parse error rather than a near miss.
    """
    return when.strftime("%-d %B %Y")


def _parse(value: Any) -> Optional[str]:
    """Fineract's [y, m, d] array as an ISO date string."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return date(int(value[0]), int(value[1]), int(value[2])).isoformat()
    return None


def _money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _today() -> date:
    return clock_today()


class FineractClient:
    def __init__(self) -> None:
        self._http = httpx.Client(
            # Fineract is self-hosted, so there is no public default: point
            # FINERACT_API_BASE at your deployment (the bench injects its twin).
            base_url=os.environ["FINERACT_API_BASE"],
            auth=(
                os.environ.get("FINERACT_USER", "mifos"),
                os.environ.get("FINERACT_PASSWORD", "password"),
            ),
            # The tenant selector is a header, separate from auth, and required
            # on every call — missing or wrong is a 400, and the legacy
            # X-Mifos alias is rejected.
            headers={
                "Fineract-Platform-TenantId": os.environ.get(
                    "FINERACT_TENANT", "default"
                )
            },
            timeout=20.0,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return vendor_request(
            self._http, FineractError, "the account system", method,
            f"{BASE_PATH}{path}", **kwargs,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def find_client_by_account_number(self, account_number: str) -> Optional[Dict[str, Any]]:
        """The client whose externalId is this account number.

        Fineract matches externalId exactly and case-sensitively — a prefix
        finds nothing, and a case difference finds nothing. That is the
        behaviour this lookup is chosen for: it is half the caller's identity
        claim, so it has to be unforgiving.
        """
        body = self._request(
            "GET", "/clients", params={"externalId": str(account_number)}
        )
        rows = body.get("pageItems") if isinstance(body, dict) else body
        if not rows:
            return None
        row = rows[0]
        return {
            "client_id": row.get("id"),
            "account_number": row.get("externalId"),
            "display_name": row.get("displayName"),
            "status": (row.get("status") or {}).get("value"),
            "activated_on": _parse(row.get("activationDate")),
        }

    # ------------------------------------------------------------------
    # Arrangements
    # ------------------------------------------------------------------
    def get_arrangement_product_id(self) -> int:
        """The loan product every arrangement is opened against.

        Created by the world seeder; looked up here rather than configured,
        so the agent never carries an id that a re-seed would invalidate.
        """
        products = self._request("GET", "/loanproducts")
        for product in products or []:
            if product.get("name") == ARRANGEMENT_PRODUCT_NAME:
                return int(product["id"])
        raise FineractError(
            f"no '{ARRANGEMENT_PRODUCT_NAME}' product is configured — payment "
            "arrangements cannot be opened"
        )

    # NOTE: Fineract exposes no loans-listing endpoint — not by client, not at
    # all. `GET /clients/{id}/accounts` does not exist and answers 501, and
    # `GET /loans` is not a route either; the only read is
    # `GET /loans/{loan_id}`. So an arrangement can only be re-found by an id
    # something else remembered, which is why the tool layer stores it on the
    # account record. See `impl._active_arrangement`.

    def get_arrangement(self, loan_id: int) -> Dict[str, Any]:
        """One arrangement with its schedule and payments.

        `associations` is what makes the schedule come back at all — without
        it the read is a header and the agent cannot tell a caller which
        instalments they have already made.
        """
        loan = self._request(
            "GET",
            f"/loans/{int(loan_id)}",
            params={"associations": "repaymentSchedule,transactions"},
        )
        return self._arrangement_detail(loan)

    def open_arrangement(
        self,
        client_id: int,
        principal_cents: int,
        installments: int,
    ) -> Dict[str, Any]:
        """Apply, approve, and disburse in one go.

        Fineract models origination as three separate state transitions, and
        it refuses them out of order. A utility enrolling a customer on a plan
        does all three at once — the caller is on the phone, and there is no
        underwriting step — so they are driven together here and any refusal
        surfaces as itself.

        Money crosses into Fineract as MAJOR UNITS (dollars), not cents.
        Everything else in this agent is cents, so the conversion happens here,
        once, at the boundary that requires it.
        """
        product_id = self.get_arrangement_product_id()
        today = _today()
        principal = round(principal_cents / 100, 2)

        applied = self._request(
            "POST",
            "/loans",
            json={
                "clientId": int(client_id),
                "productId": product_id,
                "principal": principal,
                "loanType": "individual",
                "loanTermFrequency": int(installments),
                "loanTermFrequencyType": 2,
                "numberOfRepayments": int(installments),
                "repaymentEvery": 1,
                "repaymentFrequencyType": 2,
                # A payment arrangement is a schedule, not credit.
                "interestRatePerPeriod": 0,
                "amortizationType": 1,
                "interestType": 0,
                "interestCalculationPeriodType": 1,
                "transactionProcessingStrategyCode": "mifos-standard-strategy",
                "expectedDisbursementDate": _fmt(today),
                "submittedOnDate": _fmt(today),
                "dateFormat": DATE_FORMAT,
                "locale": LOCALE,
            },
        )
        loan_id = applied.get("loanId") or applied.get("resourceId")
        if not loan_id:
            raise FineractError("the arrangement was not created")

        self._request(
            "POST",
            f"/loans/{loan_id}",
            params={"command": "approve"},
            json={
                "approvedOnDate": _fmt(today),
                "expectedDisbursementDate": _fmt(today),
                "dateFormat": DATE_FORMAT,
                "locale": LOCALE,
            },
        )
        self._request(
            "POST",
            f"/loans/{loan_id}",
            params={"command": "disburse"},
            json={
                "actualDisbursementDate": _fmt(today),
                "transactionAmount": principal,
                "dateFormat": DATE_FORMAT,
                "locale": LOCALE,
            },
        )
        return self.get_arrangement(int(loan_id))

    def pay_installment(
        self, loan_id: int, amount_cents: int, on_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record an instalment payment against an arrangement.

        `on_date` defaults to today. A date in the FUTURE is refused by
        Fineract, and that refusal is left to travel: a caller who says they
        will pay next Tuesday has made a promise, not a payment, and the agent
        has to hear the difference rather than book it.
        """
        when = date.fromisoformat(on_date) if on_date else _today()
        self._request(
            "POST",
            f"/loans/{int(loan_id)}/transactions",
            params={"command": "repayment"},
            json={
                "transactionDate": _fmt(when),
                "transactionAmount": round(amount_cents / 100, 2),
                "paymentTypeId": 1,
                "dateFormat": DATE_FORMAT,
                "locale": LOCALE,
            },
        )
        return self.get_arrangement(int(loan_id))

    # ------------------------------------------------------------------
    # Shaping
    # ------------------------------------------------------------------
    @staticmethod
    def _arrangement_detail(loan: Dict[str, Any]) -> Dict[str, Any]:
        status = loan.get("status") or {}
        summary = loan.get("summary") or {}
        schedule = (loan.get("repaymentSchedule") or {}).get("periods") or []

        installments = []
        for period in schedule:
            # Period 0 is the disbursement row, not an instalment — it has no
            # due principal and reading it out would invent a payment.
            if not period.get("principalDue"):
                continue
            due_cents = round(float(period.get("totalDueForPeriod") or 0) * 100)
            paid_cents = round(float(period.get("totalPaidForPeriod") or 0) * 100)
            installments.append(
                {
                    "number": period.get("period"),
                    "due_on": _parse(period.get("dueDate")),
                    "amount_cents": due_cents,
                    "amount_display": _money(due_cents),
                    "paid_cents": paid_cents,
                    "paid_display": _money(paid_cents),
                    "settled": paid_cents >= due_cents > 0,
                }
            )

        principal_cents = round(float(summary.get("principalDisbursed") or 0) * 100)
        outstanding_cents = round(float(summary.get("totalOutstanding") or 0) * 100)
        paid_cents = round(float(summary.get("totalRepayment") or 0) * 100)

        overdue = [
            i for i in installments
            if not i["settled"] and i["due_on"] and i["due_on"] < _today().isoformat()
        ]

        return {
            "arrangement_id": loan.get("id"),
            "status": status.get("value"),
            "active": bool(status.get("active")),
            "closed": bool(status.get("closed")),
            "principal_cents": principal_cents,
            "principal_display": _money(principal_cents),
            "outstanding_cents": outstanding_cents,
            "outstanding_display": _money(outstanding_cents),
            "paid_to_date_cents": paid_cents,
            "paid_to_date_display": _money(paid_cents),
            "installments": installments,
            "installments_total": len(installments),
            "installments_settled": sum(1 for i in installments if i["settled"]),
            "installments_overdue": len(overdue),
            # An arrangement with a missed instalment is not yet broken, but it
            # is the fact that decides whether the caller is asking to modify a
            # plan or to be forgiven one.
            "in_arrears": bool(overdue),
            "next_due": next(
                (i for i in installments if not i["settled"]), None
            ),
        }
