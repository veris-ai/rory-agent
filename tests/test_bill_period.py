"""Where a bill's service period comes from, exercised without a sandbox.

A utility bill is a one-off Stripe invoice, and on those Stripe sets
`invoice.period_start` and `invoice.period_end` to the moment the invoice was
drawn — NOT the span of service. Reading them makes every bill a zero-day
period ending today, which quietly destroys the one explanation this agent
exists to give: "your bill went up because the cycle ran 34 days, not because
you used more."

MEASURED against real Stripe (test mode), line period 2026-07-29..2026-09-01:

    invoice.period_start  2026-09-01
    invoice.period_end    2026-09-01     -> 0 days
    line.period           2026-07-29..2026-09-01 -> 34 days

So the line periods are the real ones. These pin that, and the fallback for a
bill that has none.
"""

from rory_tools.services.stripe_client import StripeClient

DAY = 86400
# 2026-07-29 and 2026-09-01 — the span actually measured against real Stripe.
JUL_29 = 1785283200
SEP_01 = 1788220800


def _line(start, end):
    return {"period": {"start": start, "end": end}}


def _invoice(lines, inv_start=SEP_01, inv_end=SEP_01):
    """An invoice shaped the way Stripe returns one from `Invoice.list`."""
    return {
        "total": 10000,
        "amount_remaining": 10000,
        "period_start": inv_start,
        "period_end": inv_end,
        "lines": {"data": lines},
    }


def test_the_service_period_comes_from_the_lines_not_the_invoice():
    start, end = StripeClient._service_period(_invoice([_line(JUL_29, SEP_01)]))
    assert (start, end) == (JUL_29, SEP_01)
    # The whole point: a 34-day cycle survives instead of collapsing to zero.
    assert round((end - start) / DAY) == 34


def test_a_period_spans_the_outermost_edge_of_every_line():
    """Fixed charges and tax carry their own periods, and they need not agree.

    The bill covers the widest span any line was charged for — taking only the
    first line would report whichever charge happened to be listed first.
    """
    start, end = StripeClient._service_period(
        _invoice([
            _line(JUL_29 + 3 * DAY, SEP_01 - 2 * DAY),   # narrower
            _line(JUL_29, SEP_01),                        # the true span
            _line(JUL_29 + DAY, SEP_01 - DAY),
        ])
    )
    assert (start, end) == (JUL_29, SEP_01)


def test_a_bill_with_no_line_periods_falls_back_to_the_invoice():
    """A bill this agent did not create may have no line periods at all.

    Wrong is better than absent here: a caller asking about a bill still gets
    dates, rather than the tool omitting them entirely.
    """
    assert StripeClient._service_period(
        _invoice([{"period": {}}, {}], inv_start=JUL_29, inv_end=SEP_01)
    ) == (JUL_29, SEP_01)

    assert StripeClient._service_period(
        _invoice([], inv_start=JUL_29, inv_end=SEP_01)
    ) == (JUL_29, SEP_01)


def test_a_bill_summary_reports_the_days_the_cycle_actually_ran():
    """The end-to-end shape: what `list_bills` hands the model.

    Verified against real Stripe through `StripeClient.list_bills` — a
    four-line invoice over a 34-day cycle reports `period_days: 34`.
    """
    summary = StripeClient._bill_summary(
        _invoice([_line(JUL_29, SEP_01), _line(JUL_29, SEP_01)])
    )
    assert summary["period_days"] == 34
    assert summary["period_start"] == "2026-07-29"
    assert summary["period_end"] == "2026-09-01"


def test_a_paid_bill_separates_original_total_from_remaining_balance():
    summary = StripeClient._bill_summary(
        {
            **_invoice([]),
            "status": "paid",
            "total": 9546,
            "amount_due": 9546,
            "amount_paid": 9546,
            "amount_remaining": 0,
        }
    )

    assert summary["total_cents"] == 9546
    assert summary["amount_due_cents"] == 0


def test_an_open_bill_reports_its_remaining_balance():
    summary = StripeClient._bill_summary(
        {
            **_invoice([]),
            "status": "open",
            "total": 10963,
            "amount_due": 10963,
            "amount_paid": 0,
            "amount_remaining": 10963,
        }
    )

    assert summary["total_cents"] == 10963
    assert summary["amount_due_cents"] == 10963


def test_a_partial_payment_keeps_the_full_total_and_only_remaining_debt():
    summary = StripeClient._bill_summary({
        **_invoice([]), "status": "open", "total": 10000,
        "amount_due": 10000, "amount_paid": 3000, "amount_remaining": 7000,
    })
    assert summary["total_cents"] == 10000
    assert summary["amount_due_cents"] == 7000
