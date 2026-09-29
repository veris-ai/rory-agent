"""Rory's tool surface, declared once in a transport-neutral form.

Each transport adapts these into its own framework's schema type — Pipecat's
``FunctionSchema``, Gemini Live's ``FunctionDeclaration`` — so the model sees
the same 16 tools with the same descriptions and the same required arguments
whichever agent is under test. That is a benchmark requirement, not tidiness:
two candidates that differ in their tool descriptions are not running the same
exam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSchema:
    """One tool, in JSON-Schema-shaped pieces every framework can consume."""

    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def parameters(self) -> Dict[str, Any]:
        """The OpenAPI/JSON-Schema object both adapters build from."""
        return {
            "type": "object",
            "properties": dict(self.properties),
            "required": list(self.required),
        }


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
# Money crosses the boundary in CENTS everywhere, and every description says
# so. Dollars-versus-cents is a live failure mode for an agent that also has to
# read the amount aloud, and it is worth catching in the trace.

SCHEMAS = [
    ToolSchema(
        name="verify_caller",
        description=(
            "Verify the caller against the account before you do anything else. "
            "Takes the answers THEY gave you and checks them against the account "
            "record. Returns only whether they match — it tells you nothing about "
            "the account, so never guess an answer on the caller's behalf. Until "
            "this returns verified, every other account tool will refuse."
        ),
        properties={
            "account_number": {
                "type": "string",
                "description": (
                    "The account number the caller gave, digits and dashes as "
                    "they would be written: '4417-88231'. This is matched "
                    "exactly — do not reformat it or fill in a missing part."
                ),
            },
            "name_on_account": {
                "type": "string",
                "description": (
                    "The full name the caller gave, spelled as it would be "
                    "written: 'Elena Ferrara'. Not a title, not a nickname."
                ),
            },
            "second_factor": {
                "type": "string",
                "description": (
                    "The caller's second answer, written the way it would be "
                    "written down rather than the way it was said. One of: the "
                    "last four digits of the card on file ('4242'); the service "
                    "address postal code ('60614'); or the amount of their last "
                    "bill in dollars and cents ('214.07' — digits only, no "
                    "dollar sign). Never invent a value the caller did not give."
                ),
            },
        },
        required=["account_number", "name_on_account", "second_factor"],
    ),
    ToolSchema(
        name="get_account",
        description=(
            "The verified caller's account: name, account number, mailing "
            "address, paperless-billing status, energy supplier, the premises "
            "they take service at, the current amount past due and how many "
            "days late it is, and whether they have an active payment "
            "arrangement. Start most calls here."
        ),
        properties={},
        required=[],
    ),
    ToolSchema(
        name="list_bills",
        description=(
            "The caller's bills, newest first — the bill id, total in cents, "
            "due date, whether it is paid or still open, and the service "
            "period. Reaches back thirteen months so the same month last year "
            "is available for comparison."
        ),
        properties={},
        required=[],
    ),
    ToolSchema(
        name="get_bill",
        description=(
            "One bill in full: total in cents, due date, status, service "
            "period, and every charge on it broken out."
        ),
        properties={
            "bill_id": {
                "type": "string",
                "description": "The bill id from list_bills.",
            }
        },
        required=["bill_id"],
    ),
    ToolSchema(
        name="explain_bill",
        description=(
            "The charge-by-charge breakdown of one bill — supply, delivery, "
            "fixed customer charge, taxes — with the kilowatt-hours and the "
            "rate behind each metered line, and the meter read dates. Use this "
            "to tell a caller what they are actually paying for."
        ),
        properties={
            "bill_id": {
                "type": "string",
                "description": "The bill id from list_bills.",
            }
        },
        required=["bill_id"],
    ),
    ToolSchema(
        name="compare_bills",
        description=(
            "Explain WHY one bill differs from another. Returns the change "
            "split into how much came from using more or less energy, how much "
            "from a change in rate, how much from a longer or shorter billing "
            "period, and how much from fixed charges or taxes. Use this "
            "whenever a caller asks why their bill went up or down — do not "
            "guess the reason from the two totals."
        ),
        properties={
            "bill_id": {
                "type": "string",
                "description": "The bill the caller is asking about.",
            },
            "compare_to_bill_id": {
                "type": "string",
                "description": (
                    "The bill to compare against — usually the previous month, "
                    "or the same month last year if they ask about the season."
                ),
            },
        },
        required=["bill_id", "compare_to_bill_id"],
    ),
    ToolSchema(
        name="get_payment_history",
        description=(
            "What the caller has paid and when — settled bills with their "
            "amounts and dates, plus any instalments paid against a payment "
            "arrangement."
        ),
        properties={},
        required=[],
    ),
    ToolSchema(
        name="make_payment",
        description=(
            "Pay the full remaining bill with a card ALREADY ON FILE. The "
            "caller-confirmed amount must match the bill balance. Partial "
            "payments and arrangement down payments require a human. Never accept a card "
            "number over the phone — this takes only the id of a payment "
            "method already on the account, from get_account. If the card is "
            "declined this returns an error: tell the caller the payment did "
            "NOT go through and offer another card on file."
        ),
        properties={
            "bill_id": {
                "type": "string",
                "description": "The open bill to pay.",
            },
            "payment_method_id": {
                "type": "string",
                "description": "Id of a card already on file, from get_account.",
            },
            "amount_cents": {
                "type": "integer",
                "minimum": 1,
                "description": "The exact payment amount confirmed by the caller, in cents.",
            },
        },
        required=["bill_id", "payment_method_id", "amount_cents"],
    ),
    ToolSchema(
        name="quote_payment_arrangement",
        description=(
            "Check whether a payment arrangement is allowed and what it would "
            "cost, WITHOUT enrolling anyone. Returns the down payment, the "
            "monthly instalment, and the first due date — or the reason a plan "
            "is not possible. Always call this before create_payment_arrangement "
            "and read the down payment and monthly amount to the caller first."
        ),
        properties={
            "installments": {
                "type": "integer",
                "description": (
                    "How many monthly instalments the caller wants. The allowed "
                    "range is the account's arrangement_terms from get_account."
                ),
            }
        },
        required=["installments"],
    ),
    ToolSchema(
        name="create_payment_arrangement",
        description=(
            "Enrol the caller in a payment arrangement over their past-due "
            "balance. Only call this after quote_payment_arrangement and after "
            "the caller has agreed to the down payment and monthly amount you "
            "quoted them."
        ),
        properties={
            "installments": {
                "type": "integer",
                "description": "How many monthly instalments, within the account's arrangement_terms.",
            }
        },
        required=["installments"],
    ),
    ToolSchema(
        name="get_payment_arrangement",
        description=(
            "The caller's current payment arrangement: the original amount, "
            "what is still outstanding, every instalment with its due date and "
            "whether it has been paid, and whether any instalment is overdue."
        ),
        properties={},
        required=[],
    ),
    ToolSchema(
        name="modify_payment_arrangement",
        description=(
            "Record a payment against the caller's existing arrangement. The "
            "amount is in CENTS. The payment date cannot be in the future — a "
            "caller who says they will pay later has made a promise, not a "
            "payment, and this will refuse it."
        ),
        properties={
            "amount_cents": {
                "type": "integer",
                "description": "Amount paid, in cents (e.g. 6875 for $68.75).",
            },
            "payment_date": {
                "type": "string",
                "description": (
                    "Date of the payment as YYYY-MM-DD. Omit for today. Never "
                    "supply a future date to record something the caller has "
                    "only promised."
                ),
            },
        },
        required=["amount_cents"],
    ),
    ToolSchema(
        name="request_due_date_extension",
        description=(
            "Move a bill's due date later. At most 15 days, and only once per "
            "twelve months. Returns the new due date, or the reason it cannot "
            "be done."
        ),
        properties={
            "bill_id": {
                "type": "string",
                "description": "The open bill to extend.",
            },
            "days": {
                "type": "integer",
                "description": "How many days later, 1 to 15.",
            },
        },
        required=["bill_id", "days"],
    ),
    ToolSchema(
        name="set_paperless_billing",
        description=(
            "Turn paperless billing on or off for the account. Confirm the "
            "email address bills will go to before turning it on."
        ),
        properties={
            "enabled": {
                "type": "boolean",
                "description": "True to enrol in paperless, false to leave it.",
            }
        },
        required=["enabled"],
    ),
    ToolSchema(
        name="get_supplier_info",
        description=(
            "Who supplies the caller's energy and how they are billed for it — "
            "whether they are on the utility's default supply or have chosen a "
            "third-party retail supplier."
        ),
        properties={},
        required=[],
    ),
    ToolSchema(
        name="transfer_to_human",
        description=(
            "Hand the call to a human representative. Use for any account in "
            "the shutoff process, disputed charges, medical or protected-status "
            "questions, and any caller who asks for a supervisor."
        ),
        properties={
            "reason": {
                "type": "string",
                "description": "Why the call is being transferred.",
            }
        },
        required=["reason"],
    ),
]

TOOL_NAMES = [s.name for s in SCHEMAS]
