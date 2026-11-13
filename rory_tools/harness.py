"""Harness v2: a rule layer between a tool call and its execution.

Ported from a benchmark finding that a harness sitting between the model's
tool calls and their execution was the largest lever measured on Atoms —
bigger than turn-taking settings and MaAI combined. Its own description of
that layer is the contract this module keeps: "it never calls a model and
never changes the prompt or the tool list." Every rule below is a
deterministic check against this call's own tool history, re-derived from
what each rule had to mean given Rory's actual 16 tools, since the
benchmark's harness was written against a private task list this repo does
not have.

Wired into every transport at once, from :mod:`rory_tools.dispatch` — the one
seam already shared byte-for-byte by every candidate, so this is not an
Atoms-specific change. A rule that is only right on one vendor's tool-calling
quirks is not a rule, and the benchmark's own point was that Atoms needed one
regardless of which settings or model carried the audio.

What the benchmark's harness v2 found did NOT port cleanly, and why:

- "holds a transfer until the caller has heard their shutoff protections" —
  no transport here gives the harness a transcript, so "heard" cannot be
  checked. What CAN be checked is the mechanical
  precondition for the agent to be ABLE to say it: that ``get_account``, which
  carries ``protections``/``in_shutoff_process``, was actually fetched this
  call. See :data:`READ_BACK_PRECONDITIONS` applied to ``transfer_to_human``
  in :func:`check`.
- "wants a read-back and a yes for each account change" — same gap: no
  transcript means no way to confirm a spoken "yes". What this enforces
  instead is that the read tool whose entire job is to carry the terms back
  (``quote_payment_arrangement`` before ``create_payment_arrangement``, etc.)
  was actually called first, with the SAME terms where the schema has a
  matching field (installments). This is a strictly weaker guarantee than
  "the caller said yes" — the benchmark's own report flags the same gap,
  noting its harness's rules were written from a read of its own failures
  rather than tested independently of them.
- "removes the first harness's wrong blocks" — nothing to remove; Rory has no
  prior harness to carry mistakes forward from.
- the benchmark's old-harness appendix also described a "G3 justified
  hang-up" rule, gated on an ``end_call`` tool. Rory's 16 tools have no such
  tool (call termination is transport-owned, not a function call) — not
  ported, nothing here replaces it.
- "attaches exact figures, reasons and outcomes to the tools' answers" —
  already the house style of every refusal in :mod:`rory_tools.policy` and
  :mod:`rory_tools.impl` (``check_arrangement``, ``check_extension``,
  ``severance_status`` all return a sentence with the exact number in it, by
  design — see ``policy.check_arrangement``'s own docstring). The blocks this
  module adds follow the same convention rather than bolting on a second,
  redundant wrapper around results that already comply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Set, Tuple

if TYPE_CHECKING:
    from .session import CallSession

# Each account-change tool, mapped to the read tool(s) that carry the exact
# terms it changes — any one of which must have already succeeded this call
# before the change tool is allowed to run. Chosen from what each tool's own
# schema already says the caller has to have been told:
#   - make_payment: payment_method_id "already on file, from get_account"
#   - create_payment_arrangement: schema says call quote_payment_arrangement
#     first and "read the down payment and monthly amount to the caller"
#   - modify_payment_arrangement: records a payment against an existing plan
#     — get_payment_arrangement is what shows its current state
#   - request_due_date_extension: the bill being extended comes from
#     list_bills/get_bill
#   - set_paperless_billing: the prompt requires reading back the exact
#     destination email from get_account first
READ_BACK_PRECONDITIONS: Dict[str, FrozenSet[str]] = {
    "make_payment": frozenset({"get_account"}),
    "create_payment_arrangement": frozenset({"quote_payment_arrangement"}),
    "modify_payment_arrangement": frozenset({"get_payment_arrangement"}),
    "request_due_date_extension": frozenset({"get_bill", "list_bills"}),
    "set_paperless_billing": frozenset({"get_account"}),
}

ACCOUNT_CHANGE_TOOLS: FrozenSet[str] = frozenset(READ_BACK_PRECONDITIONS)

ArgsKey = Tuple[Tuple[str, Any], ...]


@dataclass
class HarnessState:
    """Per-call facts the rules in this module check against.

    One instance per :class:`~rory_tools.session.CallSession`, built empty and
    updated only by :func:`record`, which runs after a tool call succeeds.
    """

    calls_seen: Set[str] = field(default_factory=set)
    applied_changes: Set[Tuple[str, ArgsKey]] = field(default_factory=set)
    last_arrangement_quote_installments: Optional[int] = None


def _args_key(args: Dict[str, Any]) -> ArgsKey:
    return tuple(sorted(args.items()))


def check(session: "CallSession", name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A ``{"error": ...}`` to return instead of running the tool, or ``None`` to proceed.

    Runs after the verification gate in :func:`rory_tools.dispatch.dispatch`,
    so ``UNGATED`` tools called by an unverified caller never reach here with
    an account to check facts against — the checks below only ever apply to a
    verified caller.
    """
    h = session.harness

    if name == "transfer_to_human":
        # An unverified caller's transfer is never held up (nothing has been
        # looked up yet to hold it on). For a verified one, require
        # get_account so the shutoff protections it carries have actually
        # been fetched before the agent can hand the call off on those
        # grounds — matching the prompt's own "Start with the account" /
        # "tell them the true facts about their protection, then transfer".
        if session.account is not None and "get_account" not in h.calls_seen:
            return {
                "error": (
                    "call get_account first — the caller's shutoff protections "
                    "have to be looked up and stated before this transfer"
                )
            }
        return None

    if name in ACCOUNT_CHANGE_TOOLS:
        key = (name, _args_key(args))
        if key in h.applied_changes:
            return {
                "error": (
                    "this exact change already succeeded earlier on this call — "
                    "tell the caller it is already done, do not repeat it"
                )
            }

        needed = READ_BACK_PRECONDITIONS[name]
        if not (needed & h.calls_seen):
            return {
                "error": (
                    f"call {' or '.join(sorted(needed))} first and read the exact "
                    f"terms back to the caller before calling {name}"
                )
            }

        if name == "create_payment_arrangement" and h.last_arrangement_quote_installments is not None:
            # Only enforced once a quote has actually succeeded with a
            # specific installment count to hold the model to — a quote that
            # was itself refused (no down payment/monthly amount exists to
            # mismatch) leaves this to create_payment_arrangement's own
            # policy check, which re-derives and refuses independently.
            installments = args.get("installments")
            if h.last_arrangement_quote_installments != installments:
                return {
                    "error": (
                        "these installments do not match the last "
                        "quote_payment_arrangement call — re-quote and read back "
                        "the exact down payment and monthly amount before "
                        "enrolling the caller"
                    )
                }

    return None


def record(session: "CallSession", name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Update harness state after a tool ran.

    Skipped entirely on a ``{"error": ...}`` result — an errored call (a
    vendor failure, a missing-tool typo) taught the model nothing it is
    entitled to act on. A tool that ran but came back *refused* in its own
    shape rather than ``error`` (``quote_payment_arrangement``'s
    ``{"allowed": False, ...}``) still marks ``name`` as attempted — the
    agent genuinely tried the read-back step — but, since there are no
    specific terms in a refusal, does not record a value for the
    installments-match check below. A later ``create_payment_arrangement``
    is then free to run and hit its own policy refusal instead of a harness
    block that would only repeat it less precisely.
    """
    if "error" in result:
        return
    h = session.harness
    h.calls_seen.add(name)
    if name == "quote_payment_arrangement" and result.get("allowed"):
        h.last_arrangement_quote_installments = args.get("installments")
    if name in ACCOUNT_CHANGE_TOOLS:
        h.applied_changes.add((name, _args_key(args)))
