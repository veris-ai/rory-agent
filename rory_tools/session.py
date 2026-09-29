"""Who the caller has proved themselves to be, and the vendor clients behind it.

Transport-neutral: nothing here knows which voice stack carried the call.
Every transport builds one ``CallSession`` per call and hands it to
:mod:`rory_tools.dispatch`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import os

import stripe

from .services import FineractClient, StripeClient
from .services.fineract_client import FineractError
from .services.stripe_client import StripeError

VENDOR_ERRORS = (stripe.StripeError, StripeError, FineractError)


class _Clients:
    """Lazily-built vendor clients, shared across pipelines.

    Built on first use rather than at import so the module can be imported —
    and the tool schemas read — without every vendor credential present.
    """

    def __init__(self) -> None:
        self._stripe: StripeClient | None = None
        self._fineract: FineractClient | None = None

    @property
    def stripe(self) -> StripeClient:
        if self._stripe is None:
            self._stripe = StripeClient()
        return self._stripe

    @property
    def fineract(self) -> FineractClient:
        if self._fineract is None:
            self._fineract = FineractClient()
        return self._fineract


clients = _Clients()

MAX_VERIFICATION_ATTEMPTS = 3


class CallSession:
    """Who the caller has proved themselves to be, for one call.

    Each transport builds exactly one per call (Pipecat, for example, passes it
    to ``PipelineTask`` as ``app_resources``), and it dies with the call. It
    holds the account that
    ``verify_caller`` matched on BOTH systems, and that account is the subject
    of every other tool — which is why none of them take an account number
    from the model. Nothing else writes ``account``.
    """

    def __init__(self) -> None:
        self.account: Dict[str, Any] | None = None
        self.fineract_client_id: int | None = None
        self.failed_attempts = 0

    @property
    def customer_id(self) -> str:
        return self.account["customer_id"]

    def refresh(self) -> None:
        """Re-read the account after a write that changed it."""
        self.account = clients.stripe.get_customer(self.customer_id)


def _norm(value: Optional[str]) -> str:
    """Case and spacing only — never content.

    Nothing here interprets speech. The caller reads out a service address or
    an amount, and the transcriber mangles both in ways no rule written here
    could enumerate — so the model, which is already reading the transcript,
    writes the answer down in its canonical form and these compare what it
    wrote. See the verify_caller schema for the format it is asked for.
    """
    return " ".join(str(value or "").casefold().split())


def _digits(value: Optional[str]) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def require_credentials(*names: str) -> None:
    """Reject incomplete candidate configuration before accepting calls."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required credentials: {', '.join(missing)}")
