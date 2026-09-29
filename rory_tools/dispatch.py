"""Run one tool call, the same way for every transport.

This is the whole shared contract between an agent and the world: gate the call
on verification, run the implementation, turn a vendor refusal into words the
model has to relay, and return the result to the transport. Bench captures model/tool messages
through its injected OpenTelemetry instrumentation.

It is deliberately synchronous. Every vendor client is blocking, and each
transport knows how to keep its own event loop free (they hop this onto a
worker thread) — but the decision of *what* a tool call means must not differ
between them.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from .impl import IMPLEMENTATIONS, UNGATED, UNVERIFIED
from .session import CallSession, VENDOR_ERRORS


def dispatch(session: CallSession, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute ``name`` against ``session`` and return the model-visible result.

    Never raises: a tool that raises leaves the model with nothing to say and
    deadlocks the call until the simulation times out. Vendor refusals become
    ``{"error": ...}`` on purpose — a declined card and a future-dated
    instalment should reach the caller as the agent explaining a "no", not as
    a silent failure or a false success.
    """
    logger.info(f"tool {name} {args}")
    fn = IMPLEMENTATIONS.get(name)
    if fn is None:
        result: Dict[str, Any] = {"error": f"no such tool: {name}"}
        return result

    if name not in UNGATED and session.account is None:
        logger.warning(f"tool {name} refused: caller not verified")
        return dict(UNVERIFIED)

    try:
        result = fn(session, args)
    except VENDOR_ERRORS as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — never deadlock the call
        logger.exception(f"tool {name} failed")
        result = {"error": f"that lookup failed: {exc}"}

    return result
