"""Atoms adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Atoms-shaped wrapper: one client tool per
schema, declared fresh on every call via ``session.update`` (see
``rory_atoms/agent.py``) rather than stored on the agent, so each tool's JSON
Schema parameters cross the wire exactly as ``ToolSchema.parameters()`` built
them — nothing about a type or a constraint like ``make_payment.amount_cents``'s
``minimum`` is lost translating into some other shape first.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema

# Generous enough that a slow vendor call is never the thing that times the
# tool out — dispatch()'s own vendor-level timeouts sit well under this.
TOOL_TIMEOUT_MS = 30_000


def _as_session_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
        "timeout_ms": TOOL_TIMEOUT_MS,
    }


# Sent verbatim in the session.update that opens every call — see
# rory_atoms/agent.py.
TOOLS: List[Dict[str, Any]] = [_as_session_tool(s) for s in SCHEMAS]
