"""Deepgram Voice Agent adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Deepgram-shaped wrapper:

    ToolSchema -> one entry of the Settings message's ``agent.think.functions``

Deepgram's function entry is flat — name, description, parameters — with no
``type: "function"`` envelope. Omitting ``endpoint`` is what marks a function
client-side: the server sends a FunctionCallRequest and waits for this process
to answer, instead of POSTing an HTTP endpoint of its own. ``parameters`` is
the same JSON-Schema object ``ToolSchema.parameters()`` already produces, so
the adaptation is mechanical — which is the point. If this file had to reword
a description to satisfy Deepgram, the candidates would no longer be sitting
the same exam.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_function(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }


# The ``agent.think.functions`` array, sent verbatim inside Settings.
FUNCTIONS: List[Dict[str, Any]] = [_as_function(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands Deepgram, read back for the parity check.

    Read out of ``FUNCTIONS`` rather than rebuilt from ``SCHEMAS``: the point is
    to prove the entries the model is sent still match the shared declaration.
    """
    return {
        fn["name"]: {
            "description": fn["description"],
            "required": list(fn["parameters"]["required"]),
            "properties": dict(fn["parameters"]["properties"]),
        }
        for fn in FUNCTIONS
    }
