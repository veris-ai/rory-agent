"""OpenAI Realtime adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Realtime-shaped wrapper:

    ToolSchema -> one entry of the session.update ``tools`` array

Realtime takes a flat ``{"type": "function", "name", "description",
"parameters"}`` entry with a JSON-Schema parameters object, which is the shape
``ToolSchema.parameters()`` already produces, so the adaptation is mechanical —
which is the point. If this file had to reword a description to satisfy the
vendor, this candidate would no longer be sitting the same exam.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_realtime_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }


# The ``tools`` array sent verbatim in session.update.
TOOLS: List[Dict[str, Any]] = [_as_realtime_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands Realtime, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the entries the model is sent still match the shared surface.
    """
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
