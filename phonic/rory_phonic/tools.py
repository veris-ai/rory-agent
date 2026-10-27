"""Phonic WebSocket adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Phonic-shaped wrapper:

    ToolSchema -> one entry of the config ``tools`` array

Phonic declares an inline tool as a ``custom_websocket`` entry whose
``tool_schema`` is the OpenAI function shape, so the shared JSON-Schema
``parameters()`` object goes in unchanged. The tool is ``sync``: the pod answers
each ``tool_call`` with a ``tool_call_output`` before the model continues.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema

# Vendor calls are blocking and can run for seconds. Phonic's default output
# timeout is 5 s, which would turn a slow but correct vendor reply into a
# timed-out tool call.
TOOL_CALL_OUTPUT_TIMEOUT_MS = 30000


def _as_phonic_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "custom_websocket",
        "execution_mode": "sync",
        "tool_call_output_timeout_ms": TOOL_CALL_OUTPUT_TIMEOUT_MS,
        "tool_schema": {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters(),
                "strict": True,
            },
        },
    }


# The shared tools, the ``tools`` array sent in the config message.
TOOLS: List[Dict[str, Any]] = [_as_phonic_tool(s) for s in SCHEMAS]

# Phonic's built-in turn-taking tool. Phonic runs it itself: the model may choose not
# to answer, and after ``respond_after_sec`` of silence it speaks anyway. It is not
# a shared tool, so it is not in ``TOOLS`` or the parity surface.
BUILT_IN_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "built_in",
        "name": "choose_not_to_respond",
        "tool_config": {"respond_after_sec": 3},
    },
]


def tool_surface() -> dict:
    """What this transport actually hands Phonic, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the entries the model is sent still match the shared surface.
    """
    surface = {}
    for tool in TOOLS:
        function = tool["tool_schema"]["function"]
        surface[function["name"]] = {
            "description": function["description"],
            "required": list(function["parameters"]["required"]),
            "properties": dict(function["parameters"]["properties"]),
        }
    return surface
