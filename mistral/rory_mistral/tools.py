"""Mistral adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Mistral-shaped wrapper:

    ToolSchema -> {"type": "function", "function": {name, description, parameters}}

Mistral's chat-completions API takes tools in the OpenAI function-calling
shape, and ``ToolSchema.parameters()`` already produces the parameter object
it expects, so the adaptation is mechanical — which is the point. If this file
had to reword a description to satisfy Mistral, the candidates would no longer
be sitting the same exam.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_function_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters(),
        },
    }


TOOLS: List[Dict[str, Any]] = [_as_function_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands Mistral, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the objects the model is sent still match the shared declaration.
    """
    return {
        tool["function"]["name"]: {
            "description": tool["function"]["description"],
            "required": list(tool["function"]["parameters"]["required"]),
            "properties": dict(tool["function"]["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
