"""Hugging Face adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the chat-completions-shaped wrapper:

    ToolSchema -> {"type": "function", "function": {name, description, parameters}}

The router's ``/v1/chat/completions`` — and a vLLM endpoint behind
``HF_LLM_URL`` — take tools in the OpenAI function-calling shape, which is the
same object ``ToolSchema.parameters()`` already produces, so the adaptation is
a plain dict wrap. If this file had to reword a description to satisfy the
model, the candidates would no longer be sitting the same exam.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_chat_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters(),
        },
    }


# The ``tools`` array sent with every chat completion.
TOOLS: List[Dict[str, Any]] = [_as_chat_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually sends the LLM, read back for the parity check.

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
