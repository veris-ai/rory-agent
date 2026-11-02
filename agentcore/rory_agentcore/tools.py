"""Rory's tool surface in the two shapes this candidate needs.

The 16 tools, their descriptions, implementations and the verification gate all
live in ``rory_tools`` and are shared byte for byte with the other transports.
This module only reshapes ``ToolSchema``:

- ``CONVERSE_TOOLS``: what the AgentCore Runtime hands Bedrock Converse. Exactly
  the shared declaration, so the model sits the same exam as every other candidate.
- ``GATEWAY_TOOLS``: what the AgentCore Gateway target declares. The shared
  declaration plus ``session_id``, which the runtime adds after the model has
  chosen a call, so the tools Lambda knows whose call it is.
"""

from __future__ import annotations

from typing import Any

from rory_tools import SCHEMAS

from .contract import SESSION_ARG

CONVERSE_TOOLS: list[dict[str, Any]] = [
    {
        "toolSpec": {
            "name": s.name,
            "description": s.description,
            "inputSchema": {"json": s.parameters()},
        }
    }
    for s in SCHEMAS
]


# The JSON Schema keywords a gateway target's inline tool schema accepts. The model
# reads the full shared schema from CONVERSE_TOOLS; the gateway copy only routes the
# call, so a constraint such as ``minimum`` is dropped from it, not from the exam.
GATEWAY_SCHEMA_KEYS = {"type", "properties", "required", "items", "description"}


def _for_gateway(node: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in node.items():
        if key not in GATEWAY_SCHEMA_KEYS:
            continue
        if key == "properties":
            value = {name: _for_gateway(prop) for name, prop in value.items()}
        elif key == "items":
            value = _for_gateway(value)
        out[key] = value
    return out


def _gateway_schema(schema) -> dict[str, Any]:
    params = _for_gateway(schema.parameters())
    return {
        "name": schema.name,
        "description": schema.description,
        "inputSchema": {
            "type": "object",
            "properties": {
                **params["properties"],
                SESSION_ARG: {"type": "string", "description": "The call this tool runs for."},
            },
            "required": [*params["required"], SESSION_ARG],
        },
    }


GATEWAY_TOOLS: list[dict[str, Any]] = [_gateway_schema(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What the runtime actually sends Bedrock, read back for the parity check."""
    return {
        tool["toolSpec"]["name"]: {
            "description": tool["toolSpec"]["description"],
            "required": list(tool["toolSpec"]["inputSchema"]["json"]["required"]),
            "properties": dict(tool["toolSpec"]["inputSchema"]["json"]["properties"]),
        }
        for tool in CONVERSE_TOOLS
    }
