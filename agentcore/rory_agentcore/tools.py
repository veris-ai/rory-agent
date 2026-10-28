"""Tool declarations for the AgentCore Runtime request.

The 16 tools, their descriptions, implementations and the verification gate all
live in ``rory_tools`` and are shared byte for byte with the other transports.
This module only reshapes ``ToolSchema`` into the neutral ``{name, description,
parameters}`` object the AgentCore request carries, which the agent side maps
onto its model's own tool format. Nothing here rewords a description.
"""

from __future__ import annotations

from typing import Any

from rory_tools import SCHEMAS


def _declaration(schema) -> dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }


TOOLS: list[dict[str, Any]] = [_declaration(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually sends, read back for the parity check."""
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
