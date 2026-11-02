"""Prism adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Prism-shaped wrapper:

    ToolSchema -> rimelabs_sdk.realtime.ToolDefinition

A ``ToolDefinition`` takes a name, a description and a JSON Schema object, so
the shared ``parameters()`` object goes in unchanged. No tool sets
``prism_filler``: Prism writes its own short phrase ("One moment.") from the
tool's description while the tool runs, which is its default.
"""

from __future__ import annotations

from rimelabs_sdk.realtime import ToolDefinition

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_prism_tool(schema: ToolSchema) -> ToolDefinition:
    return ToolDefinition(
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters(),
    )


# The shared tools, handed to ``client.realtime.connect(tools=...)``.
TOOLS: list[ToolDefinition] = [_as_prism_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands Prism, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the definitions the model is sent still match the shared surface.
    """
    return {
        tool.name: {
            "description": tool.description,
            "required": list(tool.parameters["required"]),
            "properties": dict(tool.parameters["properties"]),
        }
        for tool in TOOLS
    }
