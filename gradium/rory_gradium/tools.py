"""gradbot adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the gradbot-shaped wrapper:

    ToolSchema -> gradbot.ToolDef(name, description, parameters_json)

``parameters_json`` is the JSON-Schema object ``ToolSchema.parameters()``
already produces, serialised, so the adaptation is mechanical.
"""

from __future__ import annotations

import json
from typing import List

import gradbot

from rory_tools import SCHEMAS

TOOLS: List[gradbot.ToolDef] = [
    gradbot.ToolDef(schema.name, schema.description, json.dumps(schema.parameters())) for schema in SCHEMAS
]


def tool_surface() -> dict:
    """What this transport actually hands gradbot, read back for the parity check."""
    surface = {}
    for tool in TOOLS:
        parameters = json.loads(tool.parameters_json)
        surface[tool.name] = {
            "description": tool.description,
            "required": list(parameters["required"]),
            "properties": dict(parameters["properties"]),
        }
    return surface
