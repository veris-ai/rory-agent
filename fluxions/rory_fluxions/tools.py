"""Fluxions adapter for Rory's shared tool surface.

Fluxions takes bring-your-own functions in ``session.update`` as flat
``{name, description, parameters}`` objects. Everything that decides what a
tool *means* lives in ``rory_tools``; this module is only that wrapping:

    ToolSchema -> Fluxions tool dict

``ToolSchema.parameters()`` already produces the JSON-Schema object the shape
wants, so the adaptation is mechanical — which is the point. If this file had
to reword a description to satisfy the router, the candidates would no longer
be sitting the same exam. That matters more here than elsewhere: Fluxions
routes by prompting a model with the description, so the description *is* the
instruction.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }


TOOLS: List[Dict[str, Any]] = [_as_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands the model, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the dicts sent in ``session.update`` still match the shared declaration.
    """
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
