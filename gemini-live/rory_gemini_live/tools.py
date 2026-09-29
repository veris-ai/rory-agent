"""Gemini Live adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Gemini-shaped wrapper:

    ToolSchema -> google.genai types.FunctionDeclaration

Gemini takes an OpenAPI-style parameter schema, which is the same shape
``ToolSchema.parameters()`` already produces, so the adaptation is mechanical —
which is the point. If this file had to reword a description to satisfy Gemini,
the two candidates would no longer be sitting the same exam.
"""

from __future__ import annotations

from google.genai import types

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema


def _as_function_declaration(schema: ToolSchema) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters(),
    )


# One Tool carrying all 16 declarations, the shape live.connect expects.
TOOLS = [types.Tool(function_declarations=[_as_function_declaration(s) for s in SCHEMAS])]


def tool_surface() -> dict:
    """What this transport actually hands Gemini, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the declarations the model is sent still match the shared surface.
    Gemini's Schema objects come back typed, so each property is dumped to the
    plain JSON shape the shared declaration is written in — including the type
    name, which genai upper-cases on the way in.
    """
    surface = {}
    for declaration in TOOLS[0].function_declarations:
        properties = {}
        for name, schema in declaration.parameters.properties.items():
            dumped = schema.model_dump(mode="json", exclude_none=True)
            dumped["type"] = dumped["type"].lower()
            properties[name] = dumped
        surface[declaration.name] = {
            "description": declaration.description,
            "required": list(declaration.parameters.required),
            "properties": properties,
        }
    return surface
