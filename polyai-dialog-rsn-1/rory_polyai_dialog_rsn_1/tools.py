"""Adapt the shared tools to Dialog-RSN-1's function schema.

The realtime API takes OpenAI Realtime's flat function entry, which is the
shape ``ToolSchema.parameters()`` already produces, so the adaptation is
mechanical.

PolyAI's ``poly_spoken`` extension (a required string argument the model
speaks while the tool runs) is deliberately not added: it would give every
tool an argument the other candidates' tools do not have.
"""

from rory_tools import SCHEMAS

TOOLS = [
    {
        "type": "function",
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }
    for schema in SCHEMAS
]


def tool_surface() -> dict:
    """Read back the declarations actually sent to Dialog-RSN-1 for parity checks."""
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
