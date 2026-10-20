"""Adapt the shared tools to ElevenLabs' OpenAI-compatible function schema."""

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
    """Read back the declarations actually sent to ElevenLabs for parity checks."""
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
