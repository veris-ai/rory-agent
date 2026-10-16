"""The shared business tools, unchanged, in Dialt's client-tool format."""

from rory_tools import SCHEMAS

TOOLS = [
    {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
    }
    for schema in SCHEMAS
]


def tool_surface() -> dict:
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }
