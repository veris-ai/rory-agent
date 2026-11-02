"""Adapt the shared tools to Dialog-RSN-1's function schema.

The realtime API takes OpenAI Realtime's flat function entry, which is the
shape ``ToolSchema.parameters()`` already produces.

Two tools get one argument the shared declaration doesn't have: ``utterance``,
marked with PolyAI's ``poly_spoken`` extension. Dialog-RSN-1 writes text or
one tool call, never both, so without it the caller hears nothing while the
tool runs. The model fills ``utterance`` as part of the call and the server
speaks it at once. Every other argument, description and tool is the shared
one, and the shared implementations read only their own arguments, so
``utterance`` reaches dispatch unread.
"""

from rory_tools import SCHEMAS

SPOKEN_ARGUMENT = "utterance"

#: The tools that speak while they run, and what the model is told to say.
#: PolyAI's guide: only slow tools, or ones the caller should hear about.
SPOKEN_TOOLS = {
    "verify_caller": (
        "A short line to say to the caller while their details are checked, "
        "for example 'One moment while I check those details.'"
    ),
    "transfer_to_human": (
        "What to tell the caller as you transfer them."
    ),
}


def _parameters(schema) -> dict:
    parameters = schema.parameters()
    if schema.name in SPOKEN_TOOLS:
        parameters["properties"][SPOKEN_ARGUMENT] = {
            "type": "string",
            "description": SPOKEN_TOOLS[schema.name],
            "poly_spoken": True,
        }
        # Required: PolyAI notes the model often leaves an optional one empty.
        parameters["required"].append(SPOKEN_ARGUMENT)
    return parameters


TOOLS = [
    {
        "type": "function",
        "name": schema.name,
        "description": schema.description,
        "parameters": _parameters(schema),
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
