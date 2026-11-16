"""Atoms adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Atoms-shaped wrapper.

Unlike every other speech-to-speech candidate in this repo, Atoms' realtime
websocket does not take a ``tools`` array (or any per-connection
configuration at all — see ``agent.py``'s module docstring). A tool only
reaches this pod's ``function_call``/``function_call.result`` exchange if it
is registered as a **client tool** on the Atoms agent itself, not as a
server-side "API tool" — an API tool is one Atoms calls directly over HTTP,
which would bypass ``rory_tools``'s verification gate and harness entirely
and silently grade a different, ungated agent.

This IS scriptable — confirmed against the live API, not docs (which don't
enumerate the schema). The shape below is the one a real agent echoed back
once accepted, pushed through the same
``PUT /agent/{id}/branches/{branchId}/draft`` endpoint as the prompt and
turn-taking settings, nested as ``{"type": "single_prompt",
"singlePromptConfig": {"prompt": ..., "tools": [...]}}``. A bare top-level
``tools`` field (or ``workflowTools``, or nested under a plain ``workflow``
key) is rejected with ``"No recognized config fields in request body"`` — the
``type``/``singlePromptConfig`` wrapper is what makes it land. See
``atoms/scripts/provision_agent.py``, which pushes this for real.

The wire shape differs from ``ToolSchema`` in two ways, both load-bearing:
``type`` must be the literal ``"client_tool"`` (the full discriminator set,
from a validation error: ``integration_tool | end_call | transfer_call |
api_call | extract_dynamic_variables | knowledge_base_search |
send_whatsapp_message | client_tool``), and each parameter's ``type`` is one
of Atoms' own four (``text | number | boolean | enum``), not JSON Schema's —
so ``_TYPE_MAP`` translates. Atoms' parameter schema also has no field for a
JSON Schema constraint like ``minimum`` (``make_payment.amount_cents`` has
one) — there is nothing to push it into, not an oversight here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema

# JSON Schema -> Atoms parameter type. Checked against every property in
# every one of Rory's 16 tools (string, integer, boolean only — see
# atoms/tests/test_parity_atoms.py) rather than assumed complete; add a case
# only once a schema actually uses it, so a silently-wrong mapping can't ship
# for a type nobody tested.
_TYPE_MAP = {"string": "text", "integer": "number", "boolean": "boolean"}

# Every one of Rory's 16 tools returns something the model needs before it
# can speak next (a balance, a quote, a refusal), so all expect a response.
EXPECTS_RESPONSE = True
TIMEOUT_MS = 10_000


def _as_atoms_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "type": "client_tool",
        "enabled": True,
        "llmParameters": [
            {
                "name": pname,
                "description": prop.get("description", ""),
                "type": _TYPE_MAP[prop["type"]],
                "required": pname in schema.required,
            }
            for pname, prop in schema.properties.items()
        ],
        "timeoutMs": TIMEOUT_MS,
        "expectsResponse": EXPECTS_RESPONSE,
        "fillerPhrases": [],
    }


# Sent verbatim, nested under singlePromptConfig.tools — see
# atoms/scripts/provision_agent.py.
TOOLS: List[Dict[str, Any]] = [_as_atoms_tool(s) for s in SCHEMAS]

# make_payment.amount_cents is the one property with a JSON Schema
# constraint Atoms' parameter shape has no field for (see module docstring).
# Excluded from the per-property equality check in test_parity_atoms.py,
# which documents exactly what's dropped and why — not excluded here, so
# this module stays data, not test logic.
_UNREPRESENTABLE_CONSTRAINTS = {("make_payment", "amount_cents"): {"minimum"}}


def tool_surface() -> dict:
    """What was actually sent, read back for the parity check.

    Reconstructs each parameter's JSON-Schema-shaped entry by inverting
    ``_TYPE_MAP``, so the parity check proves the real wire format round-trips
    to the shared declaration — not that ``TOOLS`` matches itself.
    """
    inverse_type = {v: k for k, v in _TYPE_MAP.items()}
    surface = {}
    for tool in TOOLS:
        properties = {}
        required = []
        for param in tool["llmParameters"]:
            properties[param["name"]] = {
                "type": inverse_type[param["type"]],
                "description": param["description"],
            }
            if param["required"]:
                required.append(param["name"])
        surface[tool["name"]] = {
            "description": tool["description"],
            "required": required,
            "properties": properties,
        }
    return surface


def provisioning_payload() -> List[Dict[str, Any]]:
    """The exact client-tool definitions to push onto the Atoms agent.

    Pushed by ``atoms/scripts/provision_agent.py``; print this directly
    (``python -m rory_atoms.tools``) only if you need to paste it into the
    console by hand instead.
    """
    return TOOLS


if __name__ == "__main__":
    import json

    print(json.dumps(provisioning_payload(), indent=2))
