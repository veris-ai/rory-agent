"""Atoms adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Atoms-shaped wrapper.

Unlike every other speech-to-speech candidate in this repo, Atoms' realtime
websocket does not take a ``tools`` array (or any per-connection
configuration at all — see ``agent.py``'s module docstring). A tool only
reaches this pod's ``function_call``/``function_call.result`` exchange if it
is registered as a **client tool** on the Atoms agent itself (console, or
the agent's versioned config), not as a server-side "API tool" — an API tool
is one Atoms calls directly over HTTP, which would bypass ``rory_tools``'s
verification gate and harness entirely and silently grade a different,
ungated agent. ``provisioning_payload()`` below is the exact shape to push for
each tool; there is no endpoint in Atoms' public API reference for setting an
agent's tools programmatically as of this writing, so pushing it is a manual
step — see the README.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema

# Atoms' client-tools docs (docs.smallest.ai/voice-agents/build/tools/client-tools)
# describe a tool as name/description/parameters plus two call-shape settings:
# whether the agent waits for a result, and the timeout if so. Every one of
# Rory's 16 tools returns something the model needs before it can speak next
# (a balance, a quote, a refusal), so all 16 wait for a response.
WAIT_FOR_RESPONSE = True
RESPONSE_TIMEOUT_SECONDS = 10


def _as_atoms_tool(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "name": schema.name,
        "description": schema.description,
        "parameters": schema.parameters(),
        "wait_for_response": WAIT_FOR_RESPONSE,
        "timeout_seconds": RESPONSE_TIMEOUT_SECONDS,
    }


# Not sent over the wire — Atoms has no message for it. Kept so
# ``tool_surface()`` has something built from, exactly as every other
# transport's ``TOOLS`` does, and so ``provisioning_payload()`` has one
# source of truth to print from.
TOOLS: List[Dict[str, Any]] = [_as_atoms_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport declares Atoms should have, read back for the parity check.

    Every other transport's ``tool_surface()`` reads values back out of the
    object it actually handed its vendor SDK at call time, because the point
    of the parity check is to prove the wire-sent surface matches
    ``rory_tools.schemas.SCHEMAS``. Atoms has no such object — its tools live
    in agent config, provisioned out of band — so this instead proves that
    ``provisioning_payload()``, the thing a human actually pushes to the
    agent, matches the shared declaration. A transport that cannot assert the
    stronger claim should not pretend to.
    """
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["parameters"]["required"]),
            "properties": dict(tool["parameters"]["properties"]),
        }
        for tool in TOOLS
    }


def provisioning_payload() -> List[Dict[str, Any]]:
    """The exact client-tool definitions to push onto the Atoms agent.

    Print this (``python -m rory_atoms.tools``) and paste it into the
    agent's Tools tab, or into whatever field the versioned agent-config API
    turns out to accept — see the README for why that field name is not
    pinned here.
    """
    return TOOLS


if __name__ == "__main__":
    import json

    print(json.dumps(provisioning_payload(), indent=2))
