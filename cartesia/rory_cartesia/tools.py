"""Cartesia webhook-tool adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Cartesia-shaped wrapper:

    ToolSchema -> one Managed Agents webhook tool (``POST /v1/agents/tools``)

A webhook tool is one Cartesia executes by POSTing the model's arguments to a
URL, so every definition carries its URL and the bearer secret the pod checks.

The request body schema is ``ToolSchema.parameters()`` less one keyword:
Cartesia's body schema has no ``minimum`` and rejects a tool that carries one
("Unrecognized key"). It is the only keyword Rory's schemas use that Cartesia
cannot express, and the parity test pins exactly that.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rory_tools import SCHEMAS
from rory_tools.schemas import ToolSchema

# JSON-Schema keywords Cartesia's webhook body schema rejects.
UNSUPPORTED_KEYWORDS = frozenset({"minimum"})

# Rory's tools call two twins; the default 20 s is tight for a slow payment.
RESPONSE_TIMEOUT_S = 60


def _body_param(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in schema.items() if key not in UNSUPPORTED_KEYWORDS}


def _webhook_tool(schema: ToolSchema, url: str, secret: str) -> Dict[str, Any]:
    parameters = schema.parameters()
    return {
        "type": "webhook",
        "name": schema.name,
        "description": schema.description,
        "pre_tool_speech": "auto",
        "execution_mode": "immediate",
        "response_timeout_secs": RESPONSE_TIMEOUT_S,
        "api_schema": {
            "url": url,
            "method": "POST",
            "request_body_schema": {
                "type": "object",
                "properties": {name: _body_param(prop) for name, prop in parameters["properties"].items()},
                "required": parameters["required"],
            },
            "authentication": {"mode": "bearer", "token": {"type": "secret", "secret_value": secret}},
        },
    }


def build_tools(base_url: str, secret: str) -> List[Dict[str, Any]]:
    """One webhook tool per shared tool, each POSTing to ``<base_url>/<tool name>``."""
    return [_webhook_tool(schema, f"{base_url}/{schema.name}", secret) for schema in SCHEMAS]


def tool_surface(tools: List[Dict[str, Any]]) -> dict:
    """What this transport actually hands Cartesia, read back for the parity check."""
    return {
        tool["name"]: {
            "description": tool["description"],
            "required": list(tool["api_schema"]["request_body_schema"]["required"]),
            "properties": dict(tool["api_schema"]["request_body_schema"]["properties"]),
        }
        for tool in tools
    }
