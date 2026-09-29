"""Vapi adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Vapi-shaped wrapper:

    ToolSchema    -> a ``function`` tool with a ``server.url``
    dispatch(...) -> the handler behind that URL

Vapi has no client-tool round trip that carries a result: a tool without a
server URL is delivered as a ``tool-calls`` notification and the model never
hears back. So every tool is a *server* tool. When the hosted LLM calls one,
Vapi POSTs a ``tool-calls`` envelope to the URL, waits for the HTTP response,
and adds each ``result`` string to the conversation. The URL is this process
(``rory_vapi.web``), reached through the tunnel ``rory_tools.tunnel`` opens, so
every call still resolves locally through the shared gate.

Vapi takes an OpenAI-style function declaration, which is the same shape
``ToolSchema.parameters()`` already produces, so the adaptation is mechanical —
which is the point. If this file had to reword a description to satisfy the
platform, the candidates would no longer be sitting the same exam.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from loguru import logger

from rory_tools import SCHEMAS, CallSession, dispatch
from rory_tools.schemas import ToolSchema


def _as_function_tool(schema: ToolSchema, server_url: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters(),
        },
        "server": {"url": server_url},
    }


def build_tools(server_url: str) -> List[Dict[str, Any]]:
    """The ``model.tools`` entry of the inline assistant, every tool POSTing to ``server_url``."""
    return [_as_function_tool(schema, server_url) for schema in SCHEMAS]


def extract_tool_call(call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    """``(call_id, name, args)`` from one ``toolCallList`` entry.

    The live envelope is OpenAI-shaped — ``{id, type: "function",
    function: {name, arguments}}`` — with ``arguments`` either a JSON string
    or an already-parsed object.
    """
    fn = call["function"]
    raw = fn.get("arguments")
    if isinstance(raw, str):
        args = json.loads(raw) if raw else {}
    else:
        args = dict(raw or {})
    return call["id"], fn["name"], args


async def run_tool_calls(session: CallSession, lock: asyncio.Lock, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve one ``tool-calls`` envelope into Vapi's ``results`` list.

    Calls run in the order the model emitted them, under the call's lock:
    a second envelope arriving while the first is still at the vendor must
    not race it on the same verification and payment state. The thread hop
    keeps the caller hearing the line while a blocking vendor client waits.

    ``result`` is a string field on Vapi's side — a dict is answered with
    "No result returned" and the model carries on with no observation.
    """
    results: List[Dict[str, Any]] = []
    async with lock:
        for call in tool_calls:
            call_id, name, args = extract_tool_call(call)
            result = await asyncio.to_thread(dispatch, session, name, args)
            logger.info(f"[tool] {name} -> {json.dumps(result, default=str)}")
            results.append({"toolCallId": call_id, "result": json.dumps(result, default=str)})
    return results


def tool_surface(server_url: str = "https://example.invalid/tool") -> Dict[str, dict]:
    """What this transport actually hands Vapi, read back for the parity check.

    Read out of ``build_tools`` rather than rebuilt from ``SCHEMAS``: the
    point is to prove the declarations sent with each call still match the
    shared surface.
    """
    return {
        tool["function"]["name"]: {
            "description": tool["function"]["description"],
            "required": list(tool["function"]["parameters"]["required"]),
            "properties": dict(tool["function"]["parameters"]["properties"]),
        }
        for tool in build_tools(server_url)
    }
