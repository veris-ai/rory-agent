"""PolyAI Agent Studio adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the Agent Studio-shaped wrapper:

    ToolSchema    -> one Agent Studio function file, ``functions/<name>.py``
    dispatch(...) -> the handler behind the ``/tool`` webhook those files POST to

Agent Studio has no per-call tool declaration and no client-tool round trip.
A tool is a Python function inside the project, executed in PolyAI's cloud,
described to the model by ``@func_description`` / ``@func_parameter`` and
typed by its signature. So every shared tool is rendered as one such function
whose body POSTs the ``{"name", "args", "conversation_id"}`` envelope to this
process and returns the JSON reply as context for the model's next turn. The
description strings are the shared ones verbatim; the parameter types are the
shared JSON-Schema types mapped onto the four PolyAI accepts.

Two things the shared declaration says cannot be said to Agent Studio, and the
parity run names them rather than papering over them: a parameter cannot be
optional (every one is required, with no default), and a parameter carries a
type and a description and nothing else, so ``minimum`` on
``make_payment.amount_cents`` has no place. The rendered body enforces the
first instead — an optional parameter the model leaves empty is dropped from
the envelope — and the shared implementation refuses a non-positive amount.
"""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Any, Dict

from loguru import logger

from rory_tools import SCHEMAS, CallSession, dispatch
from rory_tools.schemas import ToolSchema

# JSON-Schema type -> the Python annotation Agent Studio maps back to it.
_PY_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}
_JSON_TYPES = {py: js for js, py in _PY_TYPES.items()}


def render_function(schema: ToolSchema, tool_url: str) -> str:
    """One shared tool as the text of an Agent Studio function file.

    ``from _gen import *`` is the ADK's required first line and is not pushed.
    The body uses only the standard library: it runs in PolyAI's runtime, not
    in this image. The result is returned as a string, which Agent Studio
    injects as system context for the model's next turn.
    """
    lines = [
        "from _gen import *  # <AUTO GENERATED>",
        "",
        "import json",
        "import urllib.request",
        "",
        "",
        f"@func_description({schema.description!r})",
    ]
    for name, prop in schema.properties.items():
        lines.append(f"@func_parameter({name!r}, {prop['description']!r})")
    params = [f"{name}: {_PY_TYPES[prop['type']]}" for name, prop in schema.properties.items()]
    signature = ", ".join(["conv: Conversation", *params])
    args_literal = ", ".join(f"{name!r}: {name}" for name in schema.properties)
    lines += [
        f"def {schema.name}({signature}):",
        f"    args = {{{args_literal}}}",
    ]
    for name in schema.properties:
        if name not in schema.required:
            # Agent Studio has no optional parameters; an empty string is the
            # model's only way to leave one out.
            lines += [
                f"    if {name} == '':",
                f"        del args[{name!r}]",
            ]
    lines += [
        "    req = urllib.request.Request(",
        f"        {tool_url!r},",
        f"        data=json.dumps({{'name': {schema.name!r}, 'args': args, 'conversation_id': conv.id}}).encode(),",
        "        headers={'Content-Type': 'application/json'},",
        "    )",
        "    with urllib.request.urlopen(req, timeout=30) as resp:",
        "        return resp.read().decode()",
        "",
    ]
    return "\n".join(lines)


def render_functions(tool_url: str) -> Dict[str, str]:
    """``{function name: file text}`` for the whole surface."""
    return {schema.name: render_function(schema, tool_url) for schema in SCHEMAS}


class LiveCall:
    """One Agent Studio conversation as the webhook sees it: its session and its ordering lock."""

    def __init__(self) -> None:
        self.session = CallSession()
        self.lock = asyncio.Lock()


# PolyAI conversation id -> the call the /tool webhook resolves against. The
# gateway never tells the bridge which conversation it opened, so the registry
# fills in from the first webhook of each conversation instead of from the
# /voice handler. Two conversations on one process still cannot share a session.
LIVE_CALLS: Dict[str, LiveCall] = {}


async def run_tool_call(conversation_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one envelope from a rendered function through the shared gate.

    Under the conversation's lock: a second call arriving while the first is
    still at the vendor must not race it on the same verification and payment
    state. The thread hop keeps the caller hearing the line while a blocking
    vendor client waits.
    """
    live = LIVE_CALLS.setdefault(conversation_id, LiveCall())
    async with live.lock:
        result = await asyncio.to_thread(dispatch, live.session, name, args)
    logger.info(f"[tool] {name} ({conversation_id}) -> {json.dumps(result, default=str)}")
    return result


def tool_surface(tool_url: str = "https://example.invalid/tool") -> Dict[str, dict]:
    """What this transport actually hands Agent Studio, read back for the parity check.

    Parsed out of the rendered function files rather than rebuilt from
    ``SCHEMAS``: the point is to prove the declarations pushed to the project
    still match the shared surface. Every parameter is reported as required
    because that is what Agent Studio tells the model.
    """
    surface: Dict[str, dict] = {}
    for source in render_functions(tool_url).values():
        fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
        description = None
        parameter_descriptions: Dict[str, str] = {}
        for decorator in fn.decorator_list:
            if decorator.func.id == "func_description":
                description = decorator.args[0].value
            elif decorator.func.id == "func_parameter":
                parameter_descriptions[decorator.args[0].value] = decorator.args[1].value
        params = [arg for arg in fn.args.args if arg.arg != "conv"]
        surface[fn.name] = {
            "description": description,
            "required": [arg.arg for arg in params],
            "properties": {
                arg.arg: {
                    "type": _JSON_TYPES[arg.annotation.id],
                    "description": parameter_descriptions[arg.arg],
                }
                for arg in params
            },
        }
    return surface
