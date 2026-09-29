"""LiveKit Agents adapter for Rory's shared tool surface.

Everything that decides what a tool *means* — the schemas the model reads, the
implementations, the verification gate, the vendor error handling — lives in
``rory_tools`` and is shared with the other transports. This
module is only the LiveKit-shaped wrapper around it:

    ToolSchema      -> a raw-schema ``function_tool`` (RawFunctionTool)
    dispatch(...)   -> an async handler taking (RunContext, raw_arguments)

Raw-schema tools rather than the decorator-with-docstring form LiveKit
documents first: the decorator derives the parameter schema from a Python
signature and the description from a docstring, which would mean re-typing all
16 declarations by hand and trusting the derivation to match. A raw schema is
``ToolSchema.parameters()`` handed over verbatim, and the OpenAI plugin sends
it to the model unedited.

The session travels as LiveKit's ``userdata``: one ``CallSession`` per
``AgentSession``, read back by every handler as ``context.userdata`` and absent
from the tool schemas the model sees.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from livekit.agents import RunContext
from livekit.agents.llm import RawFunctionTool, function_tool
from livekit.agents.llm.tool_context import get_raw_function_info
from loguru import logger

from rory_tools import SCHEMAS, CallSession, dispatch
from rory_tools.schemas import ToolSchema


def _as_raw_tool(schema: ToolSchema) -> RawFunctionTool:
    """Wrap the shared dispatcher into a LiveKit raw function tool."""

    async def handle(context: RunContext, raw_arguments: Dict[str, Any]) -> Dict[str, Any]:
        session: CallSession = context.userdata
        args = dict(raw_arguments)
        # Every vendor client is blocking, and this runs on the job's event
        # loop — the same loop carrying STT, TTS and the room audio for this
        # call. Called directly, a ten-second vendor timeout is ten seconds of
        # dead air. The thread hop keeps the caller hearing the line.
        result = await asyncio.to_thread(dispatch, session, schema.name, args)
        logger.info(f"[tool] {schema.name} -> {json.dumps(result, default=str)}")
        return result

    return function_tool(
        handle,
        raw_schema={
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters(),
        },
    )


RORY_TOOLS: List[RawFunctionTool] = [_as_raw_tool(s) for s in SCHEMAS]


def tool_surface() -> dict:
    """What this transport actually hands LiveKit, read back for the parity check.

    Read out of ``RORY_TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is
    to prove the objects the model is sent still match the shared declaration.
    """
    surface = {}
    for tool in RORY_TOOLS:
        raw = get_raw_function_info(tool).raw_schema
        surface[raw["name"]] = {
            "description": raw["description"],
            "required": list(raw["parameters"]["required"]),
            "properties": dict(raw["parameters"]["properties"]),
        }
    return surface
