"""Pipecat adapter for Rory's shared tool surface.

Everything that decides what a tool *means* — the schemas the model reads, the
implementations, the verification gate, the vendor error handling — lives in ``rory_tools`` and is shared with every other transport.
This module is only the Pipecat-shaped wrapper around it:

    ToolSchema      -> pipecat FunctionSchema / ToolsSchema
    dispatch(...)   -> an async handler taking FunctionCallParams

The session travels as Pipecat's ``app_resources``: one object per
``PipelineTask``, handed to every handler as ``params.app_resources`` and
absent from the tool schemas the model sees.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Dict

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from rory_tools import SCHEMAS, CallSession, dispatch
from rory_tools.schemas import ToolSchema


def _as_function_schema(schema: ToolSchema) -> FunctionSchema:
    return FunctionSchema(
        name=schema.name,
        description=schema.description,
        properties=dict(schema.properties),
        required=list(schema.required),
    )


RORY_TOOLS = ToolsSchema(standard_tools=[_as_function_schema(s) for s in SCHEMAS])


def _handler(name: str) -> Callable:
    """Wrap the shared dispatcher into a Pipecat handler."""

    async def handle(params: FunctionCallParams) -> None:
        session: CallSession = params.app_resources
        args = dict(params.arguments)
        # Every vendor client is blocking, and this runs on the pipeline's
        # event loop — the same loop carrying audio, VAD and TTS for this call.
        # Called directly, a ten-second vendor timeout is ten seconds of dead
        # air. The thread hop keeps the caller hearing the line.
        result = await asyncio.to_thread(dispatch, session, name, args)
        await params.result_callback(result)

    return handle


HANDLERS: Dict[str, Callable] = {s.name: _handler(s.name) for s in SCHEMAS}


def tool_surface() -> dict:
    """What this transport actually hands Pipecat, read back for the parity check.

    Read out of ``RORY_TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is
    to prove the objects the model is sent still match the shared declaration.
    """
    return {
        tool.name: {
            "description": tool.description,
            "required": list(tool.required),
            "properties": dict(tool.properties),
        }
        for tool in RORY_TOOLS.standard_tools
    }
