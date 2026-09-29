"""ElevenLabs adapter for Rory's shared tool surface.

The 16 tools, their descriptions, their implementations and the verification
gate all live in ``rory_tools`` and are shared byte for byte with the other
transports. This module is only the ElevenLabs-shaped wrapper:

    ToolSchema    -> a ``client`` tool on the stored agent
    dispatch(...) -> a ``ClientTools`` handler

A *client* tool is one the platform does not run itself: when the hosted LLM
calls it, a ``client_tool_call`` event comes down the conversation socket, the
SDK invokes the handler registered here, and the result goes back up as a
``client_tool_result``. So the tool declarations are uploaded once at agent
creation, while every call still resolves locally through the shared gate.

ElevenLabs takes an OpenAPI-style parameter schema, which is the same shape
``ToolSchema.parameters()`` already produces, so the adaptation is mechanical —
which is the point. If this file had to reword a description to satisfy the
platform, the candidates would no longer be sitting the same exam.
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Dict, List

from elevenlabs import PromptAgentApiModelInputToolsItem_Client
from elevenlabs.conversational_ai.conversation import ClientTools
from loguru import logger

from rory_tools import SCHEMAS, CallSession, dispatch
from rory_tools.schemas import ToolSchema


def _as_client_tool(schema: ToolSchema) -> PromptAgentApiModelInputToolsItem_Client:
    return PromptAgentApiModelInputToolsItem_Client(
        name=schema.name,
        description=schema.description,
        parameters=schema.parameters(),
        # Hold the turn until the result is back. Without this the platform
        # keeps talking with no lookup to talk about.
        expects_response=True,
    )


# The ``tools`` entry of the stored agent's prompt config, the shape
# agents.create expects.
TOOLS: List[PromptAgentApiModelInputToolsItem_Client] = [_as_client_tool(s) for s in SCHEMAS]


def _handler(
    name: str, session: CallSession, lock: asyncio.Lock
) -> Callable[[dict], Awaitable[str]]:
    """Wrap the shared dispatcher into a ClientTools handler."""

    async def handle(parameters: dict) -> str:
        # The SDK folds its own correlation id into the model's arguments.
        args = {k: v for k, v in parameters.items() if k != "tool_call_id"}
        # The SDK schedules every client_tool_call as its own task, so two
        # calls in one turn would otherwise run side by side against the
        # same verification and payment state. The lock keeps them in the
        # order they were emitted; the thread hop keeps the caller hearing
        # the line while a blocking vendor client waits.
        async with lock:
            result = await asyncio.to_thread(dispatch, session, name, args)
        logger.info(f"[tool] {name} -> {json.dumps(result, default=str)}")
        # ``client_tool_result.result`` is a string field: a dict is refused
        # by the orchestrator as a 1008 policy violation.
        return json.dumps(result, default=str)

    return handle


def build_client_tools(session: CallSession, loop: asyncio.AbstractEventLoop) -> ClientTools:
    """One ``ClientTools`` per call, bound to that call's ``CallSession``.

    Handlers run on the conversation's own loop rather than the SDK's private
    thread, so ``session`` is only ever touched from one place.
    """
    tools = ClientTools(loop=loop)
    lock = asyncio.Lock()
    for schema in SCHEMAS:
        tools.register(schema.name, _handler(schema.name, session, lock), is_async=True)
    return tools


def tool_surface() -> Dict[str, dict]:
    """What this transport actually hands ElevenLabs, read back for the parity check.

    Read out of ``TOOLS`` rather than rebuilt from ``SCHEMAS``: the point is to
    prove the declarations uploaded to the agent still match the shared
    surface. The SDK parses each property into a typed schema model, so it is
    dumped back to the plain JSON shape the shared declaration is written in.
    """
    return {
        tool.name: {
            "description": tool.description,
            "required": list(tool.parameters.required),
            "properties": {
                name: schema.model_dump(mode="json", exclude_none=True)
                for name, schema in tool.parameters.properties.items()
            },
        }
        for tool in TOOLS
    }
