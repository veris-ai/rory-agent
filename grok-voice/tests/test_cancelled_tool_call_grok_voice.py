"""A function call cut off by server VAD is skipped, not dispatched, not fatal.

Grok speaks the Realtime protocol, so it has the Realtime failure: when the
caller speaks over a tool-generating response, ``response.function_call_arguments.done``
still arrives with the arguments string ending wherever generation stopped.
``json.loads`` on it killed the call. The bridge must treat that event as a
cancelled call and keep bridging.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

from rory_tools import CallSession

from rory_grok_voice import agent


class FakeGrok:
    """Async-iterable stand-in for the Grok websocket; records what we send."""

    def __init__(self, events: list[dict]):
        self._events = events
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return json.dumps(self._events.pop(0))

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _run(events: list[dict], monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_dispatch(session_state, name, args):
        calls.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    gk_ws = FakeGrok(events)
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_grok_to_actor(gk_ws, actor_ws, CallSession()))
    return calls, gk_ws.sent


def test_truncated_arguments_are_skipped_and_bridging_continues(monkeypatch):
    truncated = '{"account_number":"0000-00001","name_on_account":"'
    events = [
        {"type": "response.created"},
        {"type": "input_audio_buffer.speech_started"},
        {
            "type": "response.function_call_arguments.done",
            "name": "verify_caller",
            "call_id": "call_cancelled",
            "arguments": truncated,
        },
        {"type": "response.done"},
        {"type": "response.created"},
        {
            "type": "response.function_call_arguments.done",
            "name": "verify_caller",
            "call_id": "call_ok",
            "arguments": '{"account_number":"0000-00001","name_on_account":"Ana"}',
        },
        {"type": "response.done"},
    ]

    calls, sent = _run(events, monkeypatch)

    assert calls == [("verify_caller", {"account_number": "0000-00001", "name_on_account": "Ana"})]
    outputs = [m for m in sent if m["type"] == "conversation.item.create"]
    assert [m["item"]["call_id"] for m in outputs] == ["call_ok"]
    # The cancelled response's response.done must not kick a response.create;
    # only the completed tool round does.
    assert [m["type"] for m in sent].count("response.create") == 1


def test_empty_arguments_still_dispatch_a_no_argument_tool(monkeypatch):
    events = [
        {"type": "response.created"},
        {
            "type": "response.function_call_arguments.done",
            "name": "get_account",
            "call_id": "call_1",
            "arguments": "",
        },
        {"type": "response.done"},
    ]

    calls, sent = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert [m["type"] for m in sent] == ["conversation.item.create", "response.create"]
