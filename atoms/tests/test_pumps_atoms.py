"""The Atoms<->actor bridge, exercised against a fake websocket.

Atoms' client-tools docs don't document a truncated-arguments case the way
Hydra's OpenAI-Realtime-shaped events do, but the same shape of failure (a
barge-in cutting generation off mid-argument) is possible on any realtime
tool-calling wire, so the defensive handling in ``agent.py`` is tested the
same way Hydra's is.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock

import pytest

from rory_tools import CallSession

from rory_atoms import agent


class FakeAtoms:
    """Async-iterable stand-in for the Atoms websocket; records what we send."""

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

    def fake_dispatch(_session_state, name, args):
        calls.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    atoms_ws = FakeAtoms(events)
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_atoms_to_actor(atoms_ws, actor_ws, CallSession()))
    return calls, atoms_ws.sent, actor_ws


def test_tool_call_dispatches_and_replies_with_function_call_result(monkeypatch):
    events = [
        {
            "type": "function_call",
            "name": "verify_caller",
            "call_id": "call_1",
            "arguments": '{"account_number":"0000-00001","name_on_account":"Ana"}',
        },
    ]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == [("verify_caller", {"account_number": "0000-00001", "name_on_account": "Ana"})]
    assert sent == [{"type": "function_call.result", "call_id": "call_1", "output": json.dumps({"ok": True})}]


def test_truncated_arguments_are_skipped_and_bridging_continues(monkeypatch):
    truncated = '{"account_number":"0000-00001","name_on_account":"'
    events = [
        {"type": "function_call", "name": "verify_caller", "call_id": "call_cancelled", "arguments": truncated},
        {
            "type": "function_call",
            "name": "verify_caller",
            "call_id": "call_ok",
            "arguments": '{"account_number":"0000-00001","name_on_account":"Ana"}',
        },
    ]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == [("verify_caller", {"account_number": "0000-00001", "name_on_account": "Ana"})]
    assert [m["call_id"] for m in sent] == ["call_ok"]


def test_empty_arguments_still_dispatch_a_no_argument_tool(monkeypatch):
    events = [{"type": "function_call", "name": "get_account", "call_id": "call_1", "arguments": ""}]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert [m["call_id"] for m in sent] == ["call_1"]


def test_audio_deltas_are_forwarded_as_raw_bytes(monkeypatch):
    pcm = b"\x01\x02\x03\x04"
    events = [{"type": "output_audio.delta", "audio": base64.b64encode(pcm).decode()}]

    _calls, _sent, actor_ws = _run(events, monkeypatch)

    actor_ws.send_bytes.assert_called_once_with(pcm)


def test_interruption_is_logged_and_not_fatal(monkeypatch):
    events = [
        {"type": "interruption"},
        {"type": "function_call", "name": "get_account", "call_id": "call_1", "arguments": "{}"},
    ]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert [m["call_id"] for m in sent] == ["call_1"]


def test_error_event_raises(monkeypatch):
    events = [{"type": "error", "code": "internal_error", "message": "fixture failure"}]

    with pytest.raises(RuntimeError, match="fixture failure"):
        _run(events, monkeypatch)
