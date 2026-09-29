"""The receive loop against a scripted GPT-Live socket.

Audio deltas reach the actor as bytes, the greeting instruction goes out on
``session.started``, a delegated function call is dispatched through the shared
gate and answered with ``response.item.create`` + ``response.create`` once the
backend run completes, and ``session.closed`` ends the loop.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock

from rory_tools import CallSession

from rory_gpt_live import agent

AUDIO = bytes(4800)  # 100 ms of PCM16 at 24 kHz


class FakeLive:
    """Async-iterable stand-in for the GPT-Live websocket; records what we send."""

    def __init__(self, events: list[dict]):
        self._events = events
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        if event["type"] == "session.closed":
            # Let a tool task queued by an earlier event finish before the
            # session ends, as it would on a live call.
            await asyncio.sleep(0.05)
        return json.dumps(event)

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _envelope(delegation_id: str, nested: dict) -> dict:
    return {"type": "response.event", "delegation_id": delegation_id, "event": nested}


def _run(events: list[dict], monkeypatch, result=None):
    """Drive the receive loop; returns (dispatched calls, frames sent, the actor)."""
    calls: list[tuple[str, dict]] = []

    def fake_dispatch(session_state, name, args):
        calls.append((name, args))
        return result if result is not None else {"ok": True}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    live_ws = FakeLive(events)
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_live_to_actor(live_ws, actor_ws, CallSession()))
    return calls, live_ws.sent, actor_ws


def test_started_sends_the_greeting_and_audio_reaches_the_actor(monkeypatch):
    events = [
        {"type": "session.started", "session": {"id": "sess_1", "model": "gpt-live-1"}},
        {"type": "session.output_audio.delta", "delta": base64.b64encode(AUDIO).decode(),
         "start_ms": 0, "end_ms": 100},
        {"type": "session.output_transcript.delta", "delta": "Thanks for calling", "start_ms": 0,
         "end_ms": 100},
        {"type": "session.closed", "reason": "remote_hangup", "usage": {"seconds": 3}},
    ]

    calls, sent, actor_ws = _run(events, monkeypatch)

    assert calls == []
    assert sent == [agent._greeting_instruction()]
    actor_ws.send_bytes.assert_awaited_once_with(AUDIO)


def test_completed_backend_run_answers_its_function_call_then_continues(monkeypatch):
    events = [
        {"type": "session.delegation.created",
         "delegation": {"id": "item_d1", "target": "responses", "response_id": "resp_1"}},
        _envelope("item_d1", {"type": "response.created", "response": {"id": "resp_1"}}),
        _envelope("item_d1", {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "call_1", "name": "verify_caller",
                     "arguments": json.dumps({"account_number": "0000-00001"})},
        }),
        _envelope("item_d1", {"type": "response.completed",
                              "response": {"id": "resp_1", "usage": {"input_tokens": 10,
                                                                     "output_tokens": 5}}}),
        {"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 40}},
    ]

    calls, sent, _ = _run(events, monkeypatch, result={"verified": True})

    assert calls == [("verify_caller", {"account_number": "0000-00001"})]
    assert sent == [
        {
            "type": "response.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1",
                     "output": json.dumps({"verified": True})},
        },
        {"type": "response.create"},
    ]


def test_function_call_is_not_dispatched_before_the_run_completes(monkeypatch):
    events = [
        _envelope("item_d1", {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "call_1", "name": "get_account",
                     "arguments": "{}"},
        }),
        {"type": "session.closed", "reason": "remote_hangup", "usage": {"seconds": 1}},
    ]

    calls, sent, _ = _run(events, monkeypatch)

    assert calls == []
    assert sent == []


def test_text_only_backend_run_sends_nothing_back(monkeypatch):
    events = [
        _envelope("item_d1", {"type": "response.output_item.done",
                              "item": {"type": "message", "id": "msg_1"}}),
        _envelope("item_d1", {"type": "response.completed", "response": {"id": "resp_1"}}),
        {"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 2}},
    ]

    calls, sent, _ = _run(events, monkeypatch)

    assert calls == []
    assert sent == []


def test_live_error_is_fatal(monkeypatch):
    import pytest

    events = [{"type": "error", "error": {"type": "invalid_request_error", "code": "forbidden",
                                          "message": "Voice session access denied."}}]
    monkeypatch.setattr(agent, "dispatch", lambda *a: None)
    with pytest.raises(RuntimeError, match="access denied"):
        asyncio.run(agent._pump_live_to_actor(FakeLive(events), AsyncMock(), CallSession()))


def test_session_closed_from_the_vendor_reports_it(monkeypatch):
    monkeypatch.setattr(agent, "dispatch", lambda *a: None)
    closed = asyncio.run(agent._pump_live_to_actor(
        FakeLive([{"type": "session.closed", "reason": "expired", "usage": {"seconds": 9}}]),
        AsyncMock(), CallSession(),
    ))
    assert closed is True


def test_actor_hang_up_closes_the_session_and_reads_its_usage():
    live_ws = FakeLive([
        {"type": "session.usage.updated", "usage": {"seconds": 40}},
        {"type": "session.closed", "reason": "close_requested", "usage": {"seconds": 41}},
    ])
    asyncio.run(agent._close_session(live_ws))
    assert live_ws.sent == [{"type": "session.close"}]
    assert live_ws._events == []


def test_transcript_lines_close_on_a_pause_not_on_overlap(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(agent.logger, "info", lambda m: lines.append(m))
    t = agent._Transcript()
    t.add("rory", "Paperless billing", 1000, 1600)
    t.add("caller", "Did you do it?", 1400, 2000)  # over the top of Rory
    t.add("rory", " is on now.", 1600, 2200)
    t.add("caller", " Quote a three-month plan.", 2000, 3000)
    t.add("rory", "The three-month plan", 4500, 5200)  # a real pause: caller's line closes
    t.flush()
    said = [m.split("] ", 1)[1] for m in lines if "_said:" in m]
    assert said == [
        "rory_said: Paperless billing is on now.",
        "caller_said: Did you do it? Quote a three-month plan.",
        "rory_said: The three-month plan",
    ]
