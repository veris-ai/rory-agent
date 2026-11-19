"""The Atoms<->actor bridge, exercised against a fake websocket."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock

import pytest
import websockets
from websockets.frames import Close, CloseCode

from rory_tools import CallSession

from rory_atoms import agent


class FakeAtoms:
    """Async-iterable stand-in for the Atoms websocket; records what we send.

    Events with no internal ``await`` run back-to-back in one scheduling
    step, so pushing audio then an interruption in the same event list
    reliably exercises "dropped before it was ever sent".
    """

    def __init__(self, events: list[dict], raise_on_exhaustion: Exception | None = None):
        self._events = events
        self._raise_on_exhaustion = raise_on_exhaustion
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            if self._raise_on_exhaustion is not None:
                raise self._raise_on_exhaustion
            raise StopAsyncIteration
        return json.dumps(self._events.pop(0))

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _run(events, monkeypatch, raise_on_exhaustion=None, dispatch=None):
    calls: list[tuple[str, dict]] = []

    def default_dispatch(_session_state, name, args):
        calls.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", dispatch or default_dispatch)
    atoms_ws = FakeAtoms(events, raise_on_exhaustion)
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


def test_truncated_arguments_are_still_answered(monkeypatch):
    """Unlike skipping a broken call outright, every function_call gets a
    function_call.result — the model has no way to recover from a call it
    never heard back from."""
    truncated = '{"account_number":"0000-00001","name_on_account":"'
    events = [
        {"type": "function_call", "name": "verify_caller", "call_id": "call_cancelled", "arguments": truncated},
    ]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == []
    assert len(sent) == 1
    assert sent[0]["call_id"] == "call_cancelled"
    assert "error" in json.loads(sent[0]["output"])


def test_empty_arguments_still_dispatch_a_no_argument_tool(monkeypatch):
    events = [{"type": "function_call", "name": "get_account", "call_id": "call_1", "arguments": ""}]

    calls, sent, _actor = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert [m["call_id"] for m in sent] == ["call_1"]


def test_tool_calls_run_sequentially_in_arrival_order(monkeypatch):
    """Two calls each 'slow' by 0.1s must take ~0.2s: both run on the same
    CallSession, so a second call starting before the first answered would
    race that shared state."""

    order: list[str] = []

    def slow_dispatch(_session_state, name, _args):
        order.append(name)
        time.sleep(0.1)
        return {"ok": True}

    events = [
        {"type": "function_call", "name": "get_account", "call_id": "a", "arguments": "{}"},
        {"type": "function_call", "name": "list_bills", "call_id": "b", "arguments": "{}"},
    ]

    start = time.monotonic()
    _calls, sent, _actor = _run(events, monkeypatch, dispatch=slow_dispatch)
    elapsed = time.monotonic() - start

    assert order == ["get_account", "list_bills"]
    assert [m["call_id"] for m in sent] == ["a", "b"]
    assert elapsed >= 0.2, f"took {elapsed:.3f}s — looks concurrent, not sequential"


def test_audio_is_paced_and_forwarded(monkeypatch):
    pcm = b"\x01\x02\x03\x04"
    events = [
        {"type": "output_audio.delta", "audio": base64.b64encode(pcm).decode()},
        {"type": "session.closed", "reason": "ended"},
    ]

    _calls, _sent, actor_ws = _run(events, monkeypatch)

    actor_ws.send_bytes.assert_called_once_with(pcm)


def test_pacer_clock_restarts_after_a_silence_gap():
    """A real gap between chunks must not leave the pacer scheduling off a
    stale start time — that's what makes everything after the first pause go
    out with no pacing at all, since the (unreset) target is already in the
    past by the time audio resumes."""
    frame_s = 0.2
    pcm = b"\x00" * int(agent.ACTOR_RATE_HZ * agent.BYTES_PER_SAMPLE * frame_s)
    actor_ws = AsyncMock()

    async def scenario():
        paced = agent._PacedOutput(actor_ws)
        paced.push(pcm)
        await paced.drain()
        await asyncio.sleep(0.5)  # a real silence gap, well past OUTPUT_LOOKAHEAD_S

        start = time.monotonic()
        for _ in range(3):
            paced.push(pcm)
        await paced.drain()
        elapsed = time.monotonic() - start
        await paced.aclose()
        return elapsed

    elapsed = asyncio.run(scenario())
    # 3 * 0.2s = 0.6s of audio outpaces the 0.3s lookahead by 0.3s once the
    # clock has correctly restarted; a clock that never reset would instead
    # flood all three out with ~0 pacing.
    assert elapsed > 0.2, f"took {elapsed:.3f}s — chunks after a gap were not paced, clock never reset"


def test_interruption_drops_audio_queued_but_not_yet_sent(monkeypatch):
    pcm = b"\x01\x02\x03\x04"
    events = [
        {"type": "output_audio.delta", "audio": base64.b64encode(pcm).decode()},
        {"type": "interruption"},
    ]

    _calls, _sent, actor_ws = _run(events, monkeypatch)

    actor_ws.send_bytes.assert_not_called()


def test_normal_session_closed_ends_quietly(monkeypatch):
    events = [{"type": "session.closed", "reason": "ended"}]
    _run(events, monkeypatch)  # must not raise


def test_abnormal_session_closed_is_raised_as_a_failure(monkeypatch):
    events = [{"type": "session.closed", "reason": "setup_failed"}]
    with pytest.raises(RuntimeError, match="abnormally"):
        _run(events, monkeypatch)


def test_connection_close_not_preceded_by_ended_is_a_failure_even_at_code_1000(monkeypatch):
    """A clean-looking close (1000/1001/1005) is not proof the agent ended
    the call normally — only a session.closed(ended) we actually saw is.
    Reaching this path at all means we never saw one."""
    close = websockets.ConnectionClosedOK(Close(CloseCode.NORMAL_CLOSURE, ""), None)
    with pytest.raises(RuntimeError, match="abnormally"):
        _run([], monkeypatch, raise_on_exhaustion=close)


def test_abnormal_connection_close_is_raised_as_a_failure(monkeypatch):
    close = websockets.ConnectionClosedError(Close(CloseCode.ABNORMAL_CLOSURE, "boom"), None)
    with pytest.raises(RuntimeError, match="abnormally"):
        _run([], monkeypatch, raise_on_exhaustion=close)


@pytest.mark.parametrize("code", ["tool_timeout", "unknown_call_id", "result_too_large", "some_future_code"])
def test_error_events_reaching_the_live_loop_never_fail_the_call(monkeypatch, code):
    """Setup-time failures (a rejected tool declaration, a bad connect) are
    raised in _connect_once/_declare_tools, before this loop ever starts —
    so every error this loop sees is scoped to one call or one tool call,
    known code or not, and never tears the session down."""
    events = [
        {"type": "error", "code": code, "message": "fixture"},
        {"type": "session.closed", "reason": "ended"},
    ]

    _run(events, monkeypatch)  # must not raise
