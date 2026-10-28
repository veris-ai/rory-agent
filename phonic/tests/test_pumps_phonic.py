"""The Phonic pumps against fake sockets: tool concurrency and close semantics.

The parity tests check the declared surface. These drive the receive loop, the tool
worker and the close paths, which a live call exercises but a parity check can't.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import pytest
import websockets
from websockets.frames import OK_CLOSE_CODES, Close

from rory_phonic import agent


class FakePhonic:
    """Yields scripted Phonic messages, then closes the way websockets does: the OK
    codes raise ConnectionClosedOK, every other code raises ConnectionClosedError."""

    def __init__(self, messages, close_code=agent.NORMAL_CLOSE):
        self._messages = list(messages)
        self._close_code = close_code
        self.sent: list[dict] = []

    async def recv(self):
        if self._messages:
            return json.dumps(self._messages.pop(0))
        await asyncio.sleep(0)
        rcvd = Close(self._close_code, "test close")
        if self._close_code in OK_CLOSE_CODES:
            raise websockets.ConnectionClosedOK(rcvd, None)
        raise websockets.ConnectionClosedError(rcvd, None)

    async def send(self, raw):
        self.sent.append(json.loads(raw))


class FakeActor:
    """Records audio the candidate sends back toward the Veris actor."""

    def __init__(self):
        self.audio: list[bytes] = []

    async def send_bytes(self, data: bytes):
        self.audio.append(data)


def _audio(text: str) -> dict:
    return {"type": "audio_chunk", "audio": base64.b64encode(b"\x01\x00").decode(), "text": text}


def _run_pumps(messages, close_code, dispatch_fn, monkeypatch):
    monkeypatch.setattr(agent, "dispatch", dispatch_fn)

    async def go():
        ph = FakePhonic(messages, close_code)
        actor = FakeActor()
        queue: asyncio.Queue = asyncio.Queue()
        worker = asyncio.create_task(agent._run_tool_calls(ph, object(), queue))
        try:
            await agent._pump_phonic_to_actor(ph, actor, queue)
            # Wait for the worker to finish every call the pump queued.
            await asyncio.wait_for(queue.join(), timeout=10)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        return ph, actor

    return asyncio.run(go())


def test_upstream_abnormal_close_is_raised_not_swallowed(monkeypatch):
    with pytest.raises(websockets.ConnectionClosed) as info:
        _run_pumps([_audio("Hello.")], 1011, lambda *a: {}, monkeypatch)
    assert info.value.rcvd.code == 1011


@pytest.mark.parametrize("close_code", [1001, 1005])
def test_upstream_close_that_plain_iteration_would_hide_is_raised(close_code, monkeypatch):
    """A plain `async for` ends quietly on 1001 and 1005; the receive loop must raise."""
    with pytest.raises(websockets.ConnectionClosed) as info:
        _run_pumps([_audio("Hello.")], close_code, lambda *a: {}, monkeypatch)
    assert info.value.rcvd.code == close_code


def test_upstream_normal_close_returns(monkeypatch):
    ph, actor = _run_pumps([_audio("Hello.")], agent.NORMAL_CLOSE, lambda *a: {}, monkeypatch)
    assert actor.audio == [b"\x01\x00"]


def test_slow_tool_does_not_block_audio_or_interruptions(monkeypatch):
    """Audio after a tool_call reaches the actor before that tool's output is sent."""
    order: list[str] = []

    def slow_dispatch(session, name, args):
        time.sleep(0.3)
        order.append("tool-done")
        return {"ok": True}

    messages = [
        {"type": "tool_call", "tool_call_id": "c1", "tool_name": "get_account", "parameters": {}},
        _audio("one"),
        _audio("two"),
    ]
    ph, actor = _run_pumps(messages, agent.NORMAL_CLOSE, slow_dispatch, monkeypatch)

    assert len(actor.audio) == 2, "audio kept flowing while the tool ran"
    assert ph.sent == [{"type": "tool_call_output", "tool_call_id": "c1", "output": {"ok": True}}]
    assert order == ["tool-done"]


def test_output_is_sent_even_if_phonic_reports_the_call_interrupted(monkeypatch):
    """A tool that has run has acted, so Phonic gets its output even after an interrupt.

    Dropping it is how one $25 repayment got recorded twice: Phonic never heard the
    first one went through and asked again.
    """

    def slow_dispatch(session, name, args):
        time.sleep(0.3)
        return {"ok": True}

    messages = [
        {"type": "tool_call", "tool_call_id": "c2", "tool_name": "modify_payment_arrangement", "parameters": {}},
        {"type": "tool_call_interrupted", "tool_call_id": "c2", "tool_name": "modify_payment_arrangement"},
    ]
    ph, _ = _run_pumps(messages, agent.NORMAL_CLOSE, slow_dispatch, monkeypatch)

    assert ph.sent == [{"type": "tool_call_output", "tool_call_id": "c2", "output": {"ok": True}}]


def test_tool_calls_run_in_order(monkeypatch):
    ran: list[str] = []

    def record(session, name, args):
        ran.append(args["n"])
        time.sleep(0.05)
        return {"n": args["n"]}

    messages = [
        {"type": "tool_call", "tool_call_id": f"c{i}", "tool_name": "get_account", "parameters": {"n": i}}
        for i in range(3)
    ]
    ph, _ = _run_pumps(messages, agent.NORMAL_CLOSE, record, monkeypatch)

    assert ran == [0, 1, 2]
    assert [m["tool_call_id"] for m in ph.sent] == ["c0", "c1", "c2"]
