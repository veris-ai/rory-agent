"""What the Fluxions bridge does with the realtime socket, without a vendor.

The parity run proves the declaration; these prove the round trip — a
``tool.call`` is answered through the shared dispatcher with the id it came in
on, barge-in drops what the caller has not heard, and a session at the wrong
rates fails instead of transcribing badly.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rory_fluxions import agent


class _Fluxions:
    """A realtime socket that plays back a scripted event stream."""

    def __init__(self, events):
        self._events = events
        self.send = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        return event if isinstance(event, bytes) else json.dumps(event)


def _session_created(mic=16000, out=24000):
    return {"type": "session.created", "session_id": "s1", "model": "m", "voice": "maeve",
            "mic_sample_rate": mic, "audio_sample_rate": out}


def _call(monkeypatch, events):
    monkeypatch.setattr(agent, "_greeting_pcm", b"\x00" * 48000, raising=False)
    return agent._Call(SimpleNamespace(client=None, send_bytes=AsyncMock()), _Fluxions(events))


def test_tool_calls_are_answered_in_order_with_their_call_id(monkeypatch):
    seen = []

    def dispatch(session, name, args):
        if name == "verify_caller":
            session.account = {"customer_id": "cus_verified"}
        else:
            assert session.account["customer_id"] == "cus_verified"
        seen.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    call = _call(monkeypatch, [
        _session_created(),
        {"type": "tool.call", "call_id": "c1", "name": "verify_caller",
         "arguments": {"account_number": "4417-88231"}},
        {"type": "tool.call", "call_id": "c2", "name": "get_account", "arguments": {}},
    ])
    asyncio.run(call._pump_fluxions())
    assert [name for name, _ in seen] == ["verify_caller", "get_account"]
    sent = [json.loads(c.args[0]) for c in call.fx_ws.send.await_args_list]
    assert [s["call_id"] for s in sent] == ["c1", "c2"]
    assert all(s["type"] == "tool.result" and json.loads(s["output"]) == {"ok": True} for s in sent)


def test_audio_flush_drops_unplayed_audio(monkeypatch):
    call = _call(monkeypatch, [
        _session_created(),
        b"\x01" * 3840,
        b"\x01" * 3840,
        {"type": "audio.flush"},
        b"\x02" * 3840,
    ])
    asyncio.run(call._pump_fluxions())
    assert call._flushes == 1
    assert [call._playback.get_nowait()] == [b"\x02" * 3840]
    assert call._playback.empty()


def test_wrong_session_rates_fail_the_call(monkeypatch):
    call = _call(monkeypatch, [_session_created(mic=24000)])
    with pytest.raises(RuntimeError, match="rates"):
        asyncio.run(call._pump_fluxions())


def test_session_update_pins_greeting_and_tools(monkeypatch):
    monkeypatch.setattr(agent, "_token", None, raising=False)
    update = agent._session_update()
    assert "token" not in update
    monkeypatch.setattr(agent, "_token", "k")
    assert agent._session_update()["token"] == "k"
    assert update["greet"] is False
    assert update["echo_guard"] is False
    assert update["allow_hangup"] is False
    assert (update["idle_check_s"], update["idle_hangup_s"]) == (3600, 3600)
    assert update["tools"] is agent.TOOLS
    assert update["soul"].startswith(agent.AGENT_PROMPT)


def test_greeting_is_queued_first_and_spoken_through_the_pacer(monkeypatch):
    monkeypatch.setattr(agent, "_greeting_pcm", b"\x07" * 48000, raising=False)
    monkeypatch.setattr(agent, "PLAYBACK_LEAD_S", 10.0)
    actor = SimpleNamespace(client=None, send_bytes=AsyncMock())
    call = agent._Call(actor, _Fluxions([]))

    async def run():
        pacer = asyncio.create_task(call._pump_playback())
        chunk = int(agent.PLAYBACK_CHUNK_S * agent.ACTOR_RATE_HZ) * 2
        for i in range(0, len(agent._greeting_pcm), chunk):
            call._playback.put_nowait(agent._greeting_pcm[i:i + chunk])
        while not call._playback.empty():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)
        pacer.cancel()

    asyncio.run(run())
    assert b"".join(c.args[0] for c in actor.send_bytes.await_args_list) == b"\x07" * 48000
    assert call._sent_s == pytest.approx(1.0)
