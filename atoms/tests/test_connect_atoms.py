"""The connect + per-session tool handshake, exercised against a fake websocket."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
import websockets

from rory_atoms import agent
from rory_atoms.tools import TOOLS

ALL_NAMES = [t["name"] for t in TOOLS]


class FakeAtomsWS:
    """Stands in for the object ``await websockets.connect(...)`` resolves to.

    ``recv()`` blocks forever once the scripted events run out, so a caller
    waiting on it via ``asyncio.wait_for`` times out the way it would against
    a real socket with nothing more to say yet — rather than raising.
    """

    def __init__(self, to_recv: list[dict]):
        self._to_recv = list(to_recv)
        self.sent: list[dict] = []

    async def recv(self):
        if not self._to_recv:
            await asyncio.Future()
        return json.dumps(self._to_recv.pop(0))

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        pass


def _connect(monkeypatch, to_recv):
    fake = FakeAtomsWS(to_recv)

    async def fake_connect(*_args, **_kwargs):
        return fake

    monkeypatch.setattr(agent.websockets, "connect", fake_connect)
    monkeypatch.setenv("ATOMS_API_KEY", "sk_fixture")
    return fake


def _audio_event(pcm: bytes) -> dict:
    return {"type": "output_audio.delta", "audio": base64.b64encode(pcm).decode()}


def test_session_update_declares_every_tool_verbatim_at_the_top_level(monkeypatch):
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    atoms_ws, sample_rate, call_id, early_audio = asyncio.run(agent._connect_once())

    assert atoms_ws is fake
    assert sample_rate == 24000
    assert call_id == "CALL-1"
    assert early_audio == []
    assert fake.sent == [{"type": "session.update", "tools": TOOLS}]


def test_greeting_audio_during_setup_is_buffered_not_dropped(monkeypatch):
    pcm = b"\x01\x02\x03\x04"
    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        _audio_event(pcm),
        _audio_event(pcm),
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    _atoms_ws, _sr, _cid, early_audio = asyncio.run(agent._connect_once())

    assert early_audio == [pcm, pcm]


def test_resend_is_timer_based_not_per_event(monkeypatch):
    """A burst of non-ack events (routine greeting audio, say) arriving while
    the tool declaration is unacknowledged must not each trigger a resend —
    only real elapsed time should. Before this fix, every such event
    resent the declaration immediately, which could exhaust the whole send
    budget before a real (if slow) acknowledgment ever arrived."""
    monkeypatch.setattr(agent, "TOOLS_RESEND_S", 10.0)
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 20.0)
    pcm = b"\x00\x01"
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        *[_audio_event(pcm) for _ in range(5)],
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    _atoms_ws, _sr, _cid, early_audio = asyncio.run(agent._connect_once())

    assert len(fake.sent) == 1, "a burst of non-ack events triggered extra resends"
    assert early_audio == [pcm] * 5


def test_tool_declaration_error_is_a_setup_failure(monkeypatch):
    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "error", "code": "invalid_tools", "message": "bad schema"},
    ])

    with pytest.raises(agent._SetupError, match="invalid_tools") as exc_info:
        asyncio.run(agent._connect_once())
    assert exc_info.value.call_id == "CALL-1"
    assert exc_info.value.code == "invalid_tools"
    assert exc_info.value.retryable is True


def test_tools_stored_on_the_agent_is_not_retryable(monkeypatch):
    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "error", "code": "client_tools_managed_by_config", "message": "stored"},
    ])

    with pytest.raises(agent._SetupError) as exc_info:
        asyncio.run(agent._connect_once())
    assert exc_info.value.retryable is False


def test_partial_ack_keeps_waiting_for_the_rest(monkeypatch):
    monkeypatch.setattr(agent, "TOOLS_RESEND_S", 0.05)
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 1.0)
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "session.updated", "tools": ALL_NAMES[:1]},  # missing the rest
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    atoms_ws, _sample_rate, _call_id, _early_audio = asyncio.run(agent._connect_once())

    assert atoms_ws is fake


def test_session_update_is_resent_while_unacknowledged(monkeypatch):
    monkeypatch.setattr(agent, "TOOLS_RESEND_S", 0.05)
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 1.0)
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
    ])

    async def deliver_late_ack():
        await asyncio.sleep(0.15)  # past at least one resend interval
        fake._to_recv.append({"type": "session.updated", "tools": ALL_NAMES})

    async def scenario():
        delivery = asyncio.create_task(deliver_late_ack())
        try:
            return await agent._connect_once()
        finally:
            await delivery

    atoms_ws, _sample_rate, _call_id, _early_audio = asyncio.run(scenario())

    assert atoms_ws is fake
    assert len(fake.sent) >= 2, "no resend happened while the ack was outstanding"
    assert all(s == {"type": "session.update", "tools": TOOLS} for s in fake.sent)


def test_never_acknowledged_times_out_as_a_setup_failure(monkeypatch):
    monkeypatch.setattr(agent, "TOOLS_RESEND_S", 0.02)
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 0.08)
    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
    ])

    with pytest.raises(RuntimeError, match="session.updated"):
        asyncio.run(agent._connect_once())


def test_missing_session_created_is_a_setup_failure_within_the_deadline(monkeypatch):
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 0.05)
    _connect(monkeypatch, [])  # recv() never resolves

    with pytest.raises(agent._SetupError, match="session.created"):
        asyncio.run(agent._connect_once())


def test_connect_retries_once_on_a_retryable_failure(monkeypatch):
    monkeypatch.setattr(agent, "_retry_delay", lambda: 0)
    attempts = []

    async def fake_connect_once():
        attempts.append(None)
        if len(attempts) == 1:
            raise agent._SetupError("transient failure")
        return "ws", 24000, "CALL-2", []

    monkeypatch.setattr(agent, "_connect_once", fake_connect_once)

    result = asyncio.run(agent._connect_with_retry())

    assert result == ("ws", 24000, "CALL-2", [])
    assert len(attempts) == 2


def test_connect_does_not_retry_on_a_4xx_upgrade_failure(monkeypatch):
    attempts = []

    class _FakeResponse:
        status_code = 400

    async def fake_websockets_connect(*_args, **_kwargs):
        attempts.append(None)
        raise websockets.InvalidStatus(_FakeResponse())

    monkeypatch.setattr(agent.websockets, "connect", fake_websockets_connect)
    monkeypatch.setenv("ATOMS_API_KEY", "sk_fixture")

    with pytest.raises(agent._SetupError):
        asyncio.run(agent._connect_with_retry())

    assert len(attempts) == 1, "a 4xx upgrade failure should not be retried"


def test_connect_does_not_retry_when_tools_are_managed_by_config(monkeypatch):
    attempts = []

    async def fake_connect_once():
        attempts.append(None)
        raise agent._SetupError("stored", code="client_tools_managed_by_config", retryable=False)

    monkeypatch.setattr(agent, "_connect_once", fake_connect_once)

    with pytest.raises(agent._SetupError):
        asyncio.run(agent._connect_with_retry())

    assert len(attempts) == 1


def test_retry_delay_is_within_the_jitter_range():
    for _ in range(50):
        delay = agent._retry_delay()
        assert agent.RETRY_JITTER_MIN_S <= delay < agent.RETRY_JITTER_MAX_S


def test_every_tool_carries_the_configured_timeout():
    assert all(t["timeout_ms"] == 30_000 for t in TOOLS)
