"""The connect + per-session tool handshake, exercised against a fake websocket."""

from __future__ import annotations

import asyncio
import json

import pytest

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


def test_session_update_declares_every_tool_verbatim_at_the_top_level(monkeypatch):
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    atoms_ws, sample_rate, call_id = asyncio.run(agent._connect_once())

    assert atoms_ws is fake
    assert sample_rate == 24000
    assert call_id == "CALL-1"
    assert fake.sent == [{"type": "session.update", "tools": TOOLS}]


def test_tool_declaration_error_is_a_setup_failure(monkeypatch):
    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "error", "code": "invalid_tools", "message": "bad schema"},
    ])

    with pytest.raises(RuntimeError, match="invalid_tools"):
        asyncio.run(agent._connect_once())


def test_partial_ack_keeps_waiting_for_the_rest(monkeypatch):
    monkeypatch.setattr(agent, "TOOLS_RESEND_S", 0.05)
    monkeypatch.setattr(agent, "TOOLS_READY_TIMEOUT_S", 1.0)
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "session.updated", "tools": ALL_NAMES[:1]},  # missing the rest
        {"type": "session.updated", "tools": ALL_NAMES},
    ])

    atoms_ws, _sample_rate, _call_id = asyncio.run(agent._connect_once())

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

    atoms_ws, _sample_rate, _call_id = asyncio.run(scenario())

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


def test_connect_retries_once_on_a_failed_handshake(monkeypatch):
    attempts = []

    async def fake_connect_once():
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return "ws", 24000, "CALL-2"

    monkeypatch.setattr(agent, "_connect_once", fake_connect_once)

    result = asyncio.run(agent._connect_with_retry())

    assert result == ("ws", 24000, "CALL-2")
    assert len(attempts) == 2
