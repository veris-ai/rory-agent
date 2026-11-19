"""The connect + per-session tool handshake, exercised against a fake websocket."""

from __future__ import annotations

import json

import pytest

from rory_atoms import agent
from rory_atoms.tools import TOOLS


class FakeAtomsWS:
    """Stands in for the object ``await websockets.connect(...)`` resolves to."""

    def __init__(self, to_recv: list[dict]):
        self._to_recv = list(to_recv)
        self.sent: list[dict] = []

    async def recv(self):
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


def test_session_update_declares_every_tool_verbatim(monkeypatch):
    fake = _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "session.updated"},
    ])

    import asyncio

    atoms_ws, sample_rate, call_id = asyncio.run(agent._connect_once())

    assert atoms_ws is fake
    assert sample_rate == 24000
    assert call_id == "CALL-1"
    assert fake.sent == [{"type": "session.update", "session": {"tools": TOOLS}}]


def test_missing_session_updated_ack_is_a_setup_failure(monkeypatch):
    import asyncio

    _connect(monkeypatch, [
        {"type": "session.created", "sample_rate": 24000, "call_id": "CALL-1"},
        {"type": "error", "code": "invalid_event", "message": "bad tools"},
    ])

    with pytest.raises(RuntimeError, match="session.updated"):
        asyncio.run(agent._connect_once())


def test_connect_retries_once_on_a_failed_handshake(monkeypatch):
    import asyncio

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
