"""A loopback test through the real ASGI app: what close code the actor
side actually sees, for each way the Atoms side can end a call.

Exercises the real `/voice` route end to end — actor websocket client,
`rory_atoms.web.app`, and `rory_atoms.agent`'s connect/receive logic — against
a small local fake standing in for Atoms, rather than asserting on internal
functions directly. This is the only way to prove what web.py actually does
with the close code, since websockets 15 does not raise an exception for
every close code the fake can send (see test_pumps_atoms.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time

import pytest
import uvicorn
import websockets

from rory_atoms.tools import TOOLS

ALL_NAMES = [t["name"] for t in TOOLS]


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextlib.contextmanager
def _server(app):
    """A finite ASGI server with an OS-assigned port and guaranteed shutdown."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="websockets"))
    thread = threading.Thread(target=instance.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    try:
        while not instance.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Local ASGI server did not start")
            time.sleep(0.02)
        yield port
    finally:
        instance.should_exit = True
        thread.join(15)
        sock.close()


def _fake_atoms_handler(*, close_code: int | None, send_session_closed: bool, abrupt: bool = False):
    """One fake Atoms session: the real handshake (session.created,
    session.updated), then either a session.closed(ended) followed by a
    clean close, a clean close with no session.closed at all, or (when
    ``abrupt``) the transport dropping with no close frame whatsoever —
    standing in for an abnormal end either way. (1005 is a client-only,
    locally-inferred code for exactly that last case — a close frame with
    no status at all — and cannot be sent over the wire by a peer at all,
    the library rejects it outright; a dropped transport is the only way
    to reach it in practice.)"""

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created", "sample_rate": 24000, "call_id": "CALL-TEST"}))
        raw = await ws.recv()
        assert json.loads(raw)["type"] == "session.update"
        await ws.send(json.dumps({"type": "session.updated", "tools": ALL_NAMES}))
        if send_session_closed:
            await ws.send(json.dumps({"type": "session.closed", "reason": "ended"}))
        if abrupt:
            ws.transport.close()
        else:
            await ws.close(code=close_code)

    return handler


async def _drive_one_call(fake_atoms_port: int, app_port: int) -> int | None:
    """Connects an actor client to the real /voice route and returns the
    close code the actor side actually received."""
    async with websockets.connect(f"ws://127.0.0.1:{app_port}/voice") as actor:
        with contextlib.suppress(websockets.ConnectionClosed):
            async for _ in actor:
                pass
        return actor.close_code


def _run_scenario(monkeypatch, *, close_code: int | None = None, send_session_closed: bool,
                   abrupt: bool = False) -> int | None:
    from rory_atoms import agent
    from rory_atoms.web import app

    monkeypatch.setenv("ATOMS_API_KEY", "sk_fixture")
    monkeypatch.setenv("ATOMS_AGENT_ID", "agent_fixture")
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_fixture")

    async def scenario():
        fake = await websockets.serve(
            _fake_atoms_handler(close_code=close_code, send_session_closed=send_session_closed, abrupt=abrupt),
            "127.0.0.1", 0,
        )
        fake_port = fake.sockets[0].getsockname()[1]
        try:
            monkeypatch.setattr(agent, "ATOMS_BASE_URL", f"ws://127.0.0.1:{fake_port}/agent/connect")
            with _server(app) as app_port:
                return await _drive_one_call(fake_port, app_port)
        finally:
            fake.close()
            await fake.wait_closed()

    return asyncio.run(scenario())


@pytest.mark.parametrize("close_code", [1000, 1001])
def test_atoms_closing_cleanly_without_session_closed_ends_the_actor_abnormally(monkeypatch, close_code):
    """Atoms closing with no session.closed(ended) ever sent — whatever the
    numeric code — must reach the actor as 1011 (this bridge's own
    failure), not as a clean end of call. websockets 15 does not even raise
    an exception internally for a clean close like this, so only an
    end-to-end loopback actually proves what the actor receives."""
    close_received = _run_scenario(monkeypatch, close_code=close_code, send_session_closed=False)
    assert close_received == 1011


def test_atoms_disconnecting_abruptly_ends_the_actor_abnormally(monkeypatch):
    """No close frame at all (what a 1005/1006-style drop looks like) is
    just as much this bridge's own failure as a clean-looking close with
    no session.closed — same 1011 either way."""
    close_received = _run_scenario(monkeypatch, send_session_closed=False, abrupt=True)
    assert close_received == 1011


def test_atoms_closing_after_session_closed_ended_ends_the_actor_cleanly(monkeypatch):
    close_received = _run_scenario(monkeypatch, close_code=1000, send_session_closed=True)
    assert close_received == 1000
