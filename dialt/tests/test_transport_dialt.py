"""Exercise the transport against a local Dialt wire-protocol peer.

No Dialt credential or vendor backend is used. The entry point is the same
one the /voice handler calls; the server sees real WebSocket messages.
"""

import asyncio
import json
import struct
import threading
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.server import serve

from rory_dialt import agent
from rory_tools import SCHEMAS
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

READY = {
    "type": "ready",
    "audio": {"output_sr": 24000, "output_encoding": "pcm16"},
    "background_speaker_suppression": {"enabled": False, "agent_control": True},
}


class Actor:
    def __init__(self, frames=()):
        self.incoming = asyncio.Queue()
        for frame in frames:
            self.incoming.put_nowait(frame)
        self.audio = []
        self.first_audio = asyncio.Event()

    async def receive_bytes(self):
        frame = await self.incoming.get()
        if frame is None:
            raise WebSocketDisconnect()
        return frame

    async def send_bytes(self, audio):
        self.audio.append(audio)
        self.first_audio.set()


async def send(ws, event):
    await ws.send(json.dumps(event))


async def receive_json(ws):
    while True:
        raw = await ws.recv()
        if isinstance(raw, str):
            return json.loads(raw)


async def run_peer(monkeypatch, handler, actor):
    completed = asyncio.get_running_loop().create_future()

    async def checked(ws):
        try:
            await handler(ws)
        except Exception as exc:
            completed.set_exception(exc)
        else:
            completed.set_result(None)

    async with serve(checked, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(agent, "DIALT_URL", f"ws://127.0.0.1:{port}")
        monkeypatch.setenv("DIALT_API_KEY", "fixture-not-a-real-key")
        async with asyncio.timeout(5):
            await agent.run_voice_ws_bot(actor)
            await completed


@pytest.mark.parametrize("brain", ["fast", "smart"])
def test_start_audio_conversion_greeting_and_normal_drain(monkeypatch, brain):
    async def scenario():
        actor = Actor([struct.pack("<3h", 0, 1000, 2000), struct.pack("<3h", 3000, 4000, 5000)])
        output = b"\x21\x03" * 1200

        async def peer(ws):
            start = await receive_json(ws)
            mode = start["mode"]
            assert mode["brain"] == brain
            assert mode["instructions"] == f"{load_agent_prompt()}\n\n{today_context()}"
            assert mode["greeting"] == GREETING
            assert mode["background_speaker_suppression"] is False
            assert mode["agent_background_speaker_suppression"] is True
            assert mode["web_search"] is mode["end_call"] is False
            assert mode["tools"] == [
                {"name": s.name, "description": s.description, "parameters": s.parameters()}
                for s in SCHEMAS
            ]
            assert start["audio"] == {"sr": 16000, "output_sr": 24000, "output_encoding": "pcm16"}
            assert "api_key" not in start
            await send(ws, READY)
            uplink = await ws.recv() + await ws.recv()
            assert struct.unpack("<4h", uplink) == (0, 1500, 3000, 4500)
            await send(ws, {"type": "turn", "turn_id": "greeting"})
            await ws.send(output)
            # Closing immediately must not drop queued greeting audio.

        monkeypatch.setattr(agent, "DIALT_BRAIN", brain)
        await run_peer(monkeypatch, peer, actor)
        assert b"".join(actor.audio) == output
        assert all(len(frame) <= 960 for frame in actor.audio)

    asyncio.run(scenario())


@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("identified", [True, False, "turn_only"])
def test_interrupt_discards_or_drains_and_filters_late_audio(monkeypatch, clear, identified):
    async def scenario():
        actor = Actor()
        old = b"\x01\x00" * 4800  # 200 ms
        new = b"\x02\x00" * 480
        report = None

        async def peer(ws):
            nonlocal report
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "turn", "turn_id": "old", "response_id": "r1"})
            await ws.send(old)
            await actor.first_audio.wait()
            interruption = {"type": "interrupted", "clear": clear, "barge_seq": 7}
            if identified:
                interruption.update(turn_id="old")
                if identified != "turn_only":
                    interruption["response_id"] = "r1"
            await send(ws, interruption)
            await ws.send(b"\x03\x00" * 480)  # late old-response audio
            report = await receive_json(ws)
            await send(ws, {"type": "turn", "turn_id": "new", "response_id": "r2"})
            await ws.send(new)

        await run_peer(monkeypatch, peer, actor)
        assert report["barge_seq"] == 7
        assert report["event"] == "playback_stopped"
        heard = b"".join(actor.audio)
        assert heard.endswith(new)
        assert b"\x03\x00" not in heard
        old_bytes = len(heard) - len(new)
        if clear:
            assert 0 < old_bytes < len(old)
            assert report["remaining_ms"] == 0
            assert old_bytes + report["discarded_ms"] * 48 == len(old)
        else:
            assert old_bytes == len(old)
            assert report["discarded_ms"] == 0
            assert report["remaining_ms"] > 0

    asyncio.run(scenario())


@pytest.mark.parametrize("ready", [
    {**READY, "audio": {"output_sr": 16000, "output_encoding": "pcm16"}},
    {**READY, "background_speaker_suppression": {"enabled": True, "agent_control": True}},
    {"type": "error", "code": "permission_denied"},
])
def test_startup_rejection_or_mismatched_state_fails_closed(monkeypatch, ready):
    async def scenario():
        actor = Actor()

        async def peer(ws):
            await receive_json(ws)
            await send(ws, ready)

        with pytest.raises(RuntimeError, match="Dialt"):
            await run_peer(monkeypatch, peer, actor)
        assert not actor.audio

    asyncio.run(scenario())


def test_tools_are_sequential_without_blocking_audio_and_report_failures(monkeypatch):
    async def scenario():
        actor = Actor()
        release = threading.Event()
        states = []

        def dispatch(state, name, args):
            states.append(state)
            if name == "first":
                assert release.wait(3)
                state.account = {"fixture": True}
                return {"verified": True}
            assert state.account == {"fixture": True}
            return {"paid": False, "note": "Payment did not complete"}

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "tool_call", "id": "1", "name": "first", "args": {}})
            await send(ws, {"type": "tool_call", "id": "2", "name": "second", "args": {}})
            await send(ws, {"type": "turn", "turn_id": "bridge"})
            await ws.send(b"\x01\x00" * 480)
            await actor.first_audio.wait()
            release.set()
            first, second = await receive_json(ws), await receive_json(ws)
            assert first["id"] == "1" and first["outcome"] == "succeeded" and first["verified"]
            assert second["id"] == "2" and second["outcome"] == "failed" and not second["verified"]
            assert second["content"]["paid"] is False

        monkeypatch.setattr(agent, "dispatch", dispatch)
        try:
            await run_peer(monkeypatch, peer, actor)
            first_state = states[0]
            # The next connection must not inherit verified account state.
            actor = Actor()
            await run_peer(monkeypatch, peer, actor)
        finally:
            release.set()
        assert len(states) == 4 and states[0] is states[1]
        assert states[2] is states[3] and states[2] is not first_state

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["interrupted", "canceled"])
@pytest.mark.parametrize("mixed_ids", [False, True, "unknown"])
def test_old_response_stop_preserves_newer_queued_output(monkeypatch, kind, mixed_ids):
    async def scenario():
        actor = Actor()
        new = b"\x02\x00" * 960

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "turn", "turn_id": "old", "response_id": "r1"})
            await ws.send(b"\x01\x00" * 4800)
            await actor.first_audio.wait()
            await send(ws, {"type": "turn", "turn_id": "new", "response_id": "r2"})
            await ws.send(new)
            stop = {"type": kind, "turn_id": "old", "clear": True}
            if mixed_ids == "unknown":
                stop["turn_id"] = "never-seen"
            if not mixed_ids:
                stop["response_id"] = "r1"
            await send(ws, stop)
            if kind == "interrupted":
                report = await receive_json(ws)
                assert report["remaining_ms"] == 0
                assert "barge_seq" not in report

        await run_peer(monkeypatch, peer, actor)
        heard = b"".join(actor.audio)
        assert heard.endswith(new)
        if mixed_ids == "unknown":
            assert heard == b"\x01\x00" * 4800 + new
        else:
            assert len(heard) < 11520, "Old queued audio was not discarded"

    asyncio.run(scenario())


def test_actor_disconnect_closes_dialt(monkeypatch):
    async def scenario():
        actor = Actor()

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "turn", "turn_id": "greeting"})
            await ws.send(b"\x00\x01" * 480)
            await actor.first_audio.wait()
            actor.incoming.put_nowait(None)
            await ws.wait_closed()

        await run_peer(monkeypatch, peer, actor)

    asyncio.run(scenario())


def test_legacy_canceled_event_without_identifiers(monkeypatch):
    async def scenario():
        actor = Actor()
        new = b"\x02\x00" * 480

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "turn", "turn_id": "old"})
            await ws.send(b"\x01\x00" * 4800)
            await actor.first_audio.wait()
            # Observed on the live endpoint despite the published schema's
            # required turn_id: this refers to the current output.
            await send(ws, {"type": "canceled"})
            await ws.send(b"\x03\x00" * 480)
            await send(ws, {"type": "turn", "turn_id": "new"})
            await ws.send(new)

        await run_peer(monkeypatch, peer, actor)
        heard = b"".join(actor.audio)
        assert heard.endswith(new)
        assert len(heard) < 10560
        assert b"\x03\x00" not in heard

    asyncio.run(scenario())


def test_cancelled_queued_tool_does_not_execute(monkeypatch):
    async def scenario():
        actor = Actor()
        release = threading.Event()
        executed = []

        def dispatch(state, name, args):
            executed.append(name)
            if name == "first":
                assert release.wait(3)
            return {"verified": True}

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            for call_id in ("first", "cancelled", "last"):
                await send(ws, {"type": "tool_call", "id": call_id, "name": call_id, "args": {}})
            await send(ws, {"type": "tool_cancel", "id": "cancelled"})
            # Receiving the subsequent audio proves the cancellation reached
            # the bridge before the blocking first tool is released.
            await send(ws, {"type": "turn", "turn_id": "bridge"})
            await ws.send(b"\x01\x00" * 480)
            await actor.first_audio.wait()
            release.set()
            assert (await receive_json(ws))["id"] == "first"
            assert (await receive_json(ws))["id"] == "last"

        monkeypatch.setattr(agent, "dispatch", dispatch)
        try:
            await run_peer(monkeypatch, peer, actor)
        finally:
            release.set()
        assert executed == ["first", "last"]

    asyncio.run(scenario())


@pytest.mark.parametrize("close_code", [1000, 1011])
@pytest.mark.parametrize("inflight_tool", [False, True])
def test_clean_provider_close_drains_with_continuous_microphone(monkeypatch, close_code, inflight_tool):
    async def scenario():
        class StreamingActor(Actor):
            async def receive_bytes(self):
                await asyncio.sleep(.005)
                return b"\0" * 240

        actor = StreamingActor()
        output = b"\x01\x00" * 24000
        started = threading.Event()
        release = threading.Event()

        def dispatch(*args):
            started.set()
            assert release.wait(4)
            return {"ok": True}

        monkeypatch.setattr(agent, "dispatch", dispatch)

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            if inflight_tool:
                await send(ws, {"type": "tool_call", "id": "pending", "name": "get_account", "args": {}})
                assert await asyncio.to_thread(started.wait, 2)
            await send(ws, {"type": "turn", "turn_id": "final"})
            await ws.send(output)
            await actor.first_audio.wait()
            await ws.close(code=close_code)
            await asyncio.sleep(.03)
            release.set()

        try:
            if close_code == 1000:
                await run_peer(monkeypatch, peer, actor)
                assert b"".join(actor.audio) == output
            else:
                from websockets.exceptions import ConnectionClosedError
                with pytest.raises(ConnectionClosedError):
                    await run_peer(monkeypatch, peer, actor)
        finally:
            release.set()

    asyncio.run(scenario())


def test_clean_close_during_playback_receipt_preserves_soft_stop_audio(monkeypatch):
    from websockets.asyncio.client import ClientConnection

    async def scenario():
        actor = Actor()
        output = b"\x01\x00" * 24000
        receipt_started = asyncio.Event()
        original_send = ClientConnection.send

        async def delayed_receipt(ws, message, *args, **kwargs):
            if isinstance(message, str) and json.loads(message).get("event") == "playback_stopped":
                receipt_started.set()
                await ws.wait_closed()
            await original_send(ws, message, *args, **kwargs)

        monkeypatch.setattr(ClientConnection, "send", delayed_receipt)

        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            await send(ws, {"type": "turn", "turn_id": "final"})
            await ws.send(output)
            await actor.first_audio.wait()
            await send(ws, {"type": "interrupted", "turn_id": "final", "clear": False})
            await receipt_started.wait()
            await ws.close(code=1000)

        await run_peer(monkeypatch, peer, actor)
        assert b"".join(actor.audio) == output

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["refresh", "response_lost", None])
def test_committed_write_errors_are_unknown_not_failed(monkeypatch, failure):
    from rory_tools.session import clients

    account = {"customer_id": "cus_fixture", "name": "Jane Smith",
               "mailing_address": {"postal_code": "60614"}, "email": "jane@example.invalid"}
    committed = []

    def write(_customer, updates):
        committed.append(updates)
        if failure == "response_lost":
            raise TimeoutError("fixture response lost after commit")
        return account

    def refresh(_customer):
        if failure == "refresh":
            raise TimeoutError("fixture refresh failed after commit")
        return account

    monkeypatch.setattr(clients, "_stripe", SimpleNamespace(
        find_customer_by_account_number=lambda _: account,
        list_payment_methods=lambda _: [], list_bills=lambda *_args, **_kwargs: [],
        set_customer_metadata=write, get_customer=refresh,
    ))
    monkeypatch.setattr(clients, "_fineract", SimpleNamespace(
        find_client_by_account_number=lambda _: {"client_id": 1},
    ))

    async def scenario():
        async def peer(ws):
            await receive_json(ws)
            await send(ws, READY)
            for call_id, postal, expected in (("wrong", "99999", "failed"), ("right", "60614", "succeeded")):
                await send(ws, {"type": "tool_call", "id": call_id, "name": "verify_caller",
                                "args": {"account_number": "4417-88231", "name_on_account": "Jane Smith", "second_factor": postal}})
                result = await receive_json(ws)
                assert result["outcome"] == expected
            await send(ws, {"type": "tool_call", "id": "write", "name": "set_paperless_billing", "args": {"enabled": True}})
            result = await receive_json(ws)
            assert committed == [{"paperless": True}]
            assert result["outcome"] == ("unknown" if failure else "succeeded")
            assert result["verified"] is (failure is None)

        await run_peer(monkeypatch, peer, Actor())

    asyncio.run(scenario())
