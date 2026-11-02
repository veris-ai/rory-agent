"""Exercise the session handshake, tool rounds, audio bridge, and call teardown."""

import asyncio
import base64
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.asyncio.server import serve

from rory_tools import CallSession
from rory_tools.prompt import GREETING

from rory_polyai_dialog_rsn_1 import agent, web


class FakeRSN:
    """Async-iterable stand-in for the Dialog-RSN-1 socket; records what is sent."""

    def __init__(self, events):
        self.events = iter(events)
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return json.dumps(next(self.events))
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, raw):
        self.sent.append(json.loads(raw))


def _done(status="completed"):
    return {"type": "response.done", "response": {"status": status}}


def _call(name="get_account", call_id="call_1", arguments="{}"):
    return {
        "type": "response.function_call_arguments.done",
        "name": name, "call_id": call_id, "arguments": arguments,
    }


class Drained(Exception):
    pass


class FiniteQueue(asyncio.Queue):
    """Ends the tool worker, which waits forever on a live call, once it is empty."""

    async def get(self):
        if self.empty():
            raise Drained
        return await super().get()


def _pump(events):
    rsn = FakeRSN(events)
    actor = SimpleNamespace(send_bytes=AsyncMock())
    calls = FiniteQueue()
    asyncio.run(agent._pump_rsn_to_actor(rsn, actor, calls))
    return rsn.sent, actor, calls


def _bridge(events, session=None):
    """The pump's queue, then the tool worker drains it; returns what reached the socket."""
    sent, _, calls = _pump(events)
    rsn = FakeRSN([])
    with pytest.raises(Drained):
        asyncio.run(agent._run_tools(rsn, session or CallSession(), calls))
    assert sent == []
    return rsn.sent


def test_session_update_carries_prompt_tools_audio_and_voice(monkeypatch):
    monkeypatch.setattr(agent, "today_context", lambda: "Frozen date for this call")
    update = agent._session_update()
    session = update["session"]
    assert update["type"] == "session.update"
    assert session["instructions"] == f"{agent.AGENT_PROMPT}\n\nFrozen date for this call"
    assert session["tools"] is agent.TOOLS
    assert session["output_modalities"] == ["audio"]
    assert session["audio"] == {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"language": "en"},
            "turn_detection": {"type": "server_vad", "create_response": True},
        },
        "output": {"voice": agent.RSN_VOICE},
    }


@pytest.mark.parametrize("missing", ["POLY_API_KEY", "STRIPE_API_KEY"])
def test_missing_credentials_stop_startup(monkeypatch, missing):
    monkeypatch.setenv("POLY_API_KEY", "test")
    monkeypatch.setenv("STRIPE_API_KEY", "test")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(web.app):
            pass


def test_health_with_credentials(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test")
    monkeypatch.setenv("STRIPE_API_KEY", "test")
    with TestClient(web.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def _serve_call(monkeypatch, provider, actor):
    """Run one call against a local websocket server standing in for PolyAI."""
    monkeypatch.setenv("POLY_API_KEY", "poly_test")

    async def exercise():
        async with asyncio.timeout(5):
            async with serve(provider, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                monkeypatch.setattr(agent, "REALTIME_URL", f"ws://127.0.0.1:{port}")
                await agent.run_voice_ws_bot(actor)

    asyncio.run(exercise())


def test_call_bridges_a_real_local_websocket(monkeypatch):
    """The real handshake, greeting, both pumps and one tool round, without a vendor key.

    The tool returns only once its hold line has reached the caller, and its
    result goes back before the response is done: it runs while the line plays.
    """
    caller_audio = b"\x01\x02" * 480
    hold_audio = b"\x05\x06" * 480
    reply_audio = b"\x03\x04" * 480
    received = []
    hold_line_heard = threading.Event()

    def dispatch(state, name, args):
        assert hold_line_heard.wait(5)
        return {"verified": False}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    disconnected = asyncio.Event()
    caller_sent = False

    async def receive_bytes():
        nonlocal caller_sent
        if not caller_sent:
            caller_sent = True
            return caller_audio
        await disconnected.wait()
        raise WebSocketDisconnect(1000)

    async def send_bytes(audio):
        received.append(audio)
        hold_line_heard.set()
        if audio == reply_audio:
            disconnected.set()

    async def provider(socket):
        assert socket.request.headers["X-API-KEY"] == "poly_test"
        await socket.send(json.dumps({"type": "session.created"}))
        update = json.loads(await socket.recv())
        assert update["type"] == "session.update"
        assert update["session"]["tools"] == agent.TOOLS
        await socket.send(json.dumps({"type": "session.updated"}))
        greeting = json.loads(await socket.recv())
        assert greeting["type"] == "response.create"
        assert GREETING in greeting["response"]["instructions"]
        audio = json.loads(await socket.recv())
        assert audio["type"] == "input_audio_buffer.append"
        assert base64.b64decode(audio["audio"]) == caller_audio
        await socket.send(json.dumps(_call("verify_caller", "call_v", '{"account_number":"00001"}')))
        await socket.send(json.dumps({
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(hold_audio).decode(),
        }))
        result = json.loads(await socket.recv())
        assert result["item"] == {
            "type": "function_call_output", "call_id": "call_v",
            "output": json.dumps({"verified": False}),
        }
        await socket.send(json.dumps(_done()))
        assert json.loads(await socket.recv()) == {"type": "response.create"}
        await socket.send(json.dumps({
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(reply_audio).decode(),
        }))
        await socket.wait_closed()

    _serve_call(monkeypatch, provider, SimpleNamespace(receive_bytes=receive_bytes, send_bytes=send_bytes))
    assert received == [hold_audio, reply_audio]


def test_refused_session_update_ends_the_call_before_the_greeting(monkeypatch):
    sent = []

    async def provider(socket):
        await socket.send(json.dumps({"type": "session.created"}))
        sent.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({
            "type": "error",
            "error": {"code": "unsupported_parameter", "message": "Refused"},
        }))
        async for raw in socket:
            sent.append(json.loads(raw))

    with pytest.raises(RuntimeError, match="unsupported_parameter"):
        _serve_call(monkeypatch, provider, SimpleNamespace())
    assert [message["type"] for message in sent] == ["session.update"]


def test_actor_audio_is_forwarded_losslessly_and_disconnect_ends_the_pump():
    audio = b"\x01\x02" * 480
    actor = SimpleNamespace(receive_bytes=AsyncMock(side_effect=[audio, WebSocketDisconnect(1000)]))
    rsn = FakeRSN([])
    asyncio.run(agent._pump_actor_to_rsn(actor, rsn))
    assert rsn.sent == [{"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()}]


def test_reply_audio_streams_through_and_barge_in_sends_nothing():
    """The server paces and trims its own spoken reply, so the bridge never truncates."""
    audio = bytes(4800)
    delta = {"type": "response.output_audio.delta", "delta": base64.b64encode(audio).decode()}
    sent, actor, _ = _pump([
        {"type": "response.created"}, delta, delta,
        {"type": "input_audio_buffer.speech_started"},
        _done("cancelled"),
    ])
    assert [call.args[0] for call in actor.send_bytes.await_args_list] == [audio, audio]
    assert sent == []


def test_tool_rounds_run_in_order_on_the_call_session_with_one_continuation(monkeypatch):
    session = CallSession()
    seen = []

    def dispatch(state, name, args):
        assert state is session
        seen.append((name, args))
        return {"ok": name}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    sent = _bridge([
        _done(),
        _call("verify_caller", "verify", '{"account_number":"00001"}'), _done(),
        _call("get_account", "account"), _done(),
    ], session)
    assert seen == [("verify_caller", {"account_number": "00001"}), ("get_account", {})]
    assert [message["type"] for message in sent] == [
        "conversation.item.create", "response.create",
    ] * 2
    assert [json.loads(sent[i]["item"]["output"]) for i in (0, 2)] == [
        {"ok": "verify_caller"}, {"ok": "get_account"},
    ]


def test_a_call_in_a_cancelled_response_runs_and_answers_without_a_continuation(monkeypatch):
    """The call ran before the barge-in, so its result still goes back; the barge-in has its own response."""
    seen = []
    monkeypatch.setattr(agent, "dispatch", lambda state, name, args: seen.append(name) or {"ok": True})
    sent = _bridge([
        _call("verify_caller", "cancelled", '{"account_number":"00001"}'), _done("cancelled"),
        _call("get_account", "next"), _done(),
    ])
    assert seen == ["verify_caller", "get_account"]
    assert [(message["type"], message.get("item", {}).get("call_id")) for message in sent] == [
        ("conversation.item.create", "cancelled"),
        ("conversation.item.create", "next"),
        ("response.create", None),
    ]


@pytest.mark.parametrize("event", [
    {"type": "error", "error": {"code": "generation_failed", "message": "Backend failure"}},
    _done("failed"),
])
def test_provider_failures_end_the_call(event):
    with pytest.raises(RuntimeError, match="Dialog-RSN-1"):
        _pump([event])


@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_reports_normal_and_failed_calls(monkeypatch, failed, code):
    actor = SimpleNamespace(client_state=WebSocketState.CONNECTED, accept=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(web, "run_voice_ws_bot", AsyncMock(side_effect=RuntimeError("failed") if failed else None))
    if failed:
        with pytest.raises(RuntimeError, match="failed"):
            asyncio.run(web.voice(actor))
    else:
        asyncio.run(web.voice(actor))
    assert actor.close.await_args.kwargs["code"] == code
