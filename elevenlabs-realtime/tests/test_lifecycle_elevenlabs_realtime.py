"""Exercise the API boundary, tool completion, audio bridge, and call teardown."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.asyncio.server import serve

from rory_tools import CallSession
from rory_tools.prompt import GREETING

from rory_elevenlabs_realtime import agent, web


class FakeRealtime:
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

    async def recv(self):
        return json.dumps({"type": "session.created"})


def _response(output=(), status="completed"):
    return {"type": "response.done", "response": {"status": status, "output": list(output)}}


def _tool(name="get_account", call_id="call_1", arguments="{}"):
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


@pytest.mark.parametrize("model", ["eleven_realtime_v1_mini", "eleven_realtime_v1_max"])
def test_mint_scopes_the_model_prompt_tools_and_audio_to_the_call(monkeypatch, model):
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"value": "ek_test"})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(agent.httpx, "AsyncClient", lambda: client_class(transport=httpx.MockTransport(handle)))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    monkeypatch.setattr(agent, "REALTIME_MODEL", model)
    monkeypatch.setattr(agent, "today_context", lambda: "Frozen date for this call")
    assert asyncio.run(agent._create_client_secret()) == "ek_test"

    request, = captured
    assert request.method == "POST"
    assert str(request.url) == agent.CLIENT_SECRETS_URL
    assert request.headers["xi-api-key"] == "test_key"
    body = json.loads(request.content)
    config = body["session"]
    assert config["model"] == model
    assert config["instructions"] == f"{agent.AGENT_PROMPT}\n\nFrozen date for this call"
    assert config["tools"] == agent.TOOLS
    assert config["parallel_tool_calls"] is False
    assert config["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert config["audio"]["output"] == {"format": {"type": "audio/pcm"}, "voice": agent.REALTIME_VOICE}
    assert config["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad", "eagerness": "medium",
        "create_response": True, "interrupt_response": True,
    }
    assert body["expires_after"] == {"anchor": "created_at", "seconds": 600}


def test_mint_failure_propagates(monkeypatch):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(403, json={"detail": "Unavailable"}))
    monkeypatch.setattr(agent.httpx, "AsyncClient", lambda: client_class(transport=transport))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_key")
    with pytest.raises(httpx.HTTPStatusError, match="403") as error:
        asyncio.run(agent._create_client_secret())
    assert "Unavailable" in "\n".join(error.value.__notes__)


@pytest.mark.parametrize("missing", ["ELEVENLABS_API_KEY", "STRIPE_API_KEY"])
def test_missing_credentials_stop_startup(monkeypatch, missing):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("STRIPE_API_KEY", "test")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(web.app):
            pass


def test_health_with_credentials(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.setenv("STRIPE_API_KEY", "test")
    with TestClient(web.app) as client:
        assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("failed", [False, True])
def test_greeting_authentication_and_pump_cleanup(monkeypatch, failed):
    realtime = FakeRealtime([])
    connected = []
    cancelled = []
    sessions = []
    mint = AsyncMock(return_value="ek_per_call")

    @asynccontextmanager
    async def connect(url, **kwargs):
        connected.append((url, kwargs))
        yield realtime

    async def up(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def down(socket, actor, session):
        sessions.append(session)
        if failed:
            raise RuntimeError("broken audio pump")

    monkeypatch.setattr(agent, "_create_client_secret", mint)
    monkeypatch.setattr(agent.websockets, "connect", connect)
    monkeypatch.setattr(agent, "_pump_actor_to_realtime", up)
    monkeypatch.setattr(agent, "_pump_realtime_to_actor", down)
    for _ in range(2):
        if failed:
            with pytest.raises(RuntimeError, match="broken audio pump"):
                asyncio.run(agent.run_voice_ws_bot(SimpleNamespace()))
        else:
            asyncio.run(agent.run_voice_ws_bot(SimpleNamespace()))

    assert cancelled == [True, True]
    assert sessions[0] is not sessions[1]
    assert mint.await_count == 2
    assert connected == [(agent.REALTIME_URL, {
        "subprotocols": ["realtime", "openai-insecure-api-key.ek_per_call"],
    })] * 2
    assert [message["type"] for message in realtime.sent] == [
        "conversation.item.create", "response.create",
    ] * 2
    greeting = realtime.sent[0]["item"]
    assert greeting["type"] == "message"
    assert greeting["role"] == "user"
    assert greeting["content"][0]["type"] == "input_text"
    assert GREETING in greeting["content"][0]["text"]


def test_session_creation_error_propagates_before_greeting(monkeypatch):
    realtime = FakeRealtime([])
    realtime.recv = AsyncMock(return_value=json.dumps({"type": "error", "error": {"message": "Denied"}}))

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield realtime

    monkeypatch.setattr(agent, "_create_client_secret", AsyncMock(return_value="ek_test"))
    monkeypatch.setattr(agent.websockets, "connect", connect)
    with pytest.raises(RuntimeError, match="Denied"):
        asyncio.run(agent.run_voice_ws_bot(SimpleNamespace()))
    assert realtime.sent == []


def test_actor_audio_is_forwarded_losslessly_and_disconnect_ends_the_pump():
    audio = b"\x01\x02" * 480
    actor = SimpleNamespace(receive_bytes=AsyncMock(side_effect=[audio, WebSocketDisconnect(1000)]))
    realtime = FakeRealtime([])
    asyncio.run(agent._pump_actor_to_realtime(actor, realtime))
    assert realtime.sent == [{"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode()}]


def test_call_bridges_a_real_local_websocket(monkeypatch):
    """Exercise the actual handshake and both pumps together, without a vendor key."""
    caller_audio = b"\x01\x02" * 480
    reply_audio = b"\x03\x04" * 480
    received = []
    mint = AsyncMock(return_value="ek_loopback")
    monkeypatch.setattr(agent, "_create_client_secret", mint)
    monkeypatch.setattr(agent, "dispatch", lambda state, name, args: {"verified": False})

    async def exercise():
        disconnected = asyncio.Event()
        caller_sent = False
        provider_finished = asyncio.Event()

        async def receive_bytes():
            nonlocal caller_sent
            if not caller_sent:
                caller_sent = True
                return caller_audio
            await disconnected.wait()
            raise WebSocketDisconnect(1000)

        async def send_bytes(audio):
            received.append(audio)
            disconnected.set()

        async def provider(socket):
            assert "openai-insecure-api-key.ek_loopback" in socket.request.headers["Sec-WebSocket-Protocol"]
            await socket.send(json.dumps({"type": "session.created"}))
            greeting = json.loads(await socket.recv())
            assert greeting["type"] == "conversation.item.create"
            assert greeting["item"]["role"] == "user"
            assert GREETING in greeting["item"]["content"][0]["text"]
            assert json.loads(await socket.recv()) == {"type": "response.create"}
            audio = json.loads(await socket.recv())
            assert audio["type"] == "input_audio_buffer.append"
            assert base64.b64decode(audio["audio"]) == caller_audio
            await socket.send(json.dumps(_response([_tool()])))
            result = json.loads(await socket.recv())
            assert result["item"]["type"] == "function_call_output"
            assert json.loads(result["item"]["output"]) == {"verified": False}
            assert json.loads(await socket.recv()) == {"type": "response.create"}
            await socket.send(json.dumps({"type": "response.created"}))
            await socket.send(json.dumps({
                "type": "response.output_audio.delta", "item_id": "reply",
                "delta": base64.b64encode(reply_audio).decode(),
            }))
            await socket.wait_closed()
            provider_finished.set()

        async with asyncio.timeout(5):
            async with serve(provider, "127.0.0.1", 0, subprotocols=["realtime"]) as server:
                port = server.sockets[0].getsockname()[1]
                monkeypatch.setattr(agent, "REALTIME_URL", f"ws://127.0.0.1:{port}")
                actor = SimpleNamespace(receive_bytes=receive_bytes, send_bytes=send_bytes)
                await agent.run_voice_ws_bot(actor)
                await provider_finished.wait()

    asyncio.run(exercise())
    assert received == [reply_audio]
    mint.assert_awaited_once()


def test_streamed_audio_and_barge_in_target_the_current_item():
    audio = bytes(4800)  # 100 ms of PCM16 at 24 kHz.
    delta = {"type": "response.output_audio.delta", "item_id": "item_1", "delta": base64.b64encode(audio).decode()}
    realtime = FakeRealtime([
        {"type": "response.created"}, delta, delta,
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_started"},
        _response(),
        {"type": "input_audio_buffer.speech_started"},
    ])
    actor = SimpleNamespace(send_bytes=AsyncMock())
    asyncio.run(agent._pump_realtime_to_actor(realtime, actor, CallSession()))
    assert [call.args[0] for call in actor.send_bytes.await_args_list] == [audio, audio]
    assert realtime.sent == [{
        "type": "conversation.item.truncate", "item_id": "item_1",
        "content_index": 0, "audio_end_ms": 200,
    }]


def test_completed_tool_batch_executes_in_order_with_one_continuation(monkeypatch):
    session = CallSession()
    seen = []

    def dispatch(state, name, args):
        assert state is session
        if name == "verify_caller":
            state.account = {"customer_id": "verified"}
        else:
            assert state.customer_id == "verified"
        seen.append((name, args))
        return {"ok": name}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    realtime = FakeRealtime([_response([
        _tool("verify_caller", "verify", '{"account_number":"00001"}'),
        _tool("get_account", "account"),
    ])])
    asyncio.run(agent._pump_realtime_to_actor(realtime, SimpleNamespace(), session))
    assert seen == [("verify_caller", {"account_number": "00001"}), ("get_account", {})]
    assert [message["item"]["call_id"] for message in realtime.sent[:-1]] == ["verify", "account"]
    assert [json.loads(message["item"]["output"]) for message in realtime.sent[:-1]] == [
        {"ok": "verify_caller"}, {"ok": "get_account"},
    ]
    assert realtime.sent[-1] == {"type": "response.create"}


def test_cancelled_tool_arguments_never_execute_and_next_turn_continues(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "dispatch", lambda state, name, args: seen.append(name) or {"ok": True})
    partial = {"type": "response.function_call_arguments.done", "name": "make_payment", "arguments": '{"amount":'}
    realtime = FakeRealtime([
        partial,
        _response([_tool("make_payment", arguments='{"amount":')], status="cancelled"),
        _response([_tool("get_account")]),
    ])
    asyncio.run(agent._pump_realtime_to_actor(realtime, SimpleNamespace(), CallSession()))
    assert seen == ["get_account"]
    assert [message["type"] for message in realtime.sent] == ["conversation.item.create", "response.create"]


@pytest.mark.parametrize("event,error", [
    ({"type": "error", "error": {"message": "Provider rejected the request"}}, RuntimeError),
    (_response(status="failed"), RuntimeError),
    (_response(status="incomplete"), RuntimeError),
    (_response([_tool(arguments='{"bad":')]), json.JSONDecodeError),
])
def test_provider_failures_and_malformed_completed_calls_are_visible(event, error):
    with pytest.raises(error):
        asyncio.run(agent._pump_realtime_to_actor(FakeRealtime([event]), SimpleNamespace(), CallSession()))


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
