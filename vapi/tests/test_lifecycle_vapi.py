"""What can be checked about the Vapi bridge without a platform key.

The inline assistant's payload, the server-tool webhook round trip, the
per-call registry and teardown, and the tunnel's short-circuit. The audio
legs and the hosted turn model need a live call and are validated on the
bench, not here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from rory_tools import CallSession
from rory_tools.prompt import GREETING, today_context
from rory_tools import tunnel
from rory_vapi import agent, tools, web

WEBHOOK = "https://rory.trycloudflare.com/tool"


def test_inline_assistant_carries_the_shared_prompt_greeting_tools_and_actor_rate():
    payload = agent.build_call_payload(WEBHOOK)
    assistant = payload["assistant"]
    assert payload["transport"] == {
        "provider": "vapi.websocket",
        "audioFormat": {"format": "pcm_s16le", "container": "raw", "sampleRate": 24000},
    }
    assert assistant["firstMessage"] is GREETING
    assert assistant["firstMessageMode"] == "assistant-speaks-first"
    assert assistant["model"]["messages"] == [
        {"role": "system", "content": f"{agent.AGENT_PROMPT}\n\n{today_context()}"}
    ]
    assert assistant["model"]["tools"] == tools.build_tools(WEBHOOK)
    assert all(tool["server"] == {"url": WEBHOOK} for tool in assistant["model"]["tools"])
    # 250, Vapi's default, truncates tool results.
    assert assistant["model"]["maxTokens"] == 500


def test_webhook_envelope_dispatches_in_order_with_string_results(monkeypatch):
    seen = []

    def dispatch(session, name, args):
        if name == "verify_caller":
            session.account = {"customer_id": "cus_verified"}
        else:
            assert session.account["customer_id"] == "cus_verified"
        seen.append((name, args))
        return {"ok": name}

    monkeypatch.setattr(tools, "dispatch", dispatch)
    envelope = [
        {"id": "1", "type": "function", "function": {"name": "verify_caller", "arguments": '{"account_number": "4417-88231"}'}},
        {"id": "2", "type": "function", "function": {"name": "get_account", "arguments": {}}},
    ]
    results = asyncio.run(tools.run_tool_calls(CallSession(), asyncio.Lock(), envelope))
    assert seen == [("verify_caller", {"account_number": "4417-88231"}), ("get_account", {})]
    assert [r["toolCallId"] for r in results] == ["1", "2"]
    # ``result`` is a string field on Vapi's side; a dict is dropped silently.
    assert all(isinstance(r["result"], str) for r in results)
    assert json.loads(results[0]["result"]) == {"ok": "verify_caller"}


def test_webhook_refuses_a_call_this_process_did_not_create(monkeypatch):
    monkeypatch.setattr(agent, "LIVE_CALLS", {})
    monkeypatch.setattr(web, "LIVE_CALLS", agent.LIVE_CALLS)
    client = TestClient(web.app)  # no context manager: no lifespan, no tunnel
    envelope = {"message": {"type": "tool-calls", "call": {"id": "stranger"}, "toolCallList": []}}
    assert client.post("/tool", json=envelope).status_code == 404


def test_webhook_resolves_against_the_registered_call(monkeypatch):
    live = {"call_1": agent.LiveCall()}
    monkeypatch.setattr(agent, "LIVE_CALLS", live)
    monkeypatch.setattr(web, "LIVE_CALLS", live)
    monkeypatch.setattr(tools, "dispatch", lambda session, name, args: {"echo": name})
    client = TestClient(web.app)
    envelope = {
        "message": {
            "type": "tool-calls",
            "call": {"id": "call_1"},
            "toolCallList": [{"id": "tc", "type": "function", "function": {"name": "get_account", "arguments": "{}"}}],
        }
    }
    response = client.post("/tool", json=envelope)
    assert response.status_code == 200
    assert response.json() == {"results": [{"toolCallId": "tc", "result": '{"echo": "get_account"}'}]}


class _FakeVapiSocket:
    """``websockets.connect`` stand-in: an async context manager that is also the socket."""

    def __init__(self, events):
        self.events = events
        self.sent = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def send(self, frame):
        self.sent.append(frame)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            await asyncio.sleep(3600)
        return self.events.pop(0)


@pytest.mark.parametrize("failed", [False, True])
def test_voice_bot_registers_the_call_then_forgets_it_and_propagates_pump_failure(monkeypatch, failed):
    live = {}
    monkeypatch.setattr(agent, "LIVE_CALLS", live)
    monkeypatch.setattr(
        agent, "create_call", AsyncMock(return_value={"id": "call_1", "transport": {"websocketCallUrl": "wss://x"}})
    )
    socket = _FakeVapiSocket(events=[])
    monkeypatch.setattr(agent.websockets, "connect", lambda url, **kw: socket)

    async def pump(actor_ws, vapi_ws):
        assert "call_1" in live  # registered before any audio flows
        if failed:
            raise RuntimeError("text frame")

    monkeypatch.setattr(agent, "_pump_actor_to_vapi", pump)
    actor_ws = SimpleNamespace(client=None)
    if failed:
        with pytest.raises(RuntimeError, match="text frame"):
            asyncio.run(agent.run_voice_ws_bot(actor_ws, WEBHOOK))
    else:
        asyncio.run(agent.run_voice_ws_bot(actor_ws, WEBHOOK))
    agent.create_call.assert_awaited_once_with(WEBHOOK)
    assert live == {}
    assert socket.closed


def test_voice_bot_returns_when_vapi_ends_the_call(monkeypatch):
    monkeypatch.setattr(agent, "LIVE_CALLS", {})
    monkeypatch.setattr(
        agent, "create_call", AsyncMock(return_value={"id": "call_1", "transport": {"websocketCallUrl": "wss://x"}})
    )
    socket = _FakeVapiSocket(
        events=[b"\x00\x01", json.dumps({"message": {"type": "status-update", "status": "ended"}})]
    )
    monkeypatch.setattr(agent.websockets, "connect", lambda url, **kw: socket)

    async def actor_never_hangs_up(actor_ws, vapi_ws):
        await asyncio.sleep(3600)

    monkeypatch.setattr(agent, "_pump_actor_to_vapi", actor_never_hangs_up)
    actor_ws = SimpleNamespace(client=None, send_bytes=AsyncMock())
    asyncio.run(asyncio.wait_for(agent.run_voice_ws_bot(actor_ws, WEBHOOK), timeout=5))
    actor_ws.send_bytes.assert_awaited_once_with(b"\x00\x01")


@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_distinguishes_failure_from_normal_end(monkeypatch, failed, code):
    state = SimpleNamespace(tool_webhook_url=WEBHOOK)
    ws = SimpleNamespace(client=None, client_state=WebSocketState.CONNECTED,
                         app=SimpleNamespace(state=state), accept=AsyncMock(), close=AsyncMock())
    runner = AsyncMock(side_effect=RuntimeError("broken pump") if failed else None)
    monkeypatch.setattr(web, "run_voice_ws_bot", runner)
    if failed:
        with pytest.raises(RuntimeError, match="broken pump"):
            asyncio.run(web.voice(ws))
    else:
        asyncio.run(web.voice(ws))
    assert runner.await_args.args[1] == WEBHOOK
    assert ws.close.await_args.kwargs["code"] == code


def test_preset_public_url_skips_the_tunnel(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://rory.example.com/")
    monkeypatch.setattr(tunnel.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned a tunnel"))
    assert tunnel.public_base_url(8008) == ("https://rory.example.com", None)


def test_preset_public_url_must_be_https(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://rory.example.com")
    with pytest.raises(RuntimeError, match="https"):
        tunnel.public_base_url(8008)
