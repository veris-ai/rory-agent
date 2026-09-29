"""What can be checked about the ElevenLabs bridge without a platform key.

The stored agent's config, the client-tool round trip through the SDK's own
scheduler, and the per-call teardown. The audio legs and the hosted turn model
need a live conversation and are validated on the bench, not here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from rory_elevenlabs import agent, tools, web
from rory_tools import CallSession
from rory_tools.prompt import GREETING, today_context


def test_stored_agent_carries_the_shared_prompt_greeting_and_actor_rate():
    config = agent._build_conversation_config()
    assert config["agent"]["first_message"] is GREETING
    assert config["agent"]["prompt"]["prompt"] == f"{agent.AGENT_PROMPT}\n\n{today_context()}"
    assert config["agent"]["prompt"]["tools"] is tools.TOOLS
    assert config["tts"]["agent_output_audio_format"] == "pcm_24000"
    assert config["asr"]["user_input_audio_format"] == "pcm_24000"


def test_pinned_agent_id_skips_provisioning(monkeypatch):
    monkeypatch.setenv("AGENT_ID", "agent_pinned")
    client = SimpleNamespace(conversational_ai=None)
    assert agent.ensure_agent(client) == "agent_pinned"


def test_client_tools_dispatch_in_order_without_the_sdk_correlation_id(monkeypatch):
    seen = []

    def dispatch(session, name, args):
        assert "tool_call_id" not in args
        if name == "verify_caller":
            session.account = {"customer_id": "cus_verified"}
        else:
            assert session.account["customer_id"] == "cus_verified"
        seen.append((name, args))
        return {"ok": name}

    monkeypatch.setattr(tools, "dispatch", dispatch)
    responses = []

    async def run():
        client_tools = tools.build_client_tools(CallSession(), asyncio.get_running_loop())
        client_tools.start()
        # The SDK schedules each call as its own task; both land before
        # either handler has run, which is the race the lock exists for.
        client_tools.execute_tool("verify_caller", {"tool_call_id": "1", "account_number": "4417-88231"}, responses.append)
        client_tools.execute_tool("get_account", {"tool_call_id": "2"}, responses.append)
        while len(responses) < 2:
            await asyncio.sleep(0)
        client_tools.stop()

    asyncio.run(run())
    assert seen == [("verify_caller", {"account_number": "4417-88231"}), ("get_account", {})]
    assert [r["tool_call_id"] for r in responses] == ["1", "2"]
    assert all(r["is_error"] is False for r in responses)
    assert json.loads(responses[0]["result"]) == {"ok": "verify_caller"}


@pytest.mark.parametrize("failed", [False, True])
def test_voice_bot_ends_the_conversation_and_propagates_pump_failure(monkeypatch, failed):
    # The platform side never hangs up on its own; only end_session ends it,
    # which is the SDK contract the teardown relies on.
    ended = asyncio.Event()

    async def end_session():
        ended.set()

    conversation = SimpleNamespace(
        start_session=AsyncMock(),
        end_session=AsyncMock(side_effect=end_session),
        wait_for_session_end=ended.wait,
    )
    monkeypatch.setattr(agent, "AsyncConversation", lambda **kwargs: conversation)

    async def pump(actor_ws, audio):
        if failed:
            raise RuntimeError("text frame")

    monkeypatch.setattr(agent, "_pump_actor_to_elevenlabs", pump)
    actor_ws = SimpleNamespace(client=None)
    if failed:
        with pytest.raises(RuntimeError, match="text frame"):
            asyncio.run(agent.run_voice_ws_bot(actor_ws, object(), "agent_x"))
    else:
        asyncio.run(agent.run_voice_ws_bot(actor_ws, object(), "agent_x"))
    conversation.start_session.assert_awaited_once()
    conversation.end_session.assert_awaited_once()


@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_distinguishes_failure_from_normal_end(monkeypatch, failed, code):
    state = SimpleNamespace(eleven=object(), agent_id="agent_x")
    ws = SimpleNamespace(client=None, client_state=WebSocketState.CONNECTED,
                         app=SimpleNamespace(state=state), accept=AsyncMock(), close=AsyncMock())
    runner = AsyncMock(side_effect=RuntimeError("broken pump") if failed else None)
    monkeypatch.setattr(web, "run_voice_ws_bot", runner)
    if failed:
        with pytest.raises(RuntimeError, match="broken pump"):
            asyncio.run(web.voice(ws))
    else:
        asyncio.run(web.voice(ws))
    assert runner.await_args.args[1:] == (state.eleven, "agent_x")
    assert ws.close.await_args.kwargs["code"] == code
