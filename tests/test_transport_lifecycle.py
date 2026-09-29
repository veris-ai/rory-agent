import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors, types
from starlette.websockets import WebSocketState

from rory_gemini_live import agent as gemini
from rory_tools import CallSession
from rory_tools.prompt import GREETING


@pytest.mark.parametrize("transport", ["pipecat", "gemini"])
def test_missing_billing_credential_fails_before_health(monkeypatch, transport):
    from fastapi.testclient import TestClient
    from rory_pipecat.web import app as pipecat_app
    from rory_gemini_live.web import app as gemini_app
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(key, "test")
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="STRIPE_API_KEY"):
        with TestClient(pipecat_app if transport == "pipecat" else gemini_app):
            pass


@pytest.mark.parametrize("transport", ["pipecat", "gemini"])
@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_distinguishes_failure_from_normal_end(monkeypatch, transport, failed, code):
    from rory_pipecat import web as pipecat_web
    from rory_gemini_live import web as gemini_web
    web = pipecat_web if transport == "pipecat" else gemini_web
    ws = SimpleNamespace(client=None, client_state=WebSocketState.CONNECTED,
                         accept=AsyncMock(), close=AsyncMock())
    runner = AsyncMock(side_effect=RuntimeError("broken pump") if failed else None)
    monkeypatch.setattr(web, "run_voice_ws_bot", runner)
    if failed:
        with pytest.raises(RuntimeError, match="broken pump"):
            asyncio.run(web.voice(ws))
    else:
        asyncio.run(web.voice(ws))
    assert ws.close.await_args.kwargs["code"] == code


@pytest.mark.parametrize("failed", [False, True])
def test_gemini_seeds_real_trigger_and_propagates_pump_failure(monkeypatch, failed):
    live = SimpleNamespace(send_client_content=AsyncMock())
    cancelled = []

    @asynccontextmanager
    async def connect(**kwargs):
        yield live

    async def up(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def down(*args):
        if failed:
            raise RuntimeError("send_tool_response failed")

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(gemini.genai, "Client", lambda **kwargs: SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect))))
    monkeypatch.setattr(gemini, "_pump_actor_to_gemini", up)
    monkeypatch.setattr(gemini, "_pump_gemini_to_actor", down)
    if failed:
        with pytest.raises(RuntimeError, match="send_tool_response failed"):
            asyncio.run(gemini.run_voice_ws_bot(SimpleNamespace(client=None)))
    else:
        asyncio.run(gemini.run_voice_ws_bot(SimpleNamespace(client=None)))
    assert cancelled == [True]
    sent = live.send_client_content.await_args.kwargs
    assert sent["turns"].role == "user"
    assert sent["turns"].parts[0].text == f"<call connected — greet the caller with exactly: {GREETING}>"
    assert sent["turn_complete"] is True


def test_gemini_dispatches_batch_in_order_and_accepts_clean_provider_close(monkeypatch):
    calls = [types.FunctionCall(id="1", name="verify_caller", args={}),
             types.FunctionCall(id="2", name="get_account", args={})]
    turns = []

    async def receive():
        if turns:
            raise errors.APIError(1000, {"message": "normal close"})
        turns.append(True)
        yield types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=calls))

    seen = []
    def dispatch(session, name, args):
        if name == "verify_caller":
            session.account = {"customer_id": "cus_verified"}
        else:
            assert session.account["customer_id"] == "cus_verified"
        seen.append(name)
        return {"ok": True}

    monkeypatch.setattr(gemini, "dispatch", dispatch)
    live = SimpleNamespace(receive=receive, send_tool_response=AsyncMock())
    asyncio.run(gemini._pump_gemini_to_actor(live, SimpleNamespace(), CallSession()))
    assert seen == ["verify_caller", "get_account"]
    assert [r.id for r in live.send_tool_response.await_args.kwargs["function_responses"]] == ["1", "2"]


def test_pipecat_uses_sequential_tool_execution(monkeypatch):
    from rory_pipecat.agent import _build_llm
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert _build_llm()._run_in_parallel is False


def test_pipecat_seeds_assistant_greeting_and_speaks_it_on_connect(monkeypatch):
    from rory_pipecat import agent
    captured = []

    class ContextCaptured(Exception):
        pass

    def context(messages, tools):
        captured.extend(messages)
        raise ContextCaptured

    for name in ("_build_stt", "_build_llm", "_build_tts"):
        monkeypatch.setattr(agent, name, lambda: None)
    monkeypatch.setattr(agent, "LLMContext", context)
    with pytest.raises(ContextCaptured):
        agent._build_pipeline_task(SimpleNamespace())
    assert captured[-1] == {"role": "assistant", "content": GREETING}

    handlers = {}
    def event_handler(name):
        def register(handler):
            handlers[name] = handler
            return handler
        return register

    transport = SimpleNamespace(event_handler=event_handler)
    task = SimpleNamespace(queue_frames=AsyncMock(), cancel=AsyncMock())
    async def run(_):
        await handlers["on_client_connected"](transport, None)
    monkeypatch.setattr(agent, "_build_pipeline_task", lambda _: task)
    monkeypatch.setattr(agent, "PipelineRunner", lambda **kwargs: SimpleNamespace(run=run))
    asyncio.run(agent._run_with_transport(transport, "test"))
    frames = task.queue_frames.await_args.args[0]
    assert len(frames) == 1
    assert frames[0].text == GREETING
