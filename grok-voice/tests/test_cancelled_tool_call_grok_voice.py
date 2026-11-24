"""Tool calls follow Grok's response lifecycle.

Only a completed response runs its tools. Every call gets an output, the next
response is requested once after all of them, and a slow tool never stops the
bridge from reading Grok's events.
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock

from rory_tools import CallSession

from rory_grok_voice import agent


class FakeGrok:
    """Async-iterable stand-in for the Grok websocket; records what we send."""

    def __init__(self, events: list[dict], settle_s: float = 0.05):
        self._events = events
        self._settle_s = settle_s
        self.sent: list[dict] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        deadline = asyncio.get_running_loop().time() + self._settle_s
        while not self._events:
            if asyncio.get_running_loop().time() >= deadline:
                raise StopAsyncIteration
            await asyncio.sleep(0.002)
        await asyncio.sleep(0)
        return json.dumps(self._events.pop(0))

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _created(response_id: str) -> dict:
    return {"type": "response.created", "response": {"id": response_id}}


def _args(response_id: str, call_id: str, name: str, arguments: str) -> dict:
    return {
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _done(response_id: str, status: str, n_calls: int = 0) -> dict:
    return {
        "type": "response.done",
        "response": {
            "id": response_id,
            "status": status,
            "output": [{"type": "function_call"} for _ in range(n_calls)],
        },
    }


def _run(events: list[dict], monkeypatch, settle_s: float = 0.05):
    calls: list[tuple[str, dict]] = []

    def fake_dispatch(session_state, name, args):
        calls.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    gk_ws = FakeGrok(events, settle_s)
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_grok_to_actor(gk_ws, actor_ws, CallSession()))
    return calls, gk_ws.sent


def test_cancelled_response_runs_no_tools(monkeypatch):
    events = [
        _created("r1"),
        _args("r1", "call_cancelled", "verify_caller", '{"account_number":"0000-00001"}'),
        _done("r1", "cancelled"),
        _created("r2"),
        _args("r2", "call_ok", "verify_caller", '{"account_number":"0000-00002"}'),
        _done("r2", "completed", 1),
    ]

    calls, sent = _run(events, monkeypatch)

    assert calls == [("verify_caller", {"account_number": "0000-00002"})]
    outputs = [m for m in sent if m["type"] == "conversation.item.create"]
    assert [m["item"]["call_id"] for m in outputs] == ["call_ok"]
    assert [m["type"] for m in sent].count("response.create") == 1


def test_empty_arguments_still_dispatch_a_no_argument_tool(monkeypatch):
    events = [_created("r1"), _args("r1", "call_1", "get_account", ""), _done("r1", "completed", 1)]

    calls, sent = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert [m["type"] for m in sent] == ["conversation.item.create", "response.create"]


def test_invalid_json_on_a_completed_response_still_gets_an_output(monkeypatch):
    events = [
        _created("r1"),
        _args("r1", "call_bad", "verify_caller", '{"account_number":"00'),
        _done("r1", "completed", 1),
    ]

    calls, sent = _run(events, monkeypatch)

    assert calls == []
    assert sent[0]["item"]["call_id"] == "call_bad"
    assert "error" in json.loads(sent[0]["item"]["output"])
    assert sent[1]["type"] == "response.create"


def test_parallel_calls_all_answer_before_one_response_create(monkeypatch):
    events = [
        _created("r1"),
        _args("r1", "call_1", "get_account", "{}"),
        _args("r1", "call_2", "explain_bill", "{}"),
        _done("r1", "completed", 2),
    ]

    calls, sent = _run(events, monkeypatch)

    assert [name for name, _ in calls] == ["get_account", "explain_bill"]
    assert [m["type"] for m in sent] == [
        "conversation.item.create",
        "conversation.item.create",
        "response.create",
    ]


def test_response_from_server_vad_consumes_the_outputs(monkeypatch):
    monkeypatch.setattr(agent, "AUTO_RESPONSE_GRACE_S", 0.2)
    events = [
        _created("r1"),
        _args("r1", "call_1", "get_account", "{}"),
        {"type": "input_audio_buffer.speech_started"},
        _done("r1", "completed", 1),
        {"type": "input_audio_buffer.speech_stopped"},
    ]

    async def run():
        monkeypatch.setattr(agent, "dispatch", lambda *_: {"ok": True})
        gk_ws = FakeGrok(events, settle_s=0.4)
        task = asyncio.create_task(agent._pump_grok_to_actor(gk_ws, AsyncMock(), CallSession()))
        while not any(m["type"] == "conversation.item.create" for m in gk_ws.sent):
            await asyncio.sleep(0.005)
        gk_ws._events.append(_created("r2"))
        await task
        return gk_ws.sent

    sent = asyncio.run(run())

    assert [m["type"] for m in sent] == ["conversation.item.create"]


def test_caller_turn_without_a_response_still_gets_one(monkeypatch):
    monkeypatch.setattr(agent, "AUTO_RESPONSE_GRACE_S", 0.05)
    events = [
        _created("r1"),
        _args("r1", "call_1", "get_account", "{}"),
        {"type": "input_audio_buffer.speech_started"},
        _done("r1", "completed", 1),
        {"type": "input_audio_buffer.speech_stopped"},
    ]

    _, sent = _run(events, monkeypatch, settle_s=0.3)

    assert [m["type"] for m in sent] == ["conversation.item.create", "response.create"]


def test_error_event_does_not_end_the_call(monkeypatch):
    events = [
        {"type": "error", "error": {"type": "internal_error", "code": "internal_error"}},
        _created("r1"),
        _args("r1", "call_1", "get_account", "{}"),
        _done("r1", "completed", 1),
    ]

    calls, sent = _run(events, monkeypatch)

    assert calls == [("get_account", {})]
    assert sent[-1]["type"] == "response.create"


def test_slow_tool_does_not_block_the_reader(monkeypatch):
    release = threading.Event()

    def slow_dispatch(session_state, name, args):
        release.wait(2)
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", slow_dispatch)

    async def run():
        turn = agent._TurnState()
        rounds: asyncio.Queue = asyncio.Queue()
        events = [_created("r1"), _args("r1", "call_1", "get_account", "{}"), _done("r1", "completed", 1)]
        gk_ws = FakeGrok(events, settle_s=0.2)
        playout = agent._Playout(AsyncMock())
        worker = asyncio.create_task(agent._run_tool_rounds(rounds, gk_ws, CallSession(), turn, playout))
        events.append({"type": "input_audio_buffer.speech_started"})
        reader = asyncio.create_task(agent._read_grok(gk_ws, playout, turn, rounds))
        while not turn.caller_speaking:
            await asyncio.sleep(0.005)
        read_while_tool_ran = not release.is_set()
        release.set()
        await reader
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        return read_while_tool_ran

    assert asyncio.run(run())
