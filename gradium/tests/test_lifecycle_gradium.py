"""What the gradbot bridge does with a session, without a vendor.

The parity run proves the declaration; these prove the round trip with
``gradbot.run`` stubbed: the actor's frames reach gradbot untouched, gradbot's
48 kHz PCM leaves at the actor's 24 kHz, a tool call is answered through the
shared dispatcher with its result as JSON, and a call gradbot re-issues while
the first is still running is answered with the first result instead of
running the tool twice.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import math
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import gradbot
from starlette.websockets import WebSocketDisconnect

from rory_gradium import agent, tools


def _pcm48_sine(n_samples: int) -> bytes:
    return b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 48000))) for i in range(n_samples))


def _audio(data: bytes):
    return SimpleNamespace(msg_type="audio", data=data)


def _event(event_type: str):
    return SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type=event_type))


def _tool_call(call_id: str, name: str, args: dict):
    handle = SimpleNamespace(send=AsyncMock())
    call = SimpleNamespace(call_id=call_id, tool_name=name, args_json=json.dumps(args))
    return SimpleNamespace(msg_type="tool_call", tool_call=call, tool_call_handle=handle), handle


class _Output:
    """gradbot's output handle, playing back a script; ends once every tool handle has been answered."""

    def __init__(self, messages, handles):
        self._messages = list(messages)
        self._handles = handles

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        while not all(h.send.await_count for h in self._handles):
            await asyncio.sleep(0)
        return None


class _Actor:
    """The Veris actor: sends its frames, then holds the socket open until gradbot ends the session."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.send_bytes = AsyncMock()

    async def receive_bytes(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.Event().wait()


def _stub_gradbot(monkeypatch, messages, handles):
    started = {}
    input_handle = SimpleNamespace(send_audio=AsyncMock(), close=AsyncMock())

    async def run(**kwargs):
        started.update(kwargs)
        return input_handle, _Output(messages, handles)

    monkeypatch.setattr(gradbot, "run", run)
    monkeypatch.setenv("GRADIUM_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    return started, input_handle


def test_session_bridges_audio_both_ways_and_answers_tools_through_the_shared_gate(monkeypatch):
    seen = []

    def dispatch(session, name, args):
        seen.append((session, name, args))
        return {"verified": True}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    pcm48 = _pcm48_sine(960)
    call_msg, handle = _tool_call("c1", "verify_caller", {"account_number": "4417-88231"})
    started, input_handle = _stub_gradbot(monkeypatch, [_audio(pcm48), _audio(b""), call_msg], [handle])
    actor = _Actor([b"\x01\x00" * 480, b"\x02\x00" * 480])

    asyncio.run(agent.run_voice_ws_bot(actor))

    # The session is gradbot's, configured from the environment and the shared tools.
    assert started["gradium_api_key"] == "gsk_test"
    assert started["llm_api_key"] == "sk_test"
    assert started["llm_model_name"] == agent.GRADIUM_LLM_MODEL
    assert started["input_format"] == gradbot.AudioFormat.Pcm
    assert started["output_format"] == gradbot.AudioFormat.Pcm
    assert [t.name for t in started["session_config"].tools] == [t.name for t in tools.TOOLS]

    # Actor frames go in untouched, at gradbot's 24 kHz input rate.
    assert [c.args[0] for c in input_handle.send_audio.await_args_list] == [b"\x01\x00" * 480, b"\x02\x00" * 480]

    # gradbot's 48 kHz output is halved to the actor's 24 kHz; an empty chunk is not forwarded.
    assert actor.send_bytes.await_count == 1
    pcm24 = actor.send_bytes.await_args.args[0]
    assert len(pcm24) == len(pcm48) // 2
    assert pcm24 == audioop.ratecv(pcm48, 2, 1, 48000, 24000, None)[0]

    # The tool call ran through the shared dispatcher, on one CallSession, and its result went back as JSON.
    assert [(name, args) for _, name, args in seen] == [("verify_caller", {"account_number": "4417-88231"})]
    assert type(seen[0][0]).__name__ == "CallSession"
    handle.send.assert_awaited_once_with(json.dumps({"verified": True}))


def test_a_call_reissued_while_the_first_runs_is_answered_with_the_first_result(monkeypatch):
    """gradbot re-prompts the model during a pending tool call, and the model
    often issues the same call again. Running it twice would be a second
    verification attempt or a second payment."""
    release = asyncio.Event()
    calls = []

    def dispatch(session, name, args):
        calls.append(name)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return {"balance_cents": 21407}

    monkeypatch.setattr(agent, "dispatch", dispatch)
    first, first_handle = _tool_call("c1", "get_account", {})
    again, again_handle = _tool_call("c2", "get_account", {})
    later, later_handle = _tool_call("c3", "get_account", {})

    async def run():
        tool_calls = agent.ToolCalls(agent.CallSession())
        tool_calls.pushed_to_llm()
        tool_calls.llm_started()
        tool_calls.start(first.tool_call, first.tool_call_handle)
        # gradbot re-prompts (push) and that generation starts while c1 is still running.
        tool_calls.pushed_to_llm()
        tool_calls.llm_started()
        tool_calls.start(again.tool_call, again.tool_call_handle)
        await asyncio.sleep(0.05)
        assert first_handle.send.await_count == 0 and again_handle.send.await_count == 0
        release.set()
        while not (first_handle.send.await_count and again_handle.send.await_count):
            await asyncio.sleep(0)
        # A prompt built after the result went back is the model's own repeat: it runs.
        tool_calls.pushed_to_llm()
        tool_calls.llm_started()
        tool_calls.start(later.tool_call, later.tool_call_handle)
        while not later_handle.send.await_count:
            await asyncio.sleep(0)
        await tool_calls.close()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()

    assert calls == ["get_account", "get_account"]
    result = json.dumps({"balance_cents": 21407})
    first_handle.send.assert_awaited_once_with(result)
    again_handle.send.assert_awaited_once_with(result)
    later_handle.send.assert_awaited_once_with(result)


def test_actor_hangup_closes_gradbots_input(monkeypatch):
    started, input_handle = _stub_gradbot(monkeypatch, [], [])

    class _Hangup:
        send_bytes = AsyncMock()

        async def receive_bytes(self):
            raise WebSocketDisconnect(code=1000)

    asyncio.run(agent.run_voice_ws_bot(_Hangup()))
    input_handle.close.assert_awaited_once()
