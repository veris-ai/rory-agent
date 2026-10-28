"""Turn logic with fake AWS legs: Lex, AgentCore and Polly are replaced at their call sites."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json

import pytest
from rory_agentcore import agent
from starlette.websockets import WebSocketDisconnect

ACTOR_BYTES_PER_S = agent.BYTES_PER_S


class FakeActor:
    """Enough of a Starlette WebSocket for the pump and the speaker."""

    def __init__(self, frames=()):
        self._frames = list(frames)
        self.sent: list[bytes] = []

    async def receive_bytes(self) -> bytes:
        if not self._frames:
            raise WebSocketDisconnect(1000)
        return self._frames.pop(0)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)


def _marks(*words):
    """(value, start, end, time-ms) tuples into Polly-shaped speech marks."""
    return [
        {"time": t, "type": "word", "start": s, "end": e, "value": v}
        for v, s, e, t in words
    ]


def _call(monkeypatch, scripted_rounds, tool_result=None):
    """A _Call whose model rounds come from ``scripted_rounds``, one per invocation."""
    rounds = iter(scripted_rounds)
    monkeypatch.setattr(agent, "_agent_turn", lambda session_id, messages: next(rounds))
    monkeypatch.setattr(agent, "_synthesize", lambda text: (_marks(("x", 0, 1, 0)), bytes(ACTOR_BYTES_PER_S // 2)))
    if tool_result is not None:
        monkeypatch.setattr(agent, "dispatch", lambda session, name, args: tool_result)
    return agent._Call(FakeActor())


def test_tool_round_runs_the_shared_dispatcher_and_answers_with_its_result(monkeypatch):
    seen = {}

    def fake_dispatch(session, name, args):
        seen["call"] = (name, args)
        return {"balance": 42}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)
    call = _call(
        monkeypatch,
        [
            {"content": None, "tool_calls": [{"id": "t1", "name": "get_balance", "arguments": {"x": 1}}]},
            {"content": "Your balance is 42 dollars.", "tool_calls": []},
        ],
    )
    spoken = []

    async def record_say(text):
        spoken.append(text)

    monkeypatch.setattr(call, "_say", record_say)
    call.messages.append({"role": "user", "content": "what do I owe"})
    asyncio.run(call._take_turn("what do I owe"))

    assert seen["call"] == ("get_balance", {"x": 1})
    tool_message = call.messages[-2]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "t1",
        "name": "get_balance",
        "content": json.dumps({"balance": 42}),
    }
    assert call.messages[-1] == {
        "role": "assistant",
        "content": "Your balance is 42 dollars.",
        "tool_calls": [],
    }
    assert spoken == ["Your balance is 42 dollars."]


def test_tool_loop_that_never_settles_fails_loudly(monkeypatch):
    looping = {"content": None, "tool_calls": [{"id": "t", "name": "get_balance", "arguments": {}}]}
    call = _call(monkeypatch, [looping] * agent.MAX_TOOL_ROUNDS, tool_result={})
    call.messages.append({"role": "user", "content": "hi"})
    with pytest.raises(RuntimeError, match="did not settle"):
        asyncio.run(call._take_turn("hi"))


def test_barge_in_keeps_only_the_words_that_started_before_the_cut(monkeypatch):
    call = _call(monkeypatch, [])
    text = "Sure, here it is."
    call._marks = _marks(("Sure", 0, 4, 0), ("here", 6, 10, 400), ("it", 11, 13, 700), ("is", 14, 16, 900))
    call._sent_bytes = 500 * agent.BYTES_PER_MS  # 500 ms heard
    call.messages.append({"role": "assistant", "content": text, "tool_calls": []})

    call._trim_interrupted_reply(text)

    assert call.messages[-1]["content"] == "Sure, here" + agent.INTERRUPTED_MARKER


def test_barge_in_before_any_audio_drops_the_reply(monkeypatch):
    call = _call(monkeypatch, [])
    call._sent_bytes = 0
    call.messages.append({"role": "assistant", "content": "Hello there", "tool_calls": []})

    call._trim_interrupted_reply("Hello there")

    assert all(m["content"] != "Hello there" for m in call.messages)


def test_agentcore_request_carries_the_shared_tools_and_session(monkeypatch):
    captured = {}

    class FakeAgentCore:
        def invoke_agent_runtime(self, **kwargs):
            captured.update(kwargs)
            body = json.dumps({"content": "ok", "tool_calls": []}).encode()
            return {"response": _Body(body)}

    class _Body:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    monkeypatch.setattr(agent, "_agentcore", FakeAgentCore())
    monkeypatch.setattr(agent, "_runtime_arn", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r")

    reply = agent._agent_turn("s" * 36, [{"role": "user", "content": "hi"}])

    assert reply == {"content": "ok", "tool_calls": []}
    assert captured["agentRuntimeArn"].endswith("runtime/r")
    assert captured["runtimeSessionId"] == "s" * 36
    body = json.loads(captured["payload"])
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == agent.TOOLS


def test_lex_transcript_is_gzipped_base64_in_a_header(monkeypatch):
    header = base64.b64encode(gzip.compress(b"I want to pay my bill")).decode()

    class _Closable:
        def close(self):
            pass

    class FakeLex:
        def recognize_utterance(self, **kwargs):
            assert kwargs["requestContentType"] == "audio/l16; rate=16000; channels=1"
            assert kwargs["inputStream"] == b"\x01\x02"
            return {"inputTranscript": header, "audioStream": _Closable()}

    monkeypatch.setattr(agent, "_lex", FakeLex())
    monkeypatch.setattr(agent, "_bot_id", "bot")
    monkeypatch.setattr(agent, "_bot_alias_id", "alias")

    assert agent._recognize("session", b"\x01\x02") == "I want to pay my bill"


def test_audio_is_resampled_to_the_rate_each_leg_speaks(monkeypatch):
    one_second_at_24k = bytes(ACTOR_BYTES_PER_S)
    assert len(agent._to_rate(one_second_at_24k, agent.ACTOR_RATE_HZ, agent.LEX_RATE_HZ)) == 16000 * 2
    one_second_at_16k = bytes(16000 * 2)
    # ratecv rounds the sample count, so the length can be a sample short.
    assert abs(len(agent._to_rate(one_second_at_16k, agent.POLLY_RATE_HZ, agent.ACTOR_RATE_HZ)) - ACTOR_BYTES_PER_S) <= 4
