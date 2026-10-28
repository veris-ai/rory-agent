"""Each piece of the candidate against fakes of the services around it."""

from __future__ import annotations

import base64
import gzip
import json
from types import SimpleNamespace

import pytest
from rory_agentcore import agent, init_lambda, runtime, tools_lambda
from rory_agentcore.contract import (
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    ATTR_STRIPE_API_BASE,
    CALL_CONNECTED,
    INTERRUPTED_MARKER,
    SESSION_ARG,
)

from rory_tools.prompt import GREETING


def _compressed(value) -> str:
    """Lex's compressed fields: JSON (a transcript is a JSON string), gzipped, then base64."""
    return base64.b64encode(gzip.compress(json.dumps(value).encode())).decode()


class _Body:
    def __init__(self, data: bytes = b""):
        self._data = data

    def read(self):
        return self._data

    def close(self):
        pass


# ----------------------------------------------------------------------- the pod


def test_lex_turn_carries_the_session_attributes_and_returns_the_agent_reply(monkeypatch):
    seen = {}

    class FakeLex:
        def recognize_utterance(self, **kwargs):
            seen.update(kwargs)
            return {
                "inputTranscript": _compressed("I want to pay my bill"),
                "messages": _compressed([{"contentType": "PlainText", "content": "Sure."},
                                         {"contentType": "PlainText", "content": "Account number?"}]),
                "audioStream": _Body(),
            }

    monkeypatch.setattr(agent, "_lex", FakeLex(), raising=False)
    monkeypatch.setattr(agent, "_bot_id", "bot", raising=False)
    monkeypatch.setattr(agent, "_bot_alias_id", "alias", raising=False)

    transcript, reply = agent._recognize("s1", b"\x01\x02", {ATTR_STRIPE_API_BASE: "https://twin/stripe"})

    assert (transcript, reply) == ("I want to pay my bill", "Sure. Account number?")
    assert seen["requestContentType"] == "audio/l16; rate=16000; channels=1"
    assert seen["inputStream"] == b"\x01\x02"
    state = json.loads(gzip.decompress(base64.b64decode(seen["sessionState"])))
    assert state == {"sessionAttributes": {ATTR_STRIPE_API_BASE: "https://twin/stripe"}}


def test_lex_turn_with_no_reply_is_silence(monkeypatch):
    class FakeLex:
        def recognize_utterance(self, **kwargs):
            return {"inputTranscript": _compressed(""), "audioStream": _Body()}

    monkeypatch.setattr(agent, "_lex", FakeLex(), raising=False)
    monkeypatch.setattr(agent, "_bot_id", "bot", raising=False)
    monkeypatch.setattr(agent, "_bot_alias_id", "alias", raising=False)

    assert agent._recognize("s1", b"", {}) == ("", "")


def test_barge_in_keeps_only_the_words_that_started_before_the_cut():
    call = agent._Call(SimpleNamespace(client=None))
    call._marks = [
        {"time": 0, "start": 0, "end": 4}, {"time": 400, "start": 6, "end": 10},
        {"time": 700, "start": 11, "end": 13}, {"time": 900, "start": 14, "end": 16},
    ]
    call._sent_bytes = 500 * agent.BYTES_PER_MS  # 500 ms played

    assert call._heard_part("Sure, here it is.") == "Sure, here"


def test_barge_in_before_any_audio_keeps_nothing():
    call = agent._Call(SimpleNamespace(client=None))
    call._marks = [{"time": 0, "start": 0, "end": 5}]
    call._sent_bytes = 0

    assert call._heard_part("Hello there") == ""


def test_audio_is_resampled_to_the_rate_each_leg_speaks():
    one_second_at_24k = bytes(agent.BYTES_PER_S)
    assert len(agent._to_rate(one_second_at_24k, agent.ACTOR_RATE_HZ, agent.LEX_RATE_HZ)) == 16000 * 2
    # ratecv rounds the sample count, so the length can be a sample short.
    one_second_at_16k = bytes(16000 * 2)
    assert abs(len(agent._to_rate(one_second_at_16k, agent.POLLY_RATE_HZ, agent.ACTOR_RATE_HZ)) - agent.BYTES_PER_S) <= 4


# ------------------------------------------------------------------- the runtime


class FakeBedrock:
    """Answers each converse call with the next scripted (stopReason, content) round."""

    def __init__(self, rounds):
        self._rounds = iter(rounds)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        stop, content = next(self._rounds)
        return {
            "stopReason": stop,
            "output": {"message": {"role": "assistant", "content": content}},
            "usage": {"inputTokens": 100, "outputTokens": 10},
        }


def test_new_session_seeds_the_greeting_the_pod_already_spoke():
    session = runtime._new_session(None)
    assert session["system"][0] == {"text": runtime.AGENT_PROMPT}
    assert session["messages"] == [
        {"role": "user", "content": [{"text": CALL_CONNECTED}]},
        {"role": "assistant", "content": [{"text": GREETING}]},
    ]


def test_tool_round_adds_the_session_after_the_model_chose_the_call(monkeypatch):
    bedrock = FakeBedrock([
        ("tool_use", [{"toolUse": {"toolUseId": "t1", "name": "get_balance", "input": {"x": 1}}}]),
        ("end_turn", [{"text": "Your balance is 42 dollars."}]),
    ])
    calls = []
    monkeypatch.setattr(runtime, "_bedrock", bedrock)
    monkeypatch.setattr(runtime.gateway, "call_tool", lambda name, args: calls.append((name, args)) or {"balance": 42})
    session = runtime._new_session(None)

    reply, turn = runtime.take_turn(session, "sess-1", "what do I owe")

    assert reply == "Your balance is 42 dollars."
    # The turn the pod reports: the caller's words, the tool call, its result, the reply.
    assert [m["role"] for m in turn["messages"]] == ["user", "assistant", "user", "assistant"]
    assert [r["output_index"] for r in turn["rounds"]] == [1, 3]
    assert turn["rounds"][0]["input_tokens"] == 100
    assert turn["system"] == [runtime.AGENT_PROMPT, session["system"][1]["text"]]
    assert calls == [("get_balance", {"x": 1, SESSION_ARG: "sess-1"})]
    assert session["messages"][-2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"balance": 42}}]}}],
    }
    assert bedrock.calls[0]["toolConfig"] == {"tools": runtime.CONVERSE_TOOLS}


def test_tool_loop_that_never_settles_fails_loudly(monkeypatch):
    looping = ("tool_use", [{"toolUse": {"toolUseId": "t", "name": "get_balance", "input": {}}}])
    monkeypatch.setattr(runtime, "_bedrock", FakeBedrock([looping] * runtime.MAX_TOOL_ROUNDS))
    monkeypatch.setattr(runtime.gateway, "call_tool", lambda name, args: {})

    with pytest.raises(RuntimeError, match="did not settle"):
        runtime.take_turn(runtime._new_session(None), "sess-1", "hi")


def test_a_failed_turn_leaves_the_history_as_it_was(monkeypatch):
    class Refusing:
        def converse(self, **kwargs):
            raise RuntimeError("ValidationException")

    monkeypatch.setattr(runtime, "_bedrock", Refusing())
    session = runtime._new_session(None)
    before = [dict(m) for m in session["messages"]]

    with pytest.raises(RuntimeError):
        runtime.take_turn(session, "sess-1", "hello")

    assert session["messages"] == before


def test_barge_in_rewrites_the_last_reply_to_what_was_heard(monkeypatch):
    monkeypatch.setattr(runtime, "_bedrock", FakeBedrock([("end_turn", [{"text": "Sure."}])]))
    runtime._sessions.clear()
    heard = "Thanks for calling" + INTERRUPTED_MARKER

    runtime.invocations({"text": "I need help", ATTR_INTERRUPTED_REPLY: heard}, "sess-2")

    messages = runtime._sessions["sess-2"]["messages"]
    assert messages[1] == {"role": "assistant", "content": [{"text": heard}]}
    assert messages[-1]["content"] == [{"text": "Sure."}]


def test_gateway_reads_a_tool_result_from_either_wire_shape():
    result = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{\"ok\": true}"}]}}
    assert runtime.gateway._parse(json.dumps(result), "application/json") == result
    sse = f"event: message\ndata: {json.dumps(result)}\n\n"
    assert runtime.gateway._parse(sse, "text/event-stream") == result


# ------------------------------------------------------------- the init Lambda


class FakeTable:
    def __init__(self, item=None):
        self.item = item
        self.puts = []
        self.updates = []

        class _Exceptions:
            ConditionalCheckFailedException = type("ConditionalCheckFailedException", (Exception,), {})

        self.meta = SimpleNamespace(client=SimpleNamespace(exceptions=_Exceptions))

    def put_item(self, **kwargs):
        if self.item is not None:
            raise self.meta.client.exceptions.ConditionalCheckFailedException()
        self.puts.append(kwargs)
        self.item = kwargs["Item"]

    def get_item(self, **kwargs):
        return {"Item": self.item}

    def update_item(self, **kwargs):
        self.updates.append(kwargs)


def _lex_event(attrs):
    return {"sessionId": "sess-3", "inputTranscript": "pay my bill",
            "sessionState": {"sessionAttributes": attrs}}


TWINS = {
    "stripe_api_base": "https://twin/stripe",
    "stripe_api_key": "sk_test_fixture",
    "fineract_api_base": "https://twin/fineract",
    ATTR_REFERENCE_TIME: "2026-09-10T00:59:09Z",
}


def test_fulfillment_writes_the_record_once_and_forwards_the_turn(monkeypatch):
    table = FakeTable()
    sent = []

    class FakeRuntime:
        def invoke_agent_runtime(self, **kwargs):
            sent.append(kwargs)
            return {"response": _Body(json.dumps({"reply": "Of course.", "turn": {"rounds": []}}).encode())}

    monkeypatch.setattr(init_lambda, "_table", table)
    monkeypatch.setattr(init_lambda, "_runtime", FakeRuntime())
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:runtime")

    first = init_lambda.handler(_lex_event({**TWINS, ATTR_INTERRUPTED_REPLY: "Thanks" + INTERRUPTED_MARKER}), None)
    assert json.loads(table.updates[0]["ExpressionAttributeValues"][":turn"]) == {"rounds": []}
    init_lambda.handler(_lex_event(dict(TWINS)), None)

    assert len(table.puts) == 1
    assert len(table.updates) == 2  # one recorded turn per answered turn
    assert table.item["stripe_api_base"] == "https://twin/stripe"
    assert table.item["reference_time"] == "2026-09-10T00:59:09Z"
    assert sent[0]["runtimeSessionId"] == "sess-3"
    assert json.loads(sent[0]["payload"]) == {
        "text": "pay my bill",
        ATTR_REFERENCE_TIME: "2026-09-10T00:59:09Z",
        ATTR_INTERRUPTED_REPLY: "Thanks" + INTERRUPTED_MARKER,
    }
    assert first["messages"] == [{"contentType": "PlainText", "content": "Of course."}]
    assert ATTR_INTERRUPTED_REPLY not in first["sessionState"]["sessionAttributes"]


def test_fulfillment_stays_silent_on_a_blank_transcript(monkeypatch):
    class NoRuntime:
        def invoke_agent_runtime(self, **kwargs):
            raise AssertionError("a blank turn must not reach the agent")

    monkeypatch.setattr(init_lambda, "_table", FakeTable())
    monkeypatch.setattr(init_lambda, "_runtime", NoRuntime())
    event = _lex_event(dict(TWINS))
    event["inputTranscript"] = "  "

    response = init_lambda.handler(event, None)

    assert "messages" not in response
    assert response["sessionState"]["dialogAction"] == {"type": "ElicitIntent"}


# ------------------------------------------------------------ the tools Lambda


def _gateway_context(tool):
    return SimpleNamespace(client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": f"rory-tools___{tool}"}))


def test_tools_lambda_rebuilds_the_call_from_its_record_and_saves_verification(monkeypatch):
    table = FakeTable({**TWINS, "session_id": "sess-4", "failed_attempts": 1})
    seen = {}

    def fake_dispatch(session, name, args):
        seen["call"] = (name, args, session.account, session.failed_attempts)
        session.account = {"customer_id": "cus_1"}
        session.fineract_client_id = 7
        return {"verified": True}

    monkeypatch.setattr(tools_lambda, "_table", table)
    monkeypatch.setattr(tools_lambda, "dispatch", fake_dispatch)

    result = tools_lambda.handler({"account_number": "4417-88231", SESSION_ARG: "sess-4"}, _gateway_context("verify_caller"))

    assert result == {"verified": True}
    assert seen["call"] == ("verify_caller", {"account_number": "4417-88231"}, None, 1)
    assert tools_lambda.os.environ["STRIPE_API_BASE"] == "https://twin/stripe"
    assert tools_lambda.os.environ["RORY_REFERENCE_TIME"] == "2026-09-10T00:59:09Z"
    update = table.updates[0]
    assert update["Key"] == {"session_id": "sess-4"}
    assert update["ExpressionAttributeValues"] == {
        ":attempts": 1, ":account": json.dumps({"customer_id": "cus_1"}), ":client": 7,
    }
