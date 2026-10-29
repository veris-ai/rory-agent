"""Each piece of the candidate against fakes of the services around it."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from rory_agentcore import agent, init_lambda, runtime, tools_lambda
from rory_agentcore.contract import (
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    CALL_CONNECTED,
    INTERRUPTED_MARKER,
    SESSION_ARG,
)

from rory_tools.prompt import GREETING


class _Body:
    def __init__(self, data: bytes = b""):
        self._data = data

    def read(self):
        return self._data


# ----------------------------------------------------------------------- the pod


def _piece(round_, start, text, marks, sent_ms):
    piece = agent._Piece(round_, start, text, synth=None)
    piece.marks = marks
    piece.sent_bytes = sent_ms * agent.BYTES_PER_MS
    return piece


def test_barge_in_keeps_only_the_words_that_started_before_the_cut():
    marks = [{"time": 0, "start": 0, "end": 5}, {"time": 400, "start": 6, "end": 10},
             {"time": 700, "start": 11, "end": 13}, {"time": 900, "start": 14, "end": 17}]
    assert _piece(0, 0, "Sure, here it is.", marks, 500).heard() == "Sure, here"


def test_polly_offsets_are_bytes_so_a_dash_does_not_shift_the_cut():
    text = "Rory — how can I help?"
    # "—" is three bytes in UTF-8: "how" starts at byte 9, character 7.
    marks = [{"time": 0, "start": 0, "end": 4}, {"time": 300, "start": 9, "end": 12},
             {"time": 600, "start": 13, "end": 16}]
    assert _piece(0, 0, text, marks, 400).heard() == "Rory — how"


def test_heard_reply_is_the_last_round_up_to_the_cut():
    rounds = ["Let me check. ", "Your balance is 42 dollars. Anything else?"]
    played = [
        _piece(0, 0, "Let me check. ", [], 2000),
        _piece(1, 0, "Your balance is 42 dollars. ", [], 2000),
        _piece(1, 28, "Anything else?", [{"time": 0, "start": 0, "end": 8}, {"time": 500, "start": 9, "end": 14}], 100),
        _piece(1, 42, "unsent", [], 0),
    ]
    assert agent._Call._heard_reply(played, rounds) == "Your balance is 42 dollars. Anything"


def test_cut_during_a_tool_preamble_hears_none_of_the_reply():
    rounds = ["Let me check. ", "Your balance is 42 dollars."]
    played = [_piece(0, 0, "Let me check. ", [{"time": 0, "start": 0, "end": 3}], 200)]
    assert agent._Call._heard_reply(played, rounds) == ""


def test_runtime_events_are_read_as_they_arrive_across_chunk_boundaries(monkeypatch):
    wire = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in (
        {"type": "round", "index": 0}, {"type": "text", "text": "Hi."}, {"type": "turn", "turn": {}},
    ))
    chunks = [wire[i:i + 7] for i in range(0, len(wire), 7)]

    class Raw:
        def read1(self, _n):
            return chunks.pop(0) if chunks else b""

    sent = {}

    class FakeAgentCore:
        def invoke_agent_runtime(self, **kwargs):
            sent.update(kwargs)
            return {"response": SimpleNamespace(_raw_stream=Raw())}

    monkeypatch.setattr(agent, "_agentcore", FakeAgentCore(), raising=False)
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:runtime")
    events = []

    agent._invoke_turn("s1", {"text": "hello"}, events.append)

    assert [e["type"] for e in events] == ["round", "text", "turn"]
    assert sent["runtimeSessionId"] == "s1"
    assert sent["accept"] == "text/event-stream"
    assert json.loads(sent["payload"]) == {"text": "hello"}


def test_init_sends_the_twins_and_fails_loudly(monkeypatch):
    calls = []

    class FakeLambda:
        def __init__(self, error):
            self.error = error

        def invoke(self, **kwargs):
            calls.append(json.loads(kwargs["Payload"]))
            resp = {"Payload": _Body(b'{"errorMessage": "boom"}' if self.error else b'{"ready": true}')}
            return {**resp, "FunctionError": "Unhandled"} if self.error else resp

    monkeypatch.setenv("INIT_LAMBDA", "rory-agentcore-init")
    monkeypatch.setattr(agent, "_twins", {"stripe_api_base": "https://twin/stripe"}, raising=False)
    monkeypatch.setattr(agent, "_lambda", FakeLambda(error=False), raising=False)
    agent._init_agent("s1")
    assert calls == [{"session_id": "s1", "stripe_api_base": "https://twin/stripe"}]

    monkeypatch.setattr(agent, "_lambda", FakeLambda(error=True), raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        agent._init_agent("s1")


def test_audio_is_resampled_to_the_rate_each_leg_speaks():
    one_second_at_24k = bytes(agent.BYTES_PER_S)
    assert len(agent._to_rate(one_second_at_24k, agent.ACTOR_RATE_HZ, agent.LEX_RATE_HZ)) == 8000 * 2
    # ratecv rounds the sample count, so the length can be a sample short.
    one_second_at_16k = bytes(16000 * 2)
    assert abs(len(agent._to_rate(one_second_at_16k, agent.POLLY_RATE_HZ, agent.ACTOR_RATE_HZ)) - agent.BYTES_PER_S) <= 4


# ------------------------------------------------------------------- the runtime


def _stream(content):
    """A Converse message as the event stream ConverseStream sends it."""
    events = [{"messageStart": {"role": "assistant"}}]
    for i, block in enumerate(content):
        if "text" in block:
            for word in block["text"].split(" "):
                events.append({"contentBlockDelta": {"contentBlockIndex": i, "delta": {"text": word + " "}}})
        else:
            use = block["toolUse"]
            events.append({"contentBlockStart": {"contentBlockIndex": i, "start": {
                "toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}}}})
            events.append({"contentBlockDelta": {"contentBlockIndex": i, "delta": {
                "toolUse": {"input": json.dumps(use["input"])}}}})
        events.append({"contentBlockStop": {"contentBlockIndex": i}})
    return events


class FakeBedrock:
    """Answers each converse_stream call with the next scripted (stopReason, content) round."""

    def __init__(self, rounds):
        self._rounds = iter(rounds)
        self.calls = []

    def converse_stream(self, **kwargs):
        self.calls.append(kwargs)
        stop, content = next(self._rounds)
        return {"stream": [
            *_stream(content),
            {"messageStop": {"stopReason": stop}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 90}}},
        ]}

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {}


def test_new_session_seeds_the_greeting_the_pod_speaks():
    session = runtime._new_session(None)
    assert session["system"][0] == {"text": runtime.AGENT_PROMPT}
    assert session["system"][-1] == {"text": runtime.VOICE_NOTE}
    assert session["messages"] == [
        {"role": "user", "content": [{"text": CALL_CONNECTED}]},
        {"role": "assistant", "content": [{"text": GREETING}]},
    ]


def test_a_turn_streams_its_words_and_calls_tools_with_the_session(monkeypatch):
    bedrock = FakeBedrock([
        ("tool_use", [{"text": "Let me check."},
                      {"toolUse": {"toolUseId": "t1", "name": "get_balance", "input": {"x": 1}}}]),
        ("end_turn", [{"text": "Your balance is 42 dollars."}]),
    ])
    calls = []
    monkeypatch.setattr(runtime, "_bedrock", bedrock)
    monkeypatch.setattr(runtime.gateway, "call_tool", lambda name, args: calls.append((name, args)) or {"balance": 42})
    session = runtime._new_session(None)

    events = list(runtime.take_turn(session, "sess-1", "what do I owe"))

    assert [e["type"] for e in events if e["type"] != "text"] == ["round", "tool", "round", "turn"]
    assert "".join(e["text"] for e in events if e["type"] == "text").split() == \
        ["Let", "me", "check.", "Your", "balance", "is", "42", "dollars."]
    turn = events[-1]["turn"]
    # The turn the pod reports: the caller's words, the tool call, its result, the reply.
    assert [m["role"] for m in turn["messages"]] == ["user", "assistant", "user", "assistant"]
    assert turn["messages"][1]["content"][1] == {
        "toolUse": {"toolUseId": "t1", "name": "get_balance", "input": {"x": 1}}}
    assert [r["output_index"] for r in turn["rounds"]] == [1, 3]
    assert turn["rounds"][0]["input_tokens"] == 100  # cached tokens count as input
    assert turn["system"] == [block["text"] for block in session["system"]]
    assert calls == [("get_balance", {"x": 1, SESSION_ARG: "sess-1"})]
    assert session["messages"][-2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"balance": 42}}]}}],
    }


def test_requests_cache_the_prefix_but_the_history_stays_clean(monkeypatch):
    bedrock = FakeBedrock([("end_turn", [{"text": "Sure."}])])
    monkeypatch.setattr(runtime, "_bedrock", bedrock)
    session = runtime._new_session(None)

    list(runtime.take_turn(session, "sess-1", "hello"))

    request = bedrock.calls[0]
    assert request["system"][-1] == runtime.CACHE_POINT
    assert request["messages"][-1]["content"][-1] == runtime.CACHE_POINT
    assert request["toolConfig"] == {"tools": runtime.CONVERSE_TOOLS}
    assert not any("cachePoint" in block for m in session["messages"] for block in m["content"])


def test_tool_loop_that_never_settles_fails_loudly(monkeypatch):
    looping = ("tool_use", [{"toolUse": {"toolUseId": "t", "name": "get_balance", "input": {}}}])
    monkeypatch.setattr(runtime, "_bedrock", FakeBedrock([looping] * runtime.MAX_TOOL_ROUNDS))
    monkeypatch.setattr(runtime.gateway, "call_tool", lambda name, args: {})

    with pytest.raises(RuntimeError, match="did not settle"):
        list(runtime.take_turn(runtime._new_session(None), "sess-1", "hi"))


def test_a_failed_turn_leaves_the_history_as_it_was(monkeypatch):
    class Refusing:
        def converse_stream(self, **kwargs):
            raise RuntimeError("ValidationException")

    monkeypatch.setattr(runtime, "_bedrock", Refusing())
    session = runtime._new_session(None)
    before = [dict(m) for m in session["messages"]]

    with pytest.raises(RuntimeError):
        list(runtime.take_turn(session, "sess-1", "hello"))

    assert session["messages"] == before


async def _drain(response) -> list[dict]:
    body = b"".join([chunk.encode() if isinstance(chunk, str) else chunk async for chunk in response.body_iterator])
    return [json.loads(e.removeprefix(b"data: ")) for e in body.split(b"\n\n") if e]


def test_init_primes_the_cache_and_a_barge_in_rewrites_the_last_reply(monkeypatch):
    bedrock = FakeBedrock([("end_turn", [{"text": "Sure."}])])
    monkeypatch.setattr(runtime, "_bedrock", bedrock)
    runtime._sessions.clear()
    heard = "Thanks for calling" + INTERRUPTED_MARKER

    assert runtime.invocations({"init": True, ATTR_REFERENCE_TIME: "2026-09-10T00:59:09Z"}, "sess-2") == {"ready": True}
    assert bedrock.calls[0]["inferenceConfig"] == {"maxTokens": 1}
    events = asyncio.run(_drain(runtime.invocations({"text": "I need help", ATTR_INTERRUPTED_REPLY: heard}, "sess-2")))

    assert events[-1]["type"] == "turn"
    messages = runtime._sessions["sess-2"]["messages"]
    assert messages[1] == {"role": "assistant", "content": [{"text": heard}]}
    assert messages[-1]["content"] == [{"text": "Sure. "}]


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

    def put_item(self, **kwargs):
        self.puts.append(kwargs)
        self.item = kwargs["Item"]

    def get_item(self, **kwargs):
        return {"Item": self.item}

    def update_item(self, **kwargs):
        self.updates.append(kwargs)


TWINS = {
    "stripe_api_base": "https://twin/stripe",
    "stripe_api_key": "sk_test_fixture",
    "fineract_api_base": "https://twin/fineract",
    ATTR_REFERENCE_TIME: "2026-09-10T00:59:09Z",
}


def test_init_writes_the_record_and_warms_the_agent_session(monkeypatch):
    table = FakeTable()
    sent = []

    class FakeRuntime:
        def invoke_agent_runtime(self, **kwargs):
            sent.append(kwargs)
            return {"response": _Body(b'{"ready": true}')}

    monkeypatch.setattr(init_lambda, "_table", table)
    monkeypatch.setattr(init_lambda, "_runtime", FakeRuntime())
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:runtime")

    assert init_lambda.handler({"session_id": "sess-3", **TWINS}, None) == {"ready": True}

    assert table.puts[0]["ConditionExpression"] == "attribute_not_exists(session_id)"
    assert table.item["stripe_api_base"] == "https://twin/stripe"
    assert table.item["failed_attempts"] == 0
    assert table.item["reference_time"] == "2026-09-10T00:59:09Z"
    assert sent[0]["runtimeSessionId"] == "sess-3"
    assert json.loads(sent[0]["payload"]) == {"init": True, ATTR_REFERENCE_TIME: "2026-09-10T00:59:09Z"}


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


def test_tools_lambda_keeps_one_fineract_client_per_twin(monkeypatch):
    built = []

    class CountingFineract:
        def __init__(self):
            built.append(tools_lambda.os.environ["FINERACT_API_BASE"])

    monkeypatch.setattr(tools_lambda, "FineractClient", CountingFineract)
    monkeypatch.setattr(tools_lambda, "_fineract", {})
    monkeypatch.setattr(tools_lambda, "dispatch", lambda session, name, args: {})
    for base in ("https://twin-a/fineract", "https://twin-a/fineract", "https://twin-b/fineract"):
        monkeypatch.setattr(tools_lambda, "_table", FakeTable(
            {**TWINS, "fineract_api_base": base, "session_id": "s", "failed_attempts": 0}))
        tools_lambda.handler({SESSION_ARG: "s"}, _gateway_context("get_account"))

    assert built == ["https://twin-a/fineract", "https://twin-b/fineract"]


def test_words_before_a_tool_call_are_spoken_while_the_tool_runs(monkeypatch):
    import threading

    spoken = []
    preamble_spoken = threading.Event()

    async def synthesize(text):
        spoken.append(text)
        if text == "Let me check.":
            preamble_spoken.set()
        return [], bytes(agent.BYTES_PER_MS)

    def invoke_turn(session_id, payload, emit):
        emit({"type": "round", "index": 0})
        emit({"type": "text", "text": "Let me check."})
        emit({"type": "tool", "name": "get_account"})
        # The tool "runs" until the preamble has been spoken.
        assert preamble_spoken.wait(5), "the preamble waited for the tool"
        emit({"type": "round", "index": 1})
        emit({"type": "text", "text": "You owe 42 dollars."})
        emit({"type": "turn", "turn": {"messages": [], "rounds": []}})

    class Actor:
        client = None

        async def send_bytes(self, data):
            pass

    monkeypatch.setattr(agent, "_synthesize", synthesize)
    monkeypatch.setattr(agent, "_invoke_turn", invoke_turn)
    monkeypatch.setattr(agent.trace, "report_turn", lambda history, turn: None)
    monkeypatch.setattr(agent, "SileroVad", lambda rate: None)

    asyncio.run(agent._Call(Actor())._take_turn("what do I owe"))

    assert spoken == ["Let me check.", "You owe 42 dollars."]


def test_the_lex_stream_is_renewed_in_a_pause_and_turns_keep_flowing(monkeypatch):
    from aws_sdk_lex_runtime_v2.models import (
        StartConversationResponseEventStreamTranscriptEvent,
        TranscriptEvent,
    )

    opened = []

    class Stream:
        def __init__(self):
            self.ended = asyncio.Event()
            self.closed = False
            self.input_stream = self

        async def send(self, event):
            pass

        async def await_output(self):
            return None, self._events()

        async def _events(self):
            # The second stream carries the caller's turn; every stream ends when closed.
            if len(opened) == 2 and self is opened[1]:
                yield StartConversationResponseEventStreamTranscriptEvent(TranscriptEvent(transcript="still there?"))
            await self.ended.wait()

        async def close(self):
            self.closed = True
            self.ended.set()

    async def open_lex(session_id):
        opened.append(Stream())
        return opened[-1]

    turns = []

    class Actor:
        client = None

        async def receive_bytes(self):
            await asyncio.sleep(0.02)
            if turns:
                raise agent.WebSocketDisconnect(1000)
            return bytes(agent.ACTOR_RATE_HZ * 2 // 50)

        async def send_bytes(self, data):
            pass

    async def take_turn(self, text):
        turns.append(text)

    monkeypatch.setattr(agent, "LEX_STREAM_MAX_S", 0.1)
    monkeypatch.setattr(agent, "_open_lex", open_lex)
    monkeypatch.setattr(agent, "_init_agent", lambda session_id: None)
    monkeypatch.setattr(agent, "_synthesize", lambda text: asyncio.sleep(0, ([], b"")))
    monkeypatch.setattr(agent, "SileroVad", lambda rate: SimpleNamespace(push=lambda frame: False))
    monkeypatch.setattr(agent._Call, "_take_turn", take_turn)

    asyncio.run(asyncio.wait_for(agent._Call(Actor()).run(), 10))

    assert turns == ["still there?"]
    assert len(opened) >= 2
    assert opened[0].closed


def test_lex_hook_keeps_the_dialog_open_and_the_attributes():
    from rory_agentcore import lex_hook

    attrs = {"x-amz-lex:audio:end-timeout-ms:*:*": "400"}
    event = {"sessionState": {"sessionAttributes": attrs, "intent": {"name": "FallbackIntent"}}}

    assert lex_hook.handler(event, None) == {
        "sessionState": {"dialogAction": {"type": "ElicitIntent"}, "sessionAttributes": attrs},
        "messages": [{"contentType": "PlainText", "content": lex_hook.PLACEHOLDER}],
    }


def test_each_lex_response_is_acknowledged_so_lex_takes_the_next_turn():
    from aws_sdk_lex_runtime_v2.models import (
        StartConversationRequestEventStreamPlaybackCompletionEvent,
        StartConversationResponseEventStreamTextResponseEvent,
        StartConversationResponseEventStreamTranscriptEvent,
        TextResponseEvent,
        TranscriptEvent,
    )

    sent = []

    class Stream:
        input_stream = SimpleNamespace(send=None)

        async def await_output(self):
            async def events():
                yield StartConversationResponseEventStreamTranscriptEvent(TranscriptEvent(transcript="hi"))
                yield StartConversationResponseEventStreamTextResponseEvent(TextResponseEvent(messages=[]))
            return None, events()

    async def send(event):
        sent.append(event)

    stream = Stream()
    stream.input_stream.send = send
    call = agent._Call(SimpleNamespace(client=None))
    call._lex = stream
    call._lex_ready.set()

    with pytest.raises(RuntimeError, match="closed the conversation stream"):
        asyncio.run(call._read_lex())

    assert call._transcripts.get_nowait() == "hi"
    assert [type(e) for e in sent] == [StartConversationRequestEventStreamPlaybackCompletionEvent]
