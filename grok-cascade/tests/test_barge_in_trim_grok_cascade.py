"""The turn loop: streamed LLM text into TTS, and barge-in trimming what was said.

Each LLM round streams its text straight into an xAI TTS utterance while the
audio plays back. When the caller talks over it the audio stops, the utterance
is cleared on xAI's socket, and the history gets only the words whose audio
had gone out — otherwise the next turn reasons from things the caller never
heard. The LLM round itself always completes, so its tool calls still run.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import pytest

from rory_grok_cascade import agent

REPLY = "one two three four five six seven eight nine ten eleven twelve"
REPLY_MESSAGE = {"role": "assistant", "content": REPLY}
# One chunk per nominal word.
CHUNK_BYTES = agent.TTS_MS_PER_WORD * agent.PCM16_BYTES_PER_MS


class _Actor:
    """The actor socket: records what was sent, and can talk over it.

    ``barge_in_after`` chunks, the VAD callback fires exactly as the audio
    pump would fire it on a voiced frame. The yield before each send is where
    the resulting cancel lands, as it would on the real socket write.
    """

    def __init__(self, barge_in_after: int | None = None):
        self.client = None
        self.sent: list[bytes] = []
        self._barge_in_after = barge_in_after
        self.call: agent._Call | None = None

    async def send_bytes(self, chunk: bytes) -> None:
        await asyncio.sleep(0)
        self.sent.append(chunk)
        if len(self.sent) == self._barge_in_after:
            self.call._on_voice()


def _audio(pcm: bytes) -> str:
    return json.dumps({"type": "audio.delta", "delta": base64.b64encode(pcm).decode()})


class _Tts:
    """xAI's TTS socket: one chunk per word of an utterance, then ``audio.done``.

    ``text.clear`` drops what is queued, but one chunk is already in flight
    and arrives before ``audio.clear`` — the case the drain exists for.
    """

    def __init__(self, chunks: list[bytes] | None = None):
        self.sent: list[dict] = []
        self._chunks = chunks
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._text = ""

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg["type"] == "text.delta":
            self._text += msg["delta"]
        elif msg["type"] == "text.done":
            chunks = self._chunks or [b"\0" * CHUNK_BYTES] * len(self._text.split())
            for pcm in chunks:
                self._inbox.put_nowait(_audio(pcm))
            self._inbox.put_nowait(json.dumps({"type": "audio.done"}))
            self._text = ""
        elif msg["type"] == "text.clear":
            while not self._inbox.empty():
                self._inbox.get_nowait()
            self._inbox.put_nowait(_audio(b"\0" * CHUNK_BYTES))
            self._inbox.put_nowait(json.dumps({"type": "audio.clear"}))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self._inbox.get()

    def types(self) -> list[str]:
        """Message types sent, with each run of text deltas collapsed to one."""
        out: list[str] = []
        for m in self.sent:
            if not (m["type"] == "text.delta" and out and out[-1] == "text.delta"):
                out.append(m["type"])
        return out


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _text(reply: str) -> list:
    """A streamed reply, one token per word as the model would send it."""
    words = reply.split()
    return [_chunk(w if i == 0 else f" {w}") for i, w in enumerate(words)]


def _tool_call(index, *, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class _Stream:
    """``AsyncStream``: an async context manager over the completion's chunks."""

    def __init__(self, chunks: list):
        self._chunks = list(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _wire(monkeypatch, actor: _Actor, tts: _Tts, rounds: list[list] | None = None):
    """Stub the LLM leg and the TTS socket; return the call bound to ``actor``.

    ``rounds`` are the streamed completions handed out in order; by default
    every completion is REPLY.
    """
    # Pacing is real-time sleeps; the trim is about what was sent, not when.
    monkeypatch.setattr(agent, "PLAYBACK_LEAD_S", float("inf"))
    call = agent._Call(actor)
    call._tts = tts
    actor.call = call
    queued = list(rounds or [])

    async def create(**kwargs):
        assert kwargs["stream"] is True
        return _Stream(queued.pop(0) if queued else _text(REPLY))

    monkeypatch.setattr(agent, "_llm", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), raising=False)
    return call


async def _turn(call, text: str) -> None:
    """One caller turn, as the worker runs it."""
    call._replying, call._cut = True, False
    try:
        await call._take_turn(text)
    finally:
        call._replying = False


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_streamed_reply_goes_to_tts_token_by_token(monkeypatch, log_lines):
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts)

    asyncio.run(_turn(call, "hello"))

    assert [m["delta"] for m in tts.sent if m["type"] == "text.delta"] == [
        c.choices[0].delta.content for c in _text(REPLY)
    ]
    assert tts.types() == ["text.delta", "text.done"]
    assert len(actor.sent) == 12
    assert call.messages[-2:] == [{"role": "user", "content": "hello"}, REPLY_MESSAGE]
    assert not [line for line in log_lines if "[voice] barge-in" in line]


def test_barge_in_mid_reply_keeps_the_words_whose_audio_went_out(monkeypatch, log_lines):
    # Twelve words is a nominal 3660 ms; the caller speaks after five chunks,
    # so 1525 ms went out and five words are kept.
    actor, tts = _Actor(barge_in_after=5), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def two_turns():
        await _turn(call, "hello")
        await _turn(call, "wait, what?")

    asyncio.run(two_turns())

    # The chunk in flight at the clear is drained, not played into the next reply.
    assert len(actor.sent) == 5 + 12
    assert tts.types() == ["text.delta", "text.done", "text.clear", "text.delta", "text.done"]
    assert call.messages[-4:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": f"one two three four five{agent.INTERRUPTED_MARKER}"},
        {"role": "user", "content": "wait, what?"},
        REPLY_MESSAGE,
    ]
    assert [line for line in log_lines if "barge-in: trimmed" in line] == [
        '[voice] barge-in: trimmed reply to 1525 ms / 3660 ms ("one two three four five")'
    ]


def test_barge_in_before_any_audio_leaves_no_reply_on_the_history(monkeypatch, log_lines):
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def turn_with_early_barge_in():
        worker = asyncio.create_task(_turn(call, "hello"))
        while not call._replying:
            await asyncio.sleep(0)
        call._on_voice()
        await worker

    asyncio.run(turn_with_early_barge_in())

    assert actor.sent == []
    assert tts.sent == []  # the caller was already talking: no utterance opened
    assert call.messages[-1] == {"role": "user", "content": "hello"}
    assert [line for line in log_lines if "[voice] barge-in" in line] == [
        "[voice] barge-in: dropped reply, nothing sent before the cut"
    ]


def test_barge_in_during_a_tool_round_still_runs_the_tool_and_ends_the_turn(monkeypatch, log_lines):
    # The caller talks over the preamble. The tool the model decided on still
    # runs and is on the history; no further round is spoken.
    round_one = _text("Let me check that for you right now.") + [
        _chunk(tool_calls=[_tool_call(0, id="call_1", name="verify_caller", arguments='{"x": 1}')]),
    ]
    actor, tts = _Actor(barge_in_after=2), _Tts()
    call = _wire(monkeypatch, actor, tts, rounds=[round_one, _text(REPLY)])
    dispatched = []
    monkeypatch.setattr(agent, "dispatch", lambda session, name, args: dispatched.append(name) or {"verified": True})

    asyncio.run(_turn(call, "my account is 447"))

    assert dispatched == ["verify_caller"]
    assert call.messages[-3:] == [
        {"role": "user", "content": "my account is 447"},
        {
            "role": "assistant",
            "content": f"Let me{agent.INTERRUPTED_MARKER}",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "verify_caller", "arguments": '{"x": 1}'}}],
        },
        {"role": "tool", "name": "verify_caller", "tool_call_id": "call_1", "content": '{"verified": true}'},
    ]
    assert tts.types() == ["text.delta", "text.done", "text.clear"]


def test_tool_round_is_assembled_from_fragments_and_its_preamble_spoken(monkeypatch):
    # Round one says a few words, then streams a tool call in pieces; round
    # two answers. Both rounds are spoken, each as its own utterance.
    preamble = "Let me check."
    round_one = _text(preamble) + [
        _chunk(tool_calls=[_tool_call(0, id="call_1", name="get_account", arguments="")]),
        _chunk(tool_calls=[_tool_call(0, arguments='{"account_')]),
        _chunk(tool_calls=[_tool_call(0, arguments='number": "447129"}')]),
    ]
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts, rounds=[round_one, _text(REPLY)])
    dispatched = []

    def dispatch(session, name, args):
        dispatched.append((name, args))
        return {"ok": True}

    monkeypatch.setattr(agent, "dispatch", dispatch)

    asyncio.run(_turn(call, "what do I owe?"))

    assert dispatched == [("get_account", {"account_number": "447129"})]
    assert call.messages[-4:] == [
        {"role": "user", "content": "what do I owe?"},
        {
            "role": "assistant",
            "content": preamble,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_account", "arguments": '{"account_number": "447129"}'},
            }],
        },
        {"role": "tool", "name": "get_account", "tool_call_id": "call_1", "content": '{"ok": true}'},
        REPLY_MESSAGE,
    ]
    assert tts.types() == ["text.delta", "text.done", "text.delta", "text.done"]


def test_tool_only_round_opens_no_utterance(monkeypatch):
    round_one = [_chunk(tool_calls=[_tool_call(0, id="call_1", name="list_bills", arguments="{}")])]
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts, rounds=[round_one, _text(REPLY)])
    monkeypatch.setattr(agent, "dispatch", lambda session, name, args: {"bills": []})

    asyncio.run(_turn(call, "my bills?"))

    assert "content" not in call.messages[-3]
    assert tts.types() == ["text.delta", "text.done"]
    assert call.messages[-1] == REPLY_MESSAGE


def test_odd_length_chunks_reach_the_actor_as_whole_samples(monkeypatch):
    # xAI cuts PCM at arbitrary byte boundaries; the actor must only ever see
    # whole 16-bit samples, in order.
    stream = bytes(range(256)) * 4
    cuts = [0, 3, 10, 11, 600, 1023, 1024]
    chunks = [stream[a:b] for a, b in zip(cuts, cuts[1:])]
    actor, tts = _Actor(), _Tts(chunks)
    call = _wire(monkeypatch, actor, tts)

    asyncio.run(_turn(call, "hello"))

    assert all(len(c) % 2 == 0 for c in actor.sent)
    assert b"".join(actor.sent) == stream


def test_greeting_plays_in_full_over_caller_speech(monkeypatch):
    # Background speech at the start of the call must not silence Rory: the
    # greeting is spoken whole and recorded as said.
    actor, tts = _Actor(barge_in_after=1), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def greet_only():
        worker = asyncio.create_task(call._turn_worker())
        while len(call.messages) < 3:
            await asyncio.sleep(0)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(greet_only())

    assert len(actor.sent) == len(agent.GREETING.split())
    assert tts.types() == ["text.delta", "text.done"]
    assert call.messages[-1] == {"role": "assistant", "content": agent.GREETING}


def test_reply_is_paced_against_playback(monkeypatch):
    # xAI delivers the whole reply at once here; a slice only goes out once
    # playback is within PLAYBACK_LEAD_S of its start, so the last of four
    # 0.5 s slices waits until 0.5 s in.
    one_second = b"\0" * (1000 * agent.PCM16_BYTES_PER_MS)
    actor, tts = _Actor(), _Tts([one_second, one_second])
    call = _wire(monkeypatch, actor, tts)
    monkeypatch.setattr(agent, "PLAYBACK_LEAD_S", 1.0)

    t0 = time.monotonic()
    asyncio.run(_turn(call, "hello"))

    assert time.monotonic() - t0 >= 0.45
    assert [len(c) for c in actor.sent] == [agent.PLAYBACK_CHUNK_BYTES] * 4


class _Stt:
    def __init__(self, events: list[dict]):
        self._events = [json.dumps(e) for e in events]

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _partial(text: str, *, is_final: bool, speech_final: bool, confidence: float) -> dict:
    return {
        "type": "transcript.partial", "text": text, "words": [], "start": 0.0, "duration": 0.0,
        "is_final": is_final, "speech_final": speech_final, "end_of_turn_confidence": confidence,
    }


def test_only_utterance_finals_become_turns():
    # A chunk final Smart Turn held open must not reach the LLM, and the
    # utterance final restates it — accumulating both would double the text.
    call = agent._Call(_Actor())
    call._stt = _Stt([
        _partial("my account number is 4 4 7", is_final=True, speech_final=False, confidence=0.01),
        _partial("my account number is 4 4 7 1 2", is_final=True, speech_final=True, confidence=0.97),
        _partial("", is_final=True, speech_final=True, confidence=0.9),
    ])

    with pytest.raises(RuntimeError, match="closed the stream"):
        asyncio.run(call._pump_stt_events())

    assert call._turns.qsize() == 1
    assert call._turns.get_nowait() == "my account number is 4 4 7 1 2"
