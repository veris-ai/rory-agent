"""Barge-in trims the interrupted reply to the audio the caller heard.

The LLM's reply is on ``messages`` before it is spoken. When the caller talks
over it the TTS read is cancelled, the utterance is cleared on xAI's socket,
and the message must shrink to the words whose audio had gone out —
otherwise the next turn reasons from things the caller never heard.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage

from rory_grok_cascade import agent

REPLY = "one two three four five six seven eight nine ten eleven twelve"
WORDS = REPLY.split()
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


def _delta(pcm: bytes) -> str:
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
            words = self._text.split()
            chunks = self._chunks or [b"\0" * CHUNK_BYTES] * len(words)
            for pcm in chunks:
                self._inbox.put_nowait(_delta(pcm))
            self._inbox.put_nowait(json.dumps({"type": "audio.done"}))
            self._text = ""
        elif msg["type"] == "text.clear":
            while not self._inbox.empty():
                self._inbox.get_nowait()
            self._inbox.put_nowait(_delta(b"\0" * CHUNK_BYTES))
            self._inbox.put_nowait(json.dumps({"type": "audio.clear"}))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self._inbox.get()

    def types(self) -> list[str]:
        return [m["type"] for m in self.sent]


def _wire(monkeypatch, actor: _Actor, tts: _Tts):
    """Stub the LLM leg and the TTS socket; return the call bound to ``actor``."""
    # Pacing is real-time sleeps; the trim is about what was sent, not when.
    monkeypatch.setattr(agent, "PLAYBACK_LEAD_S", float("inf"))
    call = agent._Call(actor)
    call._tts = tts
    actor.call = call

    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=ChatCompletionMessage(role="assistant", content=REPLY)
        )])

    monkeypatch.setattr(agent, "_llm", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), raising=False)
    return call


REPLY_MESSAGE = {"role": "assistant", "content": REPLY}


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_reply_spoken_in_full_leaves_the_message_unchanged(monkeypatch, log_lines):
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts)

    asyncio.run(call._take_turn("hello"))

    assert len(actor.sent) == 12
    assert call.messages[-2:] == [{"role": "user", "content": "hello"}, REPLY_MESSAGE]
    assert tts.types() == ["text.delta", "text.done"]
    assert not [line for line in log_lines if "[voice] barge-in" in line]


def test_barge_in_mid_reply_keeps_the_words_whose_audio_went_out(monkeypatch, log_lines):
    # Twelve words is a nominal 3960 ms; the caller speaks after five chunks,
    # so 1650 ms went out and five words are kept.
    actor, tts = _Actor(barge_in_after=5), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def two_turns():
        await call._take_turn("hello")
        await call._take_turn("wait, what?")

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
        '[voice] barge-in: trimmed reply to 1650 ms / 3960 ms ("one two three four five")'
    ]


def test_barge_in_before_any_audio_drops_the_message(monkeypatch, log_lines):
    actor, tts = _Actor(), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def turn_with_early_barge_in():
        worker = asyncio.create_task(call._take_turn("hello"))
        while call._speaking is None:
            await asyncio.sleep(0)
        call._on_voice()
        await worker

    asyncio.run(turn_with_early_barge_in())

    assert actor.sent == []
    # Nothing was sent to xAI yet; the clear is still sent, and acknowledged.
    assert tts.types() == ["text.clear"]
    assert call.messages[-1] == {"role": "user", "content": "hello"}
    assert [line for line in log_lines if "[voice] barge-in" in line] == [
        "[voice] barge-in: dropped reply, nothing sent before the cut"
    ]


def test_odd_length_chunks_reach_the_actor_as_whole_samples(monkeypatch):
    # xAI cuts PCM at arbitrary byte boundaries; the actor must only ever see
    # whole 16-bit samples, in order.
    stream = bytes(range(256)) * 4
    cuts = [0, 3, 10, 11, 600, 1023, 1024]
    chunks = [stream[a:b] for a, b in zip(cuts, cuts[1:])]
    actor, tts = _Actor(), _Tts(chunks)
    call = _wire(monkeypatch, actor, tts)

    asyncio.run(call._take_turn("hello"))

    assert all(len(c) % 2 == 0 for c in actor.sent)
    assert b"".join(actor.sent) == stream


def test_greeting_is_on_the_history_before_it_is_spoken(monkeypatch):
    # A caller talking over the greeting trims it like any reply, which needs
    # it appended first.
    actor, tts = _Actor(barge_in_after=1), _Tts()
    call = _wire(monkeypatch, actor, tts)

    async def greet_only():
        worker = asyncio.create_task(call._turn_worker())
        while not actor.sent or call._speaking is not None:
            await asyncio.sleep(0)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(greet_only())

    assert call.messages[-1]["role"] == "assistant"
    assert call.messages[-1]["content"].endswith(agent.INTERRUPTED_MARKER)
    assert call.messages[-1]["content"] != agent.GREETING


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


def test_reply_is_paced_against_playback(monkeypatch):
    # xAI delivers the whole reply at once here; a slice only goes out once
    # playback is within PLAYBACK_LEAD_S of its start, so the last of four
    # 0.5 s slices waits until 0.5 s in.
    one_second = b"\0" * (1000 * agent.PCM16_BYTES_PER_MS)
    actor, tts = _Actor(), _Tts([one_second, one_second])
    call = _wire(monkeypatch, actor, tts)
    monkeypatch.setattr(agent, "PLAYBACK_LEAD_S", 1.0)

    t0 = time.monotonic()
    asyncio.run(call._take_turn("hello"))

    assert time.monotonic() - t0 >= 0.45
    assert [len(c) for c in actor.sent] == [agent.PLAYBACK_CHUNK_BYTES] * 4
