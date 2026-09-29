"""Barge-in trims the interrupted reply to the audio the caller heard.

The LLM's reply is on ``messages`` before it is spoken. When the caller talks
over it the TTS stream is cancelled, and the message must shrink to the words
whose audio had gone out — otherwise the next turn reasons from things the
caller never heard.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest
from mistralai.client.models import AssistantMessage

from rory_mistral import agent

REPLY = "one two three four five six seven eight nine ten eleven twelve"
# One chunk per nominal word: 400 ms of float32 at 24 kHz.
CHUNK_SAMPLES = agent.TTS_MS_PER_WORD * agent.ACTOR_RATE_HZ // 1000


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


class _Stream:
    """Voxtral TTS: ``n_chunks`` audio deltas, then done."""

    def __init__(self, n_chunks: int):
        delta = base64.b64encode(np.zeros(CHUNK_SAMPLES, dtype="<f4").tobytes()).decode()
        self._events = [
            SimpleNamespace(data=SimpleNamespace(type="speech.audio.delta", audio_data=delta))
            for _ in range(n_chunks)
        ] + [SimpleNamespace(data=SimpleNamespace(type="speech.audio.done"))]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _wire(monkeypatch, actor: _Actor, *, n_chunks: int, barge_in_during_tts: bool = False):
    """Stub the LLM and TTS legs; return the call bound to ``actor``."""
    call = agent._Call(actor)
    actor.call = call
    reply = AssistantMessage(content=REPLY)

    async def complete_chat(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    async def complete_speech(**kwargs):
        if barge_in_during_tts:
            call._on_voice()
            await asyncio.sleep(0)
        return _Stream(n_chunks)

    monkeypatch.setattr(agent, "_client", SimpleNamespace(
        chat=SimpleNamespace(complete_async=complete_chat),
        audio=SimpleNamespace(speech=SimpleNamespace(complete_async=complete_speech)),
    ), raising=False)
    monkeypatch.setattr(agent, "_voice_id", "voice-test", raising=False)
    return call, reply


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_reply_spoken_in_full_leaves_the_message_unchanged(monkeypatch, log_lines):
    actor = _Actor()
    call, reply = _wire(monkeypatch, actor, n_chunks=12)

    asyncio.run(call._take_turn("hello"))

    assert len(actor.sent) == 12
    assert call.messages[-2] == {"role": "user", "content": "hello"}
    assert call.messages[-1] is reply
    assert not [line for line in log_lines if "[voice] barge-in" in line]


def test_barge_in_mid_reply_keeps_the_words_whose_audio_went_out(monkeypatch, log_lines):
    # Twelve words is a nominal 4800 ms; the caller speaks after five chunks,
    # so 2000 ms went out and five words are kept.
    actor = _Actor(barge_in_after=5)
    call, reply = _wire(monkeypatch, actor, n_chunks=12)

    async def two_turns():
        await call._take_turn("hello")
        await call._take_turn("wait, what?")

    asyncio.run(two_turns())

    assert len(actor.sent) == 5 + 12
    assert call.messages[-4:-1] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": f"one two three four five{agent.INTERRUPTED_MARKER}"},
        {"role": "user", "content": "wait, what?"},
    ]
    assert call.messages[-1] is reply
    assert [line for line in log_lines if "barge-in: trimmed" in line] == [
        '[voice] barge-in: trimmed reply to 2000 ms / 4800 ms ("one two three four five")'
    ]


def test_barge_in_before_any_audio_drops_the_message(monkeypatch, log_lines):
    actor = _Actor()
    call, _ = _wire(monkeypatch, actor, n_chunks=12, barge_in_during_tts=True)

    asyncio.run(call._take_turn("hello"))

    assert actor.sent == []
    assert call.messages[-1] == {"role": "user", "content": "hello"}
    assert [line for line in log_lines if "[voice] barge-in" in line] == [
        "[voice] barge-in: dropped reply, nothing sent before the cut"
    ]


def test_greeting_is_on_the_history_before_it_is_spoken(monkeypatch):
    # A caller talking over the greeting trims it like any reply, which needs
    # it appended first.
    actor = _Actor(barge_in_after=1)
    call, _ = _wire(monkeypatch, actor, n_chunks=12)

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
