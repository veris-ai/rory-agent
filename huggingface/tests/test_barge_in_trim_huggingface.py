"""Barge-in trims the interrupted reply to the audio the caller heard.

The LLM's reply is on ``messages`` before it is spoken. When the caller talks
over it the pacer is cancelled, and the message must shrink to the words whose
audio had gone out — otherwise the next turn reasons from things the caller
never heard.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from rory_huggingface import agent

REPLY = "one two three four five six seven eight nine ten eleven twelve"
CHUNK_BYTES = int(agent.PLAYBACK_CHUNK_S * agent.ACTOR_RATE_HZ) * 2  # 500 ms


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


def _wire(monkeypatch, actor: _Actor, *, n_chunks: int, barge_in_during_tts: bool = False):
    """Stub the SDK model leg and the TTS leg; return the call bound to ``actor``."""
    call = agent._Call(actor)
    actor.call = call
    wav = agent._wav_bytes(bytes(CHUNK_BYTES * n_chunks), agent.ACTOR_RATE_HZ)

    async def create(**kwargs):
        return ChatCompletion.model_validate({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": agent.LLM_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": REPLY}}],
        })

    # _llm is only annotated at import; connect_legs binds it at boot.
    monkeypatch.setattr(
        agent, "_llm",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        raising=False,
    )

    async def post(url, *, label, **kwargs):
        assert label == "tts"
        if barge_in_during_tts:
            call._on_voice()
            await asyncio.sleep(0)
        return SimpleNamespace(json=lambda: {"audio_b64": base64.b64encode(wav).decode()})

    monkeypatch.setattr(agent, "_post_with_retry", post)
    monkeypatch.setattr(agent, "TTS_URL_OVERRIDE", "https://tts.test")
    monkeypatch.setattr(agent, "_tts_url", "https://tts.test", raising=False)
    return call


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_reply_spoken_in_full_leaves_the_message_unchanged(monkeypatch, log_lines):
    actor = _Actor()
    call = _wire(monkeypatch, actor, n_chunks=2)

    asyncio.run(call._take_turn("hello"))

    assert len(actor.sent) == 2
    assert call.messages[-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": REPLY},
    ]
    assert not [line for line in log_lines if "[voice] barge-in" in line]


def test_barge_in_mid_reply_keeps_the_words_whose_audio_went_out(monkeypatch, log_lines):
    # Four chunks is 2000 ms; the caller speaks after two, so 1000 ms of the
    # twelve words went out and six are kept.
    actor = _Actor(barge_in_after=2)
    call = _wire(monkeypatch, actor, n_chunks=4)

    async def two_turns():
        await call._take_turn("hello")
        await call._take_turn("wait, what?")

    asyncio.run(two_turns())

    assert len(actor.sent) == 2 + 4
    assert call.messages[-4:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": f"one two three four five six{agent.INTERRUPTED_MARKER}"},
        {"role": "user", "content": "wait, what?"},
        {"role": "assistant", "content": REPLY},
    ]
    assert [line for line in log_lines if "barge-in: trimmed" in line] == [
        '[voice] barge-in: trimmed reply to 1000 ms / 2000 ms ("one two three four five six")'
    ]


def test_barge_in_before_any_audio_drops_the_message(monkeypatch, log_lines):
    actor = _Actor()
    call = _wire(monkeypatch, actor, n_chunks=2, barge_in_during_tts=True)

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
    call = _wire(monkeypatch, actor, n_chunks=4)

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
