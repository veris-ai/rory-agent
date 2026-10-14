"""Barge-in trims the interrupted reply to the audio the caller heard.

The LLM's reply is on ``messages`` before it is spoken. When the caller talks
over it the Lightning connection is cancelled, and the message must shrink to
the words whose audio had gone out — otherwise the next turn reasons from
things the caller never heard.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from rory_smallest import agent

REPLY = "one two three four five six seven eight nine ten eleven twelve"
# One chunk per nominal word: TTS_MS_PER_WORD of PCM16 at the actor's rate.
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


class _TTSConnection:
    """Lightning's WebSocket: ``n_chunks`` audio chunks, then complete."""

    def __init__(self, n_chunks: int, on_send=None):
        audio = base64.b64encode(bytes(CHUNK_BYTES)).decode()
        self._messages = [
            json.dumps({"status": "chunk", "data": {"audio": audio}}) for _ in range(n_chunks)
        ] + [json.dumps({"status": "complete"})]
        self._on_send = on_send
        self._sent_text = False

    async def send(self, _payload: str) -> None:
        if not self._sent_text:
            self._sent_text = True
            if self._on_send is not None:
                self._on_send()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class _ConnectContext:
    def __init__(self, conn: _TTSConnection):
        self._conn = conn

    async def __aenter__(self) -> _TTSConnection:
        return self._conn

    async def __aexit__(self, *exc_info) -> bool:
        return False


def _wire(monkeypatch, actor: _Actor, *, n_chunks: int, barge_in_during_tts: bool = False):
    """Stub the LLM and Lightning legs; return the call bound to ``actor``."""
    # _speak_now reads SMALLEST_API_KEY per connection (a fresh socket per
    # utterance, unlike the STT socket held for the whole call), so it must be
    # set even though the connection itself is stubbed out below.
    monkeypatch.setenv("SMALLEST_API_KEY", "test-key")
    call = agent._Call(actor)
    actor.call = call
    reply = SimpleNamespace(content=REPLY, tool_calls=None)

    async def complete(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    def connect(_url, **_kwargs):
        on_send = call._on_voice if barge_in_during_tts else None
        return _ConnectContext(_TTSConnection(n_chunks, on_send=on_send))

    monkeypatch.setattr(
        agent,
        "_llm",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
        raising=False,
    )
    monkeypatch.setattr(agent.websockets, "connect", connect)
    return call, reply


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_reply_spoken_in_full_leaves_the_message_unchanged(monkeypatch, log_lines):
    actor = _Actor()
    call, _ = _wire(monkeypatch, actor, n_chunks=12)

    asyncio.run(call._take_turn("hello"))

    assert len(actor.sent) == 12
    assert call.messages[-2] == {"role": "user", "content": "hello"}
    assert call.messages[-1] == {"role": "assistant", "content": REPLY}
    assert not [line for line in log_lines if "[voice] barge-in" in line]


def test_barge_in_mid_reply_keeps_the_words_whose_audio_went_out(monkeypatch, log_lines):
    # Twelve words is a nominal 4200 ms; the caller speaks after five chunks,
    # so 1750 ms went out and five words are kept.
    actor = _Actor(barge_in_after=5)
    call, _ = _wire(monkeypatch, actor, n_chunks=12)

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
    assert call.messages[-1] == {"role": "assistant", "content": REPLY}
    assert [line for line in log_lines if "barge-in: trimmed" in line] == [
        '[voice] barge-in: trimmed reply to 1700 ms / 4080 ms ("one two three four five")'
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


def test_reply_audio_is_paced_so_barge_in_has_something_to_cut(monkeypatch):
    """Lightning delivers the whole reply at once; the actor must not get it at once.

    Twelve chunks (4.2 s of audio) land instantly. A caller who talks 0.2 s in
    must stop the rest: at most PLAYBACK_LEAD_S ahead of playback went out.
    """
    actor = _Actor()
    call, reply = _wire(monkeypatch, actor, n_chunks=12)

    async def scenario():
        call.messages.append({"role": "assistant", "content": REPLY})
        speak = asyncio.create_task(call._speak(REPLY))
        await asyncio.sleep(0.2)
        call._on_voice()
        await speak

    asyncio.run(scenario())
    sent_s = sum(len(c) for c in actor.sent) / agent.PCM16_BYTES_PER_MS / 1000
    assert sent_s <= 0.2 + agent.PLAYBACK_LEAD_S + agent.TTS_MS_PER_WORD / 1000
    assert call.messages[-1]["content"].endswith(agent.INTERRUPTED_MARKER)


def test_lightning_closing_before_complete_fails_the_reply(monkeypatch):
    """A socket that ends without `complete` is a truncated reply, not a spoken one."""
    actor = _Actor()
    call, _ = _wire(monkeypatch, actor, n_chunks=2)
    monkeypatch.setattr(_TTSConnection, "__init__", _truncated_init)

    with pytest.raises(RuntimeError, match="before `complete`"):
        asyncio.run(call._speak_now(REPLY))


def _truncated_init(self, n_chunks, on_send=None):
    audio = base64.b64encode(bytes(CHUNK_BYTES)).decode()
    self._messages = [json.dumps({"status": "chunk", "data": {"audio": audio}})] * n_chunks
    self._on_send = on_send
    self._sent_text = False
