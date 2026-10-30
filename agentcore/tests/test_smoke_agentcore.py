"""Offline smoke: the pod boots, and a whole call runs through the real pump, reader and worker.

Only the Lex stream, the init Lambda, the agent runtime, Polly and the voice
detector are replaced. The FastAPI lifespan builds the real boto3 clients
(constructing them needs no network), and the call runs the real tasks through
the greeting, Lex's transcripts, the streamed reply spoken sentence by sentence,
and a barge-in reported on the next turn. The call is driven without
TestClient's WebSocket, whose teardown cancels the app task while the call is
still closing.
"""

from __future__ import annotations

import asyncio

import pytest
from aws_sdk_lex_runtime_v2.models import (
    StartConversationResponseEventStreamTranscriptEvent,
    TranscriptEvent,
)
from fastapi.testclient import TestClient
from rory_agentcore import agent
from rory_agentcore.contract import ATTR_INTERRUPTED_REPLY, INTERRUPTED_MARKER
from rory_agentcore.web import app
from starlette.websockets import WebSocketDisconnect

FRAME_BYTES = agent.ACTOR_RATE_HZ * 2 // 50  # 20 ms of PCM16 at 24 kHz
VOICED = b"\x10\x00" * (FRAME_BYTES // 2)
SILENT = bytes(FRAME_BYTES)

POD_ENV = {
    "RORY_AGENTCORE_AWS_ACCESS_KEY_ID": "AKIATEST",
    "RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_REGION": "us-east-1",
    "LEX_BOT_ID": "BOTID",
    "LEX_BOT_ALIAS_ID": "ALIAS",
    "INIT_LAMBDA": "rory-agentcore-init",
    "AGENTCORE_RUNTIME_ARN": "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/rory",
    "STRIPE_API_BASE": "https://twin.example/stripe",
    "STRIPE_API_KEY": "sk_test_fixture",
    "FINERACT_API_BASE": "https://twin.example/fineract",
}


class EnergyVad:
    """Stands in for Silero: a frame is speech when it is not digital silence."""

    def __init__(self, _rate):
        pass

    def push(self, frame: bytes) -> bool:
        return frame != SILENT


@pytest.fixture
def pod_env(monkeypatch):
    for name, value in POD_ENV.items():
        monkeypatch.setenv(name, value)


def test_health_boots_with_the_aws_configuration(pod_env):
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_boot_fails_loudly_without_the_twins(pod_env, monkeypatch):
    monkeypatch.delenv("FINERACT_API_BASE")
    with pytest.raises(RuntimeError, match="FINERACT_API_BASE"), TestClient(app):
        pass


class FakeLexStream:
    """Takes the caller's audio; sends a transcript whenever the actor says a turn ended."""

    def __init__(self):
        self.audio_bytes = 0
        self.transcripts: asyncio.Queue[str] = asyncio.Queue()
        self.input_stream = self

    async def send(self, event):
        self.audio_bytes += len(event.value.audio_chunk)

    async def await_output(self):
        return None, self._events()

    async def _events(self):
        while True:
            text = await self.transcripts.get()
            yield StartConversationResponseEventStreamTranscriptEvent(TranscriptEvent(transcript=text))

    async def close(self):
        pass


class ScriptedActor:
    """Plays the caller's turns, each released once the pod has sent a given amount of audio."""

    client = None

    def __init__(self, lex, steps, done):
        self._lex = lex
        self._steps = [(after, list(frames), text) for after, frames, text in steps]
        self._done = done
        self.sent = 0

    async def receive_bytes(self):
        if not self._steps:
            while not self._done():
                await asyncio.sleep(0.01)
            raise WebSocketDisconnect(1000)
        after, frames, text = self._steps[0]
        while self.sent < after:
            await asyncio.sleep(0.01)
        frame = frames.pop(0)
        if not frames:
            self._steps.pop(0)
            self._lex.transcripts.put_nowait(text)
        return frame

    async def send_bytes(self, data):
        self.sent += len(data)


async def _speech(text):
    """Fake Polly: one word mark per 100 ms, and that much audio."""
    marks, offset = [], 0
    for i, word in enumerate(text.split()):
        start = text.index(word, offset)
        offset = start + len(word)
        marks.append({"time": i * 100, "start": start, "end": offset})
    return marks, bytes(len(marks) * 100 * agent.BYTES_PER_MS)


def _audio_ms(text):
    return len(text.split()) * 100 * agent.BYTES_PER_MS


def test_one_call_greets_answers_and_reports_a_barge_in(pod_env, monkeypatch):
    with TestClient(app):
        pass  # boot once: resolve_vendor reads the twins and builds the clients

    monkeypatch.setattr(agent, "SileroVad", EnergyVad)
    lex = FakeLexStream()

    async def open_lex(session_id):
        lex.session_id = session_id
        return lex

    inits, payloads, reported, sent_at_turn = [], [], [], []
    long_reply = "Of course. " * 40

    def invoke_turn(session_id, payload, emit):
        payloads.append((session_id, payload))
        sent_at_turn.append(actor.sent)
        reply = long_reply if len(payloads) == 1 else "Go ahead."
        emit({"type": "round", "index": 0})
        for word in reply.split(" "):
            emit({"type": "text", "text": word + " "})
        emit({"type": "turn", "turn": {
            "model": "m", "system": ["s"],
            "messages": [{"role": "user", "content": [{"text": payload["text"]}]},
                         {"role": "assistant", "content": [{"text": reply}]}],
            "rounds": [{"start_ns": 1, "end_ns": 2, "output_index": 1, "input_tokens": 1, "output_tokens": 1}],
        }})

    monkeypatch.setattr(agent, "_open_lex", open_lex)
    monkeypatch.setattr(agent, "_init_agent", inits.append)
    monkeypatch.setattr(agent, "_invoke_turn", invoke_turn)
    monkeypatch.setattr(agent, "_synthesize", _speech)
    monkeypatch.setattr(agent.trace, "report_turn",
                        lambda history, turn: reported.append([dict(m) for m in history]) or history.extend(turn["messages"]))

    greeting = _audio_ms(agent.GREETING)
    turn = [SILENT] * 15 + [VOICED] * 25 + [SILENT] * 40
    # The first turn follows the greeting; the second starts 1.5 s into the long reply.
    actor = ScriptedActor(
        lex,
        [(greeting, turn, "I need to pay my bill"), (greeting + 1500 * agent.BYTES_PER_MS, turn, "Hold on")],
        # Hang up once the reply to the second turn has been spoken.
        done=lambda: len(sent_at_turn) == 2 and actor.sent >= sent_at_turn[1] + _audio_ms("Go ahead."),
    )

    asyncio.run(asyncio.wait_for(agent.run_voice_ws_bot(actor), 30))

    assert inits == [lex.session_id]
    # All the caller's audio went to Lex, at 8 kHz: a third of the bytes.
    assert abs(lex.audio_bytes - 2 * len(turn) * FRAME_BYTES / 3) <= 8
    assert [p["text"] for _, p in payloads] == ["I need to pay my bill", "Hold on"]
    assert {s for s, _ in payloads} == {lex.session_id}
    assert ATTR_INTERRUPTED_REPLY not in payloads[0][1]
    interrupted = payloads[1][1][ATTR_INTERRUPTED_REPLY]
    heard = interrupted.removesuffix(INTERRUPTED_MARKER)
    assert interrupted.endswith(INTERRUPTED_MARKER)
    assert heard and long_reply.startswith(heard) and heard != long_reply.strip()
    # The second turn's trace starts from the trimmed reply, as the runtime's history does.
    assert reported[1][-1] == {"role": "assistant", "content": [{"text": interrupted}]}
