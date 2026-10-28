"""Offline smoke: the pod boots, and a whole call runs through the real pump and worker.

Only Lex (and the agent behind it), Polly and the voice detector are replaced. The
FastAPI lifespan builds the real boto3 clients (constructing them needs no
network), and the call runs the real pump and worker through greeting,
endpointing, the Lex turn, the spoken reply and a barge-in reported on the next
turn. The call is driven without TestClient's WebSocket, whose teardown cancels
the app task while the call is still closing.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from rory_agentcore import agent
from rory_agentcore.contract import (
    ATTR_INTERRUPTED_REPLY,
    ATTR_STRIPE_API_BASE,
    INTERRUPTED_MARKER,
)
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
    "STRIPE_API_BASE": "https://twin.example/stripe",
    "STRIPE_API_KEY": "sk_test_fixture",
    "FINERACT_API_BASE": "https://twin.example/fineract",
    "SESSIONS_TABLE": "rory-agentcore-sessions-test",
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


class ScriptedActor:
    """Plays the caller's turns, each released once the pod has sent a given amount of audio."""

    client = None

    def __init__(self, steps, final_bytes):
        self._steps = [(after, list(frames)) for after, frames in steps]
        self._final_bytes = final_bytes
        self.sent = 0

    async def receive_bytes(self):
        while self._steps and not self._steps[0][1]:
            self._steps.pop(0)
        if not self._steps:
            while self.sent < self._final_bytes:
                await asyncio.sleep(0.01)
            raise WebSocketDisconnect(1000)
        after, frames = self._steps[0]
        while self.sent < after:
            await asyncio.sleep(0.01)
        return frames.pop(0)

    async def send_bytes(self, data):
        self.sent += len(data)


def _speech(text):
    """Fake Polly: one word mark per 100 ms, and that much audio."""
    marks, offset = [], 0
    for i, word in enumerate(text.split()):
        start = text.index(word, offset)
        offset = start + len(word)
        marks.append({"time": i * 100, "start": start, "end": offset})
    return marks, bytes(len(marks) * 100 * agent.BYTES_PER_MS)


def test_one_call_greets_answers_and_reports_a_barge_in(pod_env, monkeypatch):
    with TestClient(app):
        pass  # boot once: resolve_vendor reads the twins and builds the clients

    monkeypatch.setattr(agent, "SileroVad", EnergyVad)
    long_reply = "Of course. " * 40
    turns = []

    def fake_recognize(session_id, pcm16_16k, attrs):
        turns.append((session_id, len(pcm16_16k), dict(attrs)))
        return "I need to pay my bill", long_reply.strip() if len(turns) == 1 else "Go ahead."

    def fake_read_turn(session_id):
        reply = long_reply.strip() if len(turns) == 1 else "Go ahead."
        return len(turns), {
            "model": "m", "system": ["s"],
            "messages": [{"role": "user", "content": [{"text": "I need to pay my bill"}]},
                         {"role": "assistant", "content": [{"text": reply}]}],
            "rounds": [{"start_ns": 1, "end_ns": 2, "output_index": 1, "input_tokens": 1, "output_tokens": 1}],
        }

    reported = []
    monkeypatch.setattr(agent, "_recognize", fake_recognize)
    monkeypatch.setattr(agent, "_read_turn", fake_read_turn)
    monkeypatch.setattr(agent.trace, "report_turn", lambda history, turn: reported.append([dict(m) for m in history]) or history.extend(turn["messages"]))
    monkeypatch.setattr(agent, "_synthesize", _speech)

    def audio_ms(text):
        return len(text.split()) * 100 * agent.BYTES_PER_MS

    greeting = audio_ms(agent.GREETING)
    turn = [SILENT] * 15 + [VOICED] * 25 + [SILENT] * 40
    # The first turn follows the greeting; the second starts 1.5 s into the long reply.
    actor = ScriptedActor(
        [(greeting, turn), (greeting + 1500 * agent.BYTES_PER_MS, turn)],
        final_bytes=greeting + 1500 * agent.BYTES_PER_MS + audio_ms("Go ahead."),
    )

    asyncio.run(asyncio.wait_for(agent.run_voice_ws_bot(actor), 30))

    assert len(turns) == 2
    first_session, first_bytes, first_attrs = turns[0]
    # 80 frames at 24 kHz is 76800 bytes, which is 51200 bytes at 16 kHz.
    assert abs(first_bytes - 51200) <= 4
    assert first_attrs[ATTR_STRIPE_API_BASE] == "https://twin.example/stripe"
    assert ATTR_INTERRUPTED_REPLY not in first_attrs
    second_session, _, second_attrs = turns[1]
    assert second_session == first_session
    heard = second_attrs[ATTR_INTERRUPTED_REPLY].removesuffix(INTERRUPTED_MARKER)
    assert second_attrs[ATTR_INTERRUPTED_REPLY].endswith(INTERRUPTED_MARKER)
    assert heard and long_reply.startswith(heard) and heard != long_reply.strip()
    # The second turn's trace starts from the trimmed reply, as the runtime's history does.
    assert reported[1][-1] == {"role": "assistant", "content": [{"text": second_attrs[ATTR_INTERRUPTED_REPLY]}]}
