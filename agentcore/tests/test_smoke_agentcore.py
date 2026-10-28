"""Offline smoke: the real app boots, and a whole call runs through the real pump and worker.

Only the AWS legs and the voice detector are replaced. The FastAPI lifespan builds the
real boto3 clients (constructing them needs no network), and the call itself runs the
real pump and worker through greeting, endpointing, transcript, model round and spoken
reply. The call is driven without TestClient's WebSocket, whose teardown cancels the
app task while the call is still closing.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from rory_agentcore import agent
from rory_agentcore.web import app
from starlette.websockets import WebSocketDisconnect

FRAME_BYTES = agent.ACTOR_RATE_HZ * 2 // 50  # 20 ms of PCM16 at 24 kHz
VOICED = b"\x10\x00" * (FRAME_BYTES // 2)
SILENT = bytes(FRAME_BYTES)
REPLY_BYTES = agent.ACTOR_RATE_HZ * 2 // 4  # a quarter second of audio per spoken reply


class EnergyVad:
    """Stands in for Silero: a frame is speech when it is not digital silence."""

    def __init__(self, _rate):
        pass

    def push(self, frame: bytes) -> bool:
        return frame != SILENT


@pytest.fixture
def offline_env(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_fixture")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("LEX_BOT_ID", "BOTID")
    monkeypatch.setenv("LEX_BOT_ALIAS_ID", "ALIAS")
    monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:1:runtime/rory")


def test_health_boots_with_the_aws_configuration(offline_env):
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_boot_fails_loudly_without_the_aws_configuration(monkeypatch):
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_fixture")
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(RuntimeError, match="AWS_REGION"), TestClient(app):
        pass


class ScriptedActor:
    """Plays the caller's frames, then stays open until the reply has played, then disconnects."""

    client = None

    def __init__(self, frames, expect_bytes):
        self._frames = list(frames)
        self._expect_bytes = expect_bytes
        self.sent = 0

    async def receive_bytes(self):
        if not self._frames:
            while self.sent < self._expect_bytes:
                await asyncio.sleep(0.01)
            raise WebSocketDisconnect(1000)
        return self._frames.pop(0)

    async def send_bytes(self, data):
        self.sent += len(data)


def test_one_call_greets_then_answers_a_caller_turn(offline_env, monkeypatch):
    monkeypatch.setattr(agent, "SileroVad", EnergyVad)
    heard = []
    model_rounds = []

    def fake_recognize(session_id, pcm16_16k):
        heard.append(pcm16_16k)
        return "I need to pay my bill"

    def fake_agent_turn(session_id, messages):
        model_rounds.append([m["role"] for m in messages])
        return {"content": "Of course, let's sort that out.", "tool_calls": []}

    monkeypatch.setattr(agent, "_recognize", fake_recognize)
    monkeypatch.setattr(agent, "_agent_turn", fake_agent_turn)
    monkeypatch.setattr(agent, "_synthesize", lambda text: ([], bytes(REPLY_BYTES)))

    # 300 ms of silence (the pre-roll), 0.5 s of speech, then 0.8 s of silence to endpoint.
    caller = [SILENT] * 15 + [VOICED] * 25 + [SILENT] * 40
    actor = ScriptedActor(caller, expect_bytes=2 * REPLY_BYTES)  # the greeting, then the reply

    asyncio.run(agent.run_voice_ws_bot(actor))

    assert actor.sent == 2 * REPLY_BYTES
    # The caller's turn reached Lex at 16 kHz: 80 frames at 24 kHz is 76800 bytes,
    # which is 51200 bytes at 16 kHz.
    assert len(heard) == 1
    assert abs(len(heard[0]) - 51200) <= 4
    # The model saw the two system messages, the greeting, then the caller's words.
    assert model_rounds == [["system", "system", "assistant", "user"]]
