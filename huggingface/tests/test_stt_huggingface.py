"""The STT leg: what goes up to Whisper, and what a hung request costs.

Both observed on live calls. Without a language, Whisper large-v3 read
accented callers into Arabic and Portuguese script ("إليس حداد" for Elise
Haddad) and the name check could never pass. And when hf-inference hung, every
``httpx.ReadTimeout`` escaped the turn loop and ended the call, for what was
one dropped utterance.
"""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest

from rory_huggingface import agent

PCM = bytes(agent.ACTOR_RATE_HZ * 2)  # one second of silence at the actor rate


@pytest.fixture
def log_lines():
    lines: list[str] = []
    handle = agent.logger.add(lambda m: lines.append(m.rstrip("\n")), format="{message}")
    yield lines
    agent.logger.remove(handle)


def test_transcription_is_pinned_to_english(monkeypatch):
    seen = {}

    async def post(url, *, label, **kwargs):
        seen.update(url=url, label=label, **kwargs)
        return SimpleNamespace(json=lambda: {"text": " Jordan Sample. "})

    monkeypatch.setattr(agent, "_post_with_retry", post)

    text = asyncio.run(agent._Call(SimpleNamespace())._transcribe(PCM))

    assert text == "Jordan Sample."
    assert seen["label"] == "stt"
    body = seen["json"]
    assert body["parameters"] == {"generate_kwargs": {"language": "en", "task": "transcribe"}}
    assert base64.b64decode(body["inputs"])[:4] == b"RIFF"


def test_a_hung_transcription_drops_the_turn_instead_of_the_call(monkeypatch, log_lines):
    async def post(url, *, label, **kwargs):
        raise httpx.ReadTimeout("hf-inference did not answer")

    monkeypatch.setattr(agent, "_post_with_retry", post)

    text = asyncio.run(agent._Call(SimpleNamespace())._transcribe(PCM))

    assert text == ""
    assert [line for line in log_lines if "[stt] no transcript" in line]
