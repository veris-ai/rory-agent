"""A turn's text is every Pulse final since the last turn, closed by the finalize ack.

Verified live (2026-10-14, Lightning's own TTS looped into Pulse): an
``is_final`` transcript covers only the audio since the previous final, and
Pulse finalizes on its own at a ~0.6 s pause — shorter than END_OF_TURN_S. So
"My account number is 447219 … and my postcode is 19103." arrives as two
finals inside one turn, then our ``finalize`` is answered by an empty final
with ``from_finalize: true``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from rory_smallest import agent


class _FakeSTT:
    """Pulse's WebSocket: a fixed list of events; records what is sent."""

    def __init__(self, messages=()):
        self._messages = list(messages)
        self.sent: list[str] = []

    async def send(self, payload) -> None:
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _final(transcript: str, from_finalize: bool = False) -> str:
    event = {"type": "transcription", "is_final": True, "transcript": transcript}
    if from_finalize:
        event["from_finalize"] = True
    return json.dumps(event)


def _interim(transcript: str) -> str:
    return json.dumps({"type": "transcription", "is_final": False, "transcript": transcript})


def _turns(messages) -> list[str]:
    call = agent._Call(SimpleNamespace())
    call._stt = _FakeSTT(messages)
    with pytest.raises(RuntimeError, match="closed the stream"):
        asyncio.run(call._pump_stt_events())
    turns = []
    while not call._turns.empty():
        turns.append(call._turns.get_nowait())
    return turns


def test_finals_within_a_turn_are_joined():
    assert _turns([
        _interim("My account"),
        _final(" My account number is 447219"),
        _interim("and my"),
        _final(" and my postcode is 19103."),
        _final("", from_finalize=True),
    ]) == ["My account number is 447219 and my postcode is 19103."]


def test_the_finalize_ack_carries_the_pending_text():
    assert _turns([_interim("yes"), _final("Yes.", from_finalize=True)]) == ["Yes."]


def test_an_automatic_final_does_not_close_the_turn():
    """Only the ack closes it: text finalized before it waits for the ack."""
    assert _turns([_final("My account number is")]) == []


def test_each_finalize_ack_closes_its_own_turn():
    """A second pause before the first ack lands is a second turn, not lost words."""
    assert _turns([
        _final("Yes.", from_finalize=True),
        _final("Actually, no.", from_finalize=True),
    ]) == ["Yes.", "Actually, no."]


def test_an_empty_turn_dispatches_nothing():
    assert _turns([_final("", from_finalize=True)]) == []


def test_endpoint_requests_finalize_without_waiting():
    call = agent._Call(SimpleNamespace())
    call._stt = _FakeSTT()
    call._voiced = True

    asyncio.run(call._endpoint())

    assert [json.loads(m) for m in call._stt.sent] == [{"type": "finalize"}]
    assert call._voiced is False
