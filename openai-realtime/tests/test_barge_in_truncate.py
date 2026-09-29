"""Barge-in truncates the live item to the audio the caller actually heard.

Server VAD cancels the response on ``input_audio_buffer.speech_started`` but
leaves the item's full transcript in the model's context. The bridge must send
``conversation.item.truncate`` at the audio it had forwarded, and only for an
item that is still in flight.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock

from rory_tools import CallSession
from test_cancelled_tool_call import FakeRealtime

from rory_openai_realtime import agent

# 4800 bytes of PCM16 at 24 kHz is 100 ms.
DELTA = bytes(4800)


def _run(events: list[dict]):
    oa_ws = FakeRealtime(events)
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_oa_to_actor(oa_ws, actor_ws, CallSession()))
    return oa_ws.sent, actor_ws


def _delta(item_id: str) -> dict:
    return {
        "type": "response.output_audio.delta",
        "item_id": item_id,
        "delta": base64.b64encode(DELTA).decode(),
    }


def _truncates(sent: list[dict]) -> list[dict]:
    return [m for m in sent if m["type"] == "conversation.item.truncate"]


def test_speech_over_live_item_truncates_at_forwarded_audio():
    events = [
        {"type": "response.created"},
        _delta("item_live"),
        _delta("item_live"),
        _delta("item_live"),
        {"type": "input_audio_buffer.speech_started"},
        {"type": "conversation.item.truncated", "item_id": "item_live", "audio_end_ms": 300},
        {"type": "response.done"},
    ]

    sent, actor_ws = _run(events)

    assert _truncates(sent) == [
        {
            "type": "conversation.item.truncate",
            "item_id": "item_live",
            "content_index": 0,
            "audio_end_ms": 300,
        }
    ]
    assert actor_ws.send_bytes.await_count == 3


def test_speech_with_no_audio_item_in_flight_does_not_truncate():
    events = [
        {"type": "response.created"},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.done"},
    ]

    sent, _ = _run(events)

    assert _truncates(sent) == []


def test_speech_after_response_done_does_not_truncate():
    events = [
        {"type": "response.created"},
        _delta("item_done"),
        {"type": "response.output_audio.done", "item_id": "item_done"},
        {"type": "response.done"},
        {"type": "input_audio_buffer.speech_started"},
    ]

    sent, _ = _run(events)

    assert _truncates(sent) == []
