"""Agent audio plays at real time, and a barge-in stops what the caller has not heard."""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock

from rory_tools import CallSession

from rory_grok_voice import agent

ONE_SECOND = bytes(agent.SAMPLE_RATE_HZ * 2)


def _sent_bytes(actor_ws: AsyncMock) -> int:
    return sum(len(call.args[0]) for call in actor_ws.send_bytes.await_args_list)


def test_audio_is_paced_at_real_time():
    async def run():
        actor_ws = AsyncMock()
        playout = agent._Playout(actor_ws)
        task = asyncio.create_task(playout.run())
        playout.write("item_1", ONE_SECOND)
        await asyncio.sleep(0.2)
        sent = _sent_bytes(actor_ws)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return sent

    sent_ms = asyncio.run(run()) / agent.PCM16_BYTES_PER_MS
    assert 200 <= sent_ms <= 200 + agent.PLAYOUT_LEAD_S * 1000 + 60


def test_interrupt_drops_unsent_audio_and_reports_what_was_sent():
    async def run():
        actor_ws = AsyncMock()
        playout = agent._Playout(actor_ws)
        task = asyncio.create_task(playout.run())
        playout.write("item_1", ONE_SECOND)
        await asyncio.sleep(0.1)
        interrupted = playout.interrupt()
        await playout.drained()
        playout.write("item_1", ONE_SECOND)
        await asyncio.sleep(0.05)
        sent = _sent_bytes(actor_ws)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return interrupted, sent

    (item_id, audio_end_ms), sent = asyncio.run(run())
    assert item_id == "item_1"
    assert audio_end_ms == sent // agent.PCM16_BYTES_PER_MS
    assert audio_end_ms < 1000


def test_nothing_to_interrupt_once_audio_has_played():
    async def run():
        playout = agent._Playout(AsyncMock())
        task = asyncio.create_task(playout.run())
        playout.write("item_1", bytes(agent.PLAYOUT_CHUNK_BYTES))
        await playout.drained()
        interrupted = playout.interrupt()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return interrupted

    assert asyncio.run(run()) is None


def test_speech_started_truncates_the_item_at_what_was_sent():
    class FakeGrok:
        def __init__(self):
            self.sent = []
            self.events = [
                {"type": "response.created", "response": {"id": "r1"}},
                {
                    "type": "response.output_audio.delta",
                    "item_id": "item_1",
                    "delta": base64.b64encode(ONE_SECOND).decode(),
                },
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.events:
                await asyncio.sleep(0)
                return json.dumps(self.events.pop(0))
            if not any(m["type"] == "conversation.item.truncate" for m in self.sent):
                await asyncio.sleep(0.1)
                return json.dumps({"type": "input_audio_buffer.speech_started"})
            raise StopAsyncIteration

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    gk_ws = FakeGrok()
    actor_ws = AsyncMock()
    asyncio.run(agent._pump_grok_to_actor(gk_ws, actor_ws, CallSession()))

    truncate = next(m for m in gk_ws.sent if m["type"] == "conversation.item.truncate")
    assert truncate["item_id"] == "item_1"
    assert truncate["content_index"] == 0
    assert 0 < truncate["audio_end_ms"] < 1000
    assert _sent_bytes(actor_ws) < len(ONE_SECOND)
