"""The paced player the conversation runner plays Prism's replies through.

Prism's playback reports are only as good as the player's clock, so these pin
what it reports: a finished reply counts its whole audio once it has had time to
play, an interrupted one counts what was sent, and replies play in the order the
runner opened them.
"""

from __future__ import annotations

import asyncio

from rimelabs_sdk.realtime import AudioChunk, OutputRef, PCMFormat, ResponseRef

from rory_rime_prism import agent


class FakeActor:
    def __init__(self):
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes):
        self.sent.append(data)


def _ref(response_id: str) -> OutputRef:
    return OutputRef(
        response=ResponseRef(session_id="sess", response_id=response_id),
        item_id=f"item_{response_id}",
        output_index=0,
        content_index=0,
    )


def _audio(ms: int, fill: bytes = b"\x01") -> AudioChunk:
    return AudioChunk(data=fill * (ms * agent.PCM16_BYTES_PER_MS), format=PCMFormat(sample_rate=24000))


def test_a_finished_reply_reports_all_its_audio_after_it_has_played():
    async def go():
        actor = FakeActor()
        playback = agent.ActorPlayback(actor)
        out = playback.open(_ref("resp_1"))
        out.write(_audio(300))
        out.end()
        loop = asyncio.get_running_loop()
        start = loop.time()
        played = await asyncio.wait_for(out.completion(), timeout=5)
        await playback.stop()
        return played, loop.time() - start, actor.sent

    played, elapsed, sent = asyncio.run(go())
    assert played == 300
    assert elapsed >= 0.29  # resolved at the playback clock, not at the last send
    assert sum(len(piece) for piece in sent) == 300 * agent.PCM16_BYTES_PER_MS
    assert max(len(piece) for piece in sent) <= agent.PLAYBACK_SLICE_BYTES


def test_an_interrupted_reply_reports_what_was_sent_and_drops_the_rest():
    async def go():
        actor = FakeActor()
        playback = agent.ActorPlayback(actor)
        ref = _ref("resp_1")
        out = playback.open(ref)
        out.write(_audio(3000))  # 3 s of audio: the lead holds most of it back
        await asyncio.sleep(0.2)
        playback.interrupt([ref])
        out.end()
        played = await asyncio.wait_for(out.completion(), timeout=1)
        sent_before = sum(len(piece) for piece in actor.sent)
        await asyncio.sleep(0.3)
        sent_after = sum(len(piece) for piece in actor.sent)
        await playback.stop()
        return played, sent_before, sent_after

    played, sent_before, sent_after = asyncio.run(go())
    assert played == sent_before / agent.PCM16_BYTES_PER_MS
    assert played < 3000
    assert sent_after == sent_before, "nothing more goes out after the interruption"


def test_replies_play_in_the_order_they_were_opened():
    async def go():
        actor = FakeActor()
        playback = agent.ActorPlayback(actor)
        first, second = playback.open(_ref("resp_1")), playback.open(_ref("resp_2"))
        second.write(_audio(100, b"\x02"))
        second.end()
        first.write(_audio(100, b"\x01"))
        first.end()
        await asyncio.wait_for(second.completion(), timeout=5)
        await playback.stop()
        return b"".join(actor.sent)

    sent = asyncio.run(go())
    half = 100 * agent.PCM16_BYTES_PER_MS
    assert sent == b"\x01" * half + b"\x02" * half


class GatedActor(FakeActor):
    """An actor whose socket holds each send until the gate opens."""

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()

    async def send_bytes(self, data: bytes):
        await self.gate.wait()
        self.sent.append(data)


def test_a_slice_on_its_way_to_the_actor_when_interrupted_counts_as_heard():
    async def go():
        actor = GatedActor()
        playback = agent.ActorPlayback(actor)
        ref = _ref("resp_1")
        out = playback.open(ref)
        out.write(_audio(1000))
        await asyncio.sleep(0.05)  # the first slice is on its way
        playback.interrupt([ref])
        actor.gate.set()
        played = await asyncio.wait_for(out.completion(), timeout=1)
        await asyncio.sleep(0.05)
        await playback.stop()
        return played, sum(len(piece) for piece in actor.sent)

    played, sent = asyncio.run(go())
    assert sent == agent.PLAYBACK_SLICE_BYTES
    assert played == sent / agent.PCM16_BYTES_PER_MS
