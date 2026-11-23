"""One Dialt WebSocket per benchmark call; no provider SDK modifications.

The actor uses PCM16/24 kHz, Dialt input uses PCM16/16 kHz, and Dialt
output is negotiated at 24 kHz. Dialt owns turn detection and managed tools;
Rory's business tools always run through the shared dispatcher.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import logging
import os
import time
import uuid
from collections import defaultdict

import websockets
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.impl import UNVERIFIED
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

DIALT_URL = "wss://api.dialt.com/v1/realtime"
DIALT_BRAIN = os.environ.get("DIALT_BRAIN", "fast")
DIALT_VOICE = os.environ.get("DIALT_VOICE", "us_female")
AGENT_PROMPT = load_agent_prompt()
OUTPUT_BYTES_PER_MS = 48
FRAME_BYTES = 960  # 20 ms of PCM16/24 kHz
PLAYBACK_LEAD_S = 0.06
# Dialt ends a session whose microphone audio runs 2 s ahead of real time.
# Stay well inside that so network batching after this bridge cannot reach it.
UPLINK_BURST_S = 1.0
logger = logging.getLogger("uvicorn.error")


def validate_config() -> None:
    if DIALT_BRAIN not in {"fast", "smart"}:
        raise ValueError("DIALT_BRAIN must be fast or smart")


def _start() -> dict:
    validate_config()
    return {
        "type": "start",
        "session_id": str(uuid.uuid4()),
        "audio": {"sr": 16000, "output_sr": 24000, "output_encoding": "pcm16"},
        "mode": {
            "kind": "dialt",
            "brain": DIALT_BRAIN,
            "voice": DIALT_VOICE,
            "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
            "greeting": GREETING,
            "tools": TOOLS,
            "web_search": False,
            "end_call": False,
            # Session-only beta extension, deliberately outside the public SDK.
            # The managed tool is supplied and executed by Dialt, not by Rory.
            "agent_background_speaker_suppression": True,
            "background_speaker_suppression": False,
        },
    }


class _Playback:
    """Pace output so an interruption can discard audio still owned here.

    voice_ws has no clear-buffer command or playback acknowledgements. Once
    sent, a frame cannot be recalled. Keep a 60 ms lead (plus the current
    20 ms frame) to absorb send/scheduler jitter without queuing a whole reply.
    """

    def __init__(self, actor: WebSocket):
        self.actor = actor
        self.queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()
        self.queued_bytes: dict[str, int] = defaultdict(int)

    def append(self, response: str, audio: bytes) -> None:
        if len(audio) % 2:
            raise RuntimeError("Dialt sent an incomplete PCM16 sample")
        self.queued_bytes[response] += len(audio)
        for offset in range(0, len(audio), FRAME_BYTES):
            self.queue.put_nowait((response, audio[offset:offset + FRAME_BYTES]))

    def clear(self, response: str) -> float:
        discarded = 0
        kept = []
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if item[0] == response:
                discarded += len(item[1])
            else:
                kept.append(item)
            self.queue.task_done()
        for item in kept:
            self.queue.put_nowait(item)
        self.queued_bytes[response] -= discarded
        return discarded / OUTPUT_BYTES_PER_MS

    async def run(self) -> None:
        started = None
        sent_s = 0.0
        while True:
            response, audio = await self.queue.get()
            self.queued_bytes[response] -= len(audio)
            try:
                now = time.monotonic()
                # Reset after actual idle time, not merely an empty queue:
                # the actor may still be playing the previous chunk's lead.
                if started is None or now >= started + sent_s:
                    started, sent_s = now, 0.0
                await self.actor.send_bytes(audio)
                sent_s += len(audio) / 48000
                ahead = sent_s - (time.monotonic() - started)
                if ahead > PLAYBACK_LEAD_S:
                    await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
            finally:
                self.queue.task_done()


async def _uplink(actor: WebSocket, provider: asyncio.Future) -> None:
    """Forward microphone audio no faster than Dialt admits it.

    Reading starts when the actor connects, so audio sent while Dialt is
    still starting is discarded as it arrives instead of queueing up and
    being delivered late in one burst. Afterwards a real-time budget drops
    whatever arrives more than UPLINK_BURST_S ahead of the clock.
    """
    state = None
    provider_closed = False
    budget = UPLINK_BURST_S
    last = time.monotonic()
    dropped_s = 0.0
    try:
        while True:
            audio = await actor.receive_bytes()
            if provider_closed:
                continue
            if len(audio) % 2:
                raise ValueError("voice_ws requires complete PCM16 samples")
            now = time.monotonic()
            budget = min(UPLINK_BURST_S, budget + now - last)
            last = now
            if not provider.done():
                continue
            ws = provider.result()
            seconds = len(audio) / 48000
            if seconds > budget:
                # The resampler history no longer precedes the next frame.
                state = None
                dropped_s += seconds
                continue
            budget -= seconds
            if dropped_s:
                logger.warning("Dialt uplink dropped %.0f ms of burst audio", dropped_s * 1000)
                dropped_s = 0.0
            # Keep resampler history across frames; resetting it causes clicks
            # and loses samples at every actor packet boundary.
            converted, state = audioop.ratecv(audio, 2, 1, 24000, 16000, state)
            if converted:
                try:
                    await ws.send(converted)
                except websockets.ConnectionClosedOK:
                    # Keep watching actor disconnects, but let the downlink
                    # finish draining queued audio before supervision exits.
                    provider_closed = True
    except WebSocketDisconnect:
        return


async def _tools(ws, calls: asyncio.Queue, cancelled: set[str], session: CallSession) -> None:
    while True:
        call = await calls.get()
        try:
            if call["id"] in cancelled:
                continue
            result = await asyncio.to_thread(dispatch, session, call["name"], call["args"])
            # A blocking vendor request already in flight cannot be undone.
            # Never start queued cancelled work or replay an uncertain write.
            if call["id"] in cancelled:
                continue
            # Generic errors cannot prove a write failed: its response or a
            # post-write refresh may have failed after the write committed.
            # Only explicit negative results establish failure. Preserve the
            # shared result verbatim so the model still sees any refusal.
            outcome = "succeeded"
            if result == UNVERIFIED or result.get("verified") is False or result.get("paid") is False:
                outcome = "failed"
            elif "error" in result:
                outcome = "unknown"
            try:
                await ws.send(json.dumps({
                    "type": "tool_result",
                    "id": call["id"],
                    "content": result,
                    "outcome": outcome,
                    "verified": outcome == "succeeded",
                }, default=str))
            except websockets.ConnectionClosedOK:
                # No more tools may start after provider closure. Stay alive
                # until downlink drain or actor disconnect ends supervision.
                await asyncio.Future()
        finally:
            calls.task_done()


async def _downlink(ws, playback: _Playback, calls: asyncio.Queue, cancelled: set[str]) -> None:
    response = ""
    responses_by_turn: dict[str, str] = {}
    suppressed: set[str] = set()
    async for raw in ws:
        if isinstance(raw, bytes):
            if response not in suppressed:
                playback.append(response, raw)
            continue
        event = json.loads(raw)
        kind = event.get("type")
        if kind == "turn":
            response = event.get("response_id") or event["turn_id"]
            responses_by_turn[event["turn_id"]] = response
        elif kind == "tool_call":
            # A dedicated sequential worker keeps audio and interruption
            # receipts flowing even while a vendor lookup blocks.
            calls.put_nowait(event)
        elif kind == "tool_cancel":
            cancelled.add(event["id"])
        elif kind in {"canceled", "interrupted"}:
            # Segment IDs and logical response IDs are different namespaces.
            # An unknown explicit old ID must not stop the current response;
            # only legacy events omitting both IDs refer to current output.
            turn = event.get("turn_id")
            target = event.get("response_id") or responses_by_turn.get(turn, turn) or response
            suppressed.add(target)
            if kind == "canceled":
                playback.clear(target)
                continue
            # Soft stops let already queued audio drain. Hard stops discard
            # only the interrupted response, not a newer response's output.
            discarded_ms = playback.clear(target) if event.get("clear") else 0
            report = {
                "type": "client_event",
                "event": "playback_stopped",
                "remaining_ms": round(playback.queued_bytes[target] / OUTPUT_BYTES_PER_MS),
                "discarded_ms": round(discarded_ms),
            }
            if "barge_seq" in event:
                report["barge_seq"] = event["barge_seq"]
            try:
                await ws.send(json.dumps(report))
            except websockets.ConnectionClosedOK:
                # The peer may close while buffered events are still being
                # consumed. Finish those events and drain surviving audio.
                pass
        elif kind == "error":
            logger.warning("Dialt error code=%s retryable=%s", event.get("code"), event.get("retryable"))
            raise RuntimeError("Dialt reported a session error")
    # Normal closure preserves the last spoken reply. Abnormal socket closure
    # raises from the iterator and fails the call instead of hiding a dropout.
    await playback.queue.join()


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    headers = {"Authorization": f"Bearer {os.environ['DIALT_API_KEY']}"}
    provider: asyncio.Future = asyncio.get_running_loop().create_future()
    uplink = asyncio.create_task(_uplink(actor_ws, provider))
    try:
        await _run(actor_ws, headers, provider, uplink)
    finally:
        uplink.cancel()
        await asyncio.gather(uplink, return_exceptions=True)


async def _run(actor_ws: WebSocket, headers: dict, provider: asyncio.Future, uplink: asyncio.Task) -> None:
    async with websockets.connect(DIALT_URL, additional_headers=headers) as ws:
        logger.info("Dialt provider connected tier=%s", DIALT_BRAIN)
        await ws.send(json.dumps(_start()))
        async with asyncio.timeout(15):
            while True:
                event = json.loads(await ws.recv())
                if event.get("type") == "error":
                    logger.warning("Dialt startup error code=%s retryable=%s", event.get("code"), event.get("retryable"))
                    raise RuntimeError("Dialt rejected session startup")
                if event.get("type") == "ready":
                    audio = event.get("audio", {})
                    if audio.get("output_sr") != 24000 or audio.get("output_encoding") != "pcm16":
                        raise RuntimeError("Dialt did not negotiate PCM16/24 kHz output")
                    suppression = event.get("background_speaker_suppression", {})
                    if suppression.get("enabled") is not False or suppression.get("agent_control") is not True:
                        raise RuntimeError("Dialt did not confirm the requested suppression settings")
                    logger.info("Dialt session ready tier=%s suppression=false agent_control=true", DIALT_BRAIN)
                    break

        provider.set_result(ws)
        playback = _Playback(actor_ws)
        calls: asyncio.Queue = asyncio.Queue()
        cancelled: set[str] = set()
        tasks = [
            uplink,
            asyncio.create_task(_downlink(ws, playback, calls, cancelled)),
            asyncio.create_task(playback.run()),
            asyncio.create_task(_tools(ws, calls, cancelled, CallSession())),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
