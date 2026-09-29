"""Rory on PolyAI Agent Studio — Acme Energy's voice support agent on a hosted platform.

PolyAI runs the whole voice loop on its side: ASR, the agent (the rules pushed
by ``rory_polyai.studio``), TTS and turn-taking, for a call that reaches the
project over its WebRTC gateway. Its only realtime audio surface is that
gateway — WebSocket signaling plus Opus media — so for each ``/voice``
connection this module dials the gateway as a WebRTC peer, exactly as PolyAI's
browser web-calling widget does, and bridges audio in both directions with
aiortc. Tool calls do not travel over this call; the Agent Studio functions
POST them to the ``/tool`` webhook in ``rory_polyai.web``, which resolves them
through ``rory_tools.dispatch``.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher.
The candidates exist to be compared, so the transport is the only thing
allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16 and the
gateway carries 48 kHz Opus. The caller's frames go out through a paced
virtual microphone track that aiortc's Opus encoder resamples on its own; the
decoded reply frames are resampled back to 24 kHz mono here.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import os
import time
import uuid

import av
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from .studio import REGION

# The Veris actor speaks and listens at 24 kHz PCM16.
SAMPLE_RATE_HZ = 24000

# One WebRTC audio frame per 20 ms, the cadence the mic track paces itself to.
AUDIO_PTIME = 0.02
SAMPLES_PER_FRAME = int(SAMPLE_RATE_HZ * AUDIO_PTIME)
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2

# The docs publish one signaling URL, the us-1 one, and it also serves
# self-serve ("studio") projects, which have no signaling route of their own.
# euw-1 and uk-1 follow the documented naming.
GATEWAY_HOSTS = {
    "us-1": "webrtc-gateway.us-1.platform.polyai.app",
    "euw-1": "webrtc-gateway.euw-1.platform.polyai.app",
    "uk-1": "webrtc-gateway.uk-1.platform.polyai.app",
    "studio": "webrtc-gateway.us-1.platform.polyai.app",
}
GATEWAY_URL = os.environ.get("POLYAI_GATEWAY_URL") or f"wss://{GATEWAY_HOSTS[REGION]}/api/v1/webrtc/signal"

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50


class ActorAudioTrack(MediaStreamTrack):
    """The actor's PCM16 stream as a paced WebRTC microphone track.

    The actor pushes frames at its own rhythm; RTP wants a steady real-time
    cadence with sane timestamps. ``recv()`` runs on its own 20 ms clock —
    real audio when buffered, silence otherwise, as a live phone microphone
    never goes quiet — so gaps or bursts on the actor side never distort RTP
    timing.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._start: float | None = None
        self._timestamp = 0

    def push(self, data: bytes) -> None:
        self._buffer.extend(data)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += SAMPLES_PER_FRAME
            wait = self._start + self._timestamp / SAMPLE_RATE_HZ - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        if len(self._buffer) >= BYTES_PER_FRAME:
            chunk = bytes(self._buffer[:BYTES_PER_FRAME])
            del self._buffer[:BYTES_PER_FRAME]
        else:
            chunk = b"\x00" * BYTES_PER_FRAME

        frame = av.AudioFrame(format="s16", layout="mono", samples=SAMPLES_PER_FRAME)
        frame.planes[0].update(chunk)
        frame.pts = self._timestamp
        frame.sample_rate = SAMPLE_RATE_HZ
        frame.time_base = fractions.Fraction(1, SAMPLE_RATE_HZ)
        return frame


def offer_message(sdp: str, call_sid: str) -> dict:
    """The signaling offer, pinned so a run is reproducible.

    ``authToken`` is the project's connector token, the same credential its
    SIP connector presents. ``sessionId`` is empty on the offer; the gateway
    assigns one in its answer.
    """
    return {
        "type": "offer",
        "sessionId": "",
        "data": {"type": "offer", "sdp": sdp},
        "authToken": os.environ["POLYAI_AUTH_TOKEN"],
        "callSid": call_sid,
        "accountId": os.environ["POLYAI_ACCOUNT_ID"],
        "projectId": os.environ["POLYAI_PROJECT_ID"],
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one WebRTC call to the live Agent Studio deployment.

    The agent greets first: the project's voice greeting is the shared one,
    spoken by the platform as soon as the call connects. The ``CallSession``
    is not built here — the gateway never says which conversation it opened,
    so the ``/tool`` webhook registers one per conversation id instead.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    t_start = time.monotonic()
    call_sid = f"rory-{uuid.uuid4()}"

    pc = RTCPeerConnection()
    mic = ActorAudioTrack()
    pc.addTrack(mic)
    agent_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()
    pc_dead = asyncio.Event()
    session_id = ""

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        logger.info(f"[voice] remote track kind={track.kind}")
        if track.kind == "audio" and not agent_track.done():
            agent_track.set_result(track)

    @pc.on("connectionstatechange")
    def on_connectionstatechange() -> None:
        logger.info(f"[voice] peer connection state={pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            pc_dead.set()

    async def pump_signaling(signal_ws) -> None:
        """Answer, trickled ICE, errors and close from the gateway."""
        nonlocal session_id
        async for raw in signal_ws:
            msg = json.loads(raw)
            mtype = msg.get("type", "?")
            if mtype == "answer":
                session_id = msg.get("sessionId", "")
                logger.info(f"[voice] gateway answer session={session_id}")
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["data"]["sdp"], type="answer"))
            elif mtype == "ice-candidate":
                data = msg["data"]
                sdp = data.get("candidate") or ""
                if not sdp:
                    continue  # end-of-candidates marker
                candidate = candidate_from_sdp(sdp.removeprefix("candidate:"))
                candidate.sdpMid = data.get("sdpMid")
                candidate.sdpMLineIndex = data.get("sdpMLineIndex")
                await pc.addIceCandidate(candidate)
            elif mtype == "close":
                logger.info(f"[voice] gateway closed session {session_id}")
                return
            elif mtype == "error":
                data = msg.get("data") or {}
                raise RuntimeError(f"gateway error {data.get('code')}: {data.get('message')}")
            else:
                logger.info(f"[voice] gateway event {mtype}: {raw[:200]}")
        logger.info("[voice] gateway signaling WS ended")

    try:
        async with websockets.connect(GATEWAY_URL, max_size=None) as signal_ws:
            logger.info(f"[voice] gateway signaling connected call={call_sid}")
            # aiortc gathers ICE candidates before setLocalDescription
            # completes, so the offer SDP is complete and nothing is trickled
            # from this side. The gateway may still trickle.
            await pc.setLocalDescription(await pc.createOffer())
            await signal_ws.send(json.dumps(offer_message(pc.localDescription.sdp, call_sid)))

            tasks = {
                asyncio.create_task(_pump_actor_to_polyai(actor_ws, mic), name="actor->polyai"),
                asyncio.create_task(_pump_polyai_to_actor(agent_track, actor_ws), name="polyai->actor"),
                asyncio.create_task(pump_signaling(signal_ws), name="signaling"),
                asyncio.create_task(pc_dead.wait(), name="pc-dead"),
            }
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if session_id:
                    try:
                        await signal_ws.send(json.dumps({"type": "close", "sessionId": session_id}))
                    except websockets.ConnectionClosed:
                        pass  # the gateway hung up first; nothing left to close
    finally:
        if not agent_track.done():
            agent_track.cancel()
        await pc.close()
        logger.info(f"[voice] handler exit call={call_sid} duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_polyai(actor_ws: WebSocket, mic: ActorAudioTrack) -> None:
    """Binary PCM16 frames from the actor → the paced microphone track."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->p] first actor frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->p] forwarded {n_frames} frames ({n_bytes} bytes)")
            mic.push(frame)
    except WebSocketDisconnect as exc:
        logger.info(f"[a->p] actor disconnected after {n_frames} frames ({n_bytes} bytes): code={exc.code}")
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(f"[a->p] non-binary frame after {n_frames} binary frames — actor protocol mismatch? ({exc})")
        raise


async def _pump_polyai_to_actor(agent_track: asyncio.Future[MediaStreamTrack], actor_ws: WebSocket) -> None:
    """Rory's WebRTC audio → PCM16 bytes back to the actor.

    Decoded Opus frames arrive at 48 kHz stereo and are resampled to the wire
    protocol's 24 kHz mono. WebRTC delivers audio continuously while the
    connection is up, so the actor's VAD sees an unbroken stream and commits
    turns on its own; no end-of-turn silence trailer is needed.
    """
    track = await agent_track
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    n_frames = 0
    n_bytes = 0
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            logger.info(f"[p->a] agent track ended after {n_frames} frames ({n_bytes} bytes)")
            return
        for out in resampler.resample(frame):
            # s16 mono is packed: one plane, 2 bytes per sample. Slice to the
            # sample count; the plane buffer may carry alignment padding.
            data = bytes(out.planes[0])[: out.samples * 2]
            n_frames += 1
            n_bytes += len(data)
            if n_frames == 1:
                logger.info(f"[p->a] first audio frame bytes={len(data)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[p->a] forwarded {n_frames} frames ({n_bytes} bytes)")
            try:
                await actor_ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                # The agent track streams continuously, so a normal actor
                # hangup usually lands mid-send: the end of the call, not a
                # pump failure.
                logger.info(f"[p->a] actor WS closed mid-stream after {n_frames} frames ({n_bytes} bytes): {exc}")
                return
