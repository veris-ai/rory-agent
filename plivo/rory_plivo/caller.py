"""The caller's leg: the bench actor's ``voice_ws`` audio placed as a real Plivo call.

For each ``/voice`` connection this module logs in as the pod's Plivo endpoint
(``rory_plivo.sip``) and places a WebRTC call with aiortc, exactly as Plivo's
browser SDK would from a softphone. Plivo answers by fetching the endpoint
Application's answer_url, which streams the call to ``/plivo/stream`` where Rory
runs (``rory_plivo.agent``). Audio is bridged both ways:

    actor PCM16/24k -> paced mic track -> Opus/WebRTC -> Plivo -> μ-law/8k stream -> Rory
    Rory -> μ-law/8k playAudio -> Plivo -> Opus/WebRTC -> resample -> actor PCM16/24k

So the agent hears, and is heard through, Plivo's media path — its codecs,
narrowband stream and latency — not a loopback.

Two interoperability fixes against Plivo's media servers:

- DTLS session tickets are off. With them, aiortc's final handshake flight is
  one 1500-byte datagram (mostly the ticket); it does not survive the path, so
  Plivo never completes DTLS and sends no media, while aiortc reports
  ``connected``. Browsers send no ticket.
- Plivo's SDP answer carries no ``a=mid``, ``a=group:BUNDLE`` or direction
  line, which aiortc requires to match the answer to its offer; they are added
  back before ``setRemoteDescription``.
"""

from __future__ import annotations

import asyncio
import fractions
import time

import aiortc.rtcdtlstransport as rtcdtlstransport
import av
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from loguru import logger
from OpenSSL import SSL
from starlette.websockets import WebSocket, WebSocketDisconnect

from .plivo_api import Provisioned
from .sip import SIGNALLING_URL, SipClient

# Any destination works: the call never leaves Plivo, it runs the endpoint's
# Application. The name shows up in Plivo's call logs.
DESTINATION = "rory"

# The Veris actor speaks and listens at 24 kHz PCM16.
SAMPLE_RATE_HZ = 24000
AUDIO_PTIME = 0.02
SAMPLES_PER_FRAME = int(SAMPLE_RATE_HZ * AUDIO_PTIME)
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
LOG_EVERY_N_FRAMES = 50

_create_ssl_context = rtcdtlstransport.RTCCertificate._create_ssl_context


def _create_ssl_context_without_tickets(self, *args, **kwargs):
    ctx = _create_ssl_context(self, *args, **kwargs)
    ctx.set_options(SSL.OP_NO_TICKET)
    return ctx


rtcdtlstransport.RTCCertificate._create_ssl_context = _create_ssl_context_without_tickets


def complete_answer_sdp(sdp: str) -> str:
    """Add the BUNDLE group, mid and direction Plivo's single-audio answer omits."""
    sdp = sdp.replace("\r\nm=audio", "\r\na=group:BUNDLE 0\r\nm=audio", 1)
    return sdp.replace("\r\na=rtcp-mux", "\r\na=mid:0\r\na=sendrecv\r\na=rtcp-mux", 1)


class ActorAudioTrack(MediaStreamTrack):
    """The actor's PCM16 stream as a paced WebRTC microphone track.

    ``recv()`` runs on its own 20 ms clock — real audio when buffered, silence
    otherwise — so gaps or bursts on the actor side never distort RTP timing.
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


async def run_caller_leg(actor_ws: WebSocket, endpoint: Provisioned) -> None:
    """One actor connection ↔ one Plivo WebRTC call, until either side hangs up."""
    t_start = time.monotonic()
    pc = RTCPeerConnection()
    mic = ActorAudioTrack()
    pc.addTrack(mic)
    agent_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()
    pc_dead = asyncio.Event()

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio" and not agent_track.done():
            agent_track.set_result(track)

    @pc.on("connectionstatechange")
    def on_connectionstatechange() -> None:
        logger.info(f"[caller] peer connection state={pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            pc_dead.set()

    dialog = None
    try:
        async with websockets.connect(SIGNALLING_URL, subprotocols=["sip"], max_size=None) as signal_ws:
            sip = SipClient(signal_ws, endpoint.username, endpoint.password)
            await sip.register()
            # aiortc finishes ICE gathering inside setLocalDescription, so the
            # offer is complete and nothing is trickled.
            await pc.setLocalDescription(await pc.createOffer())
            dialog = await sip.invite(DESTINATION, pc.localDescription.sdp)
            logger.info(f"[caller] Plivo answered call_id={dialog.call_id} after {time.monotonic() - t_start:.1f}s")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=complete_answer_sdp(dialog.answer_sdp), type="answer"))

            tasks = {
                asyncio.create_task(_pump_actor_to_plivo(actor_ws, mic), name="actor->plivo"),
                asyncio.create_task(_pump_plivo_to_actor(agent_track, actor_ws), name="plivo->actor"),
                asyncio.create_task(sip.wait_for_remote_bye(), name="remote-bye"),
                asyncio.create_task(pc_dead.wait(), name="pc-dead"),
            }
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    logger.info(f"[caller] call ending: {task.get_name()} finished")
                    task.result()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if not any(t.get_name() == "remote-bye" for t in done):
                    try:
                        await sip.bye(dialog)
                    except websockets.ConnectionClosed:
                        pass  # signalling already gone; Plivo ends the call on media timeout
    finally:
        if not agent_track.done():
            agent_track.cancel()
        await pc.close()
        logger.info(f"[caller] leg closed duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_plivo(actor_ws: WebSocket, mic: ActorAudioTrack) -> None:
    """Binary PCM16 frames from the actor → the paced microphone track."""
    n_frames = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            if n_frames == 1:
                logger.info(f"[a->p] first actor frame bytes={len(frame)}")
            elif n_frames % (LOG_EVERY_N_FRAMES * 10) == 0:
                logger.info(f"[a->p] forwarded {n_frames} frames")
            mic.push(frame)
    except WebSocketDisconnect as exc:
        logger.info(f"[a->p] actor disconnected after {n_frames} frames: code={exc.code}")


async def _pump_plivo_to_actor(agent_track: asyncio.Future[MediaStreamTrack], actor_ws: WebSocket) -> None:
    """The call's WebRTC audio → 24 kHz PCM16 back to the actor."""
    track = await agent_track
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    n_frames = 0
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            logger.info(f"[p->a] call track ended after {n_frames} frames")
            return
        for out in resampler.resample(frame):
            # The plane buffer may carry alignment padding past the samples.
            data = bytes(out.planes[0])[: out.samples * 2]
            n_frames += 1
            if n_frames == 1:
                logger.info(f"[p->a] first call audio frame bytes={len(data)}")
            elif n_frames % (LOG_EVERY_N_FRAMES * 10) == 0:
                logger.info(f"[p->a] forwarded {n_frames} frames")
            try:
                await actor_ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError, OSError) as exc:
                logger.info(f"[p->a] actor WS closed mid-stream after {n_frames} frames: {exc}")
                return
