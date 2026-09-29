"""rory-livekit — Rory customer-service agent on LiveKit Agents.

A cascaded LiveKit ``AgentSession`` — Deepgram STT, an OpenAI gpt-4.1-mini chat
LLM, and ElevenLabs TTS — running as an ``AgentServer`` worker. Rory handles
Acme Energy billing questions and payment assistance against two real vendor
APIs — Stripe for billing and Apache Fineract for payment arrangements — which
the bench points at its twins.

LiveKit is WebRTC end to end, so unlike the Pipecat cascade it cannot terminate
Veris's ``voice_ws`` channel itself. Three processes share the container (see
``start.sh``):

    livekit-server           the SFU, on localhost:7880
    python -m rory_livekit.agent start
                             this module as the worker: connects out to the SFU
                             and is dispatched into every room that gets created
    uvicorn rory_livekit.web:app
                             the voice_ws bridge: ``run_voice_ws_bot`` below joins
                             a fresh room per call as the caller participant and
                             relays PCM16 audio both ways

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time

from livekit import api, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import deepgram, elevenlabs, openai, silero
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import RORY_TOOLS

AGENT_PROMPT = load_agent_prompt()

# The bridge dials the in-container SFU; start.sh exports the dev defaults.
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")

# voice_ws wire format: PCM16 / 24 kHz / mono, 20 ms frames. LiveKit's room IO
# defaults to 24 kHz on both the agent's input and output tracks, so the bridge
# publishes and subscribes at the actor's rate and nothing is resampled here.
SAMPLE_RATE_HZ = 24000
NUM_CHANNELS = 1
FRAME_DURATION_MS = 20
FRAME_SAMPLES = SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000  # 480
FRAME_BYTES = FRAME_SAMPLES * 2 * NUM_CHANNELS  # 960

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

# How long the bridge waits for the worker to be dispatched in and publish its
# audio track. start.sh gates the bridge on the worker having registered, so
# in practice this is the greeting's TTS latency, not a dispatch race.
AGENT_AUDIO_TIMEOUT_S = 30

# How long the greeting may take to finish playing before the worker gives up
# on it. ElevenLabs returns the first chunk in well under a second when it
# answers at all; one observed call sat in ``say(GREETING)`` for the full 600 s
# with no error anywhere.
GREETING_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# The worker: Rory inside a LiveKit room
# ---------------------------------------------------------------------------


class RoryAgent(Agent):
    """Rory, with the shared tools registered on the LiveKit agent.

    Today's date rides along in the instructions because a bill is only "last
    month's" relative to now and an instalment date has to be real — left to
    guess, the model picks a year from its training data and Fineract refuses
    the transaction outright. It is resolved per call, off the frozen clock the
    attempt restored with its snapshot.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=f"{AGENT_PROMPT}\n\n{today_context()}",
            tools=list(RORY_TOOLS),
        )

    async def on_enter(self) -> None:
        # ``say`` speaks the fixed wording through TTS and records it as the
        # first assistant turn, so the greeting is visible to the trace-based
        # grader and the model does not repeat it.
        logger.info("[agent] entered room — speaking greeting")
        try:
            await asyncio.wait_for(self.session.say(GREETING), GREETING_TIMEOUT_S)
        except TimeoutError:
            raise RuntimeError(
                f"greeting did not finish playing within {GREETING_TIMEOUT_S}s"
            ) from None


# load_fnc always reports 0 so this dedicated, one-call-per-container worker
# never self-throttles. A `start` (prod-mode) worker's default CPU-based load
# function trips its 0.7 threshold when the SFU, worker and bridge share one
# CPU-bound container under load; the SFU then reports "no workers with
# sufficient capacity", the agent never joins the room and the caller hears no
# answer.
server = AgentServer(load_fnc=lambda *_: 0.0)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"[worker] joining room={ctx.room.name}")

    session = AgentSession(
        # Who the caller has proved themselves to be, for this call only. Tool
        # handlers read it as ``context.userdata``; the model never sees it,
        # because it is not a tool argument.
        userdata=CallSession(),
        # Plain Silero VAD with an 800 ms end-of-turn silence threshold — the
        # same ~0.8 s the Pipecat and Gemini candidates land on. No turn-
        # detector plugin, so endpointing is pure silence; the effective
        # endpoint is max(VAD silence, endpointing min_delay), so both are 0.8.
        turn_handling={"endpointing": {"min_delay": 0.8}},
        # LiveKit forces a spoken reply with tools disabled once a turn has
        # chained this many tool steps. The other candidates chain freely, and
        # a verify → account → bills → explain turn is already four, so the
        # default of 3 would cut Rory off mid-lookup where they would not be.
        max_tool_steps=10,
        stt=deepgram.STT(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            model=os.environ.get("DEEPGRAM_MODEL", "nova-3-general"),
        ),
        llm=openai.LLM(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
            # LiveKit runs a batch of tool calls concurrently. Calls share
            # verification and payment state, so the model is held to one call
            # per step; each step then runs, and is observed, in order.
            parallel_tool_calls=False,
        ),
        tts=elevenlabs.TTS(
            # The plugin reads ELEVEN_API_KEY by default; the bench provides
            # ELEVENLABS_API_KEY, so it is passed explicitly.
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2"),
        ),
        vad=silero.VAD.load(min_silence_duration=0.8, activation_threshold=0.5),
    )

    @session.on("conversation_item_added")
    def _on_item(event: ConversationItemAddedEvent) -> None:
        # Both legs of the transcript in the container log, the only debugging
        # artifact the bench keeps.
        item = event.item
        text = getattr(item, "text_content", None)
        if text:
            who = "rory_said" if item.role == "assistant" else "caller_said"
            logger.info(f"[worker] {who}: {text}")

    await session.start(agent=RoryAgent(), room=ctx.room)
    logger.info(f"[worker] session started room={ctx.room.name}")


# ---------------------------------------------------------------------------
# The bridge: one voice_ws connection ↔ one LiveKit room
# ---------------------------------------------------------------------------


def _mint_bridge_token(room_name: str, identity: str) -> str:
    grant = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grant)
        .to_jwt()
    )


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """Bridge one Veris ``voice_ws`` connection into a LiveKit room.

    The WebSocket is already accepted by the FastAPI route. Each connection
    gets its own room (``veris-<rand>``); the bridge joins as ``veris-actor``,
    publishes the actor's PCM16 as a mic track — which is what gets the worker
    dispatched in — and forwards the agent's audio back, re-sliced to 20 ms
    frames. Blocks until either side hangs up.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    room_name = f"veris-{secrets.token_hex(4)}"
    identity = "veris-actor"
    logger.info(f"[voice] actor connected peer={peer} room={room_name} url={LIVEKIT_URL}")

    room = rtc.Room()
    agent_audio: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

    @room.on("participant_connected")
    def _on_participant(participant: rtc.RemoteParticipant) -> None:
        logger.info(f"[voice] agent participant joined identity={participant.identity}")

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        logger.info(f"[voice] track_subscribed kind={track.kind} from={participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and not agent_audio.done():
            agent_audio.set_result(track)

    t_start = time.monotonic()
    try:
        await room.connect(LIVEKIT_URL, _mint_bridge_token(room_name, identity))
        logger.info(f"[voice] connected to room={room_name}")

        source = rtc.AudioSource(SAMPLE_RATE_HZ, NUM_CHANNELS)
        mic_track = rtc.LocalAudioTrack.create_audio_track("veris-mic", source)
        publication = await room.local_participant.publish_track(
            mic_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info(f"[voice] published mic track sid={publication.sid}")

        up = asyncio.create_task(_pump_actor_to_room(actor_ws, source), name="actor->room")
        down = asyncio.create_task(_pump_room_to_actor(actor_ws, agent_audio), name="room->actor")
        try:
            done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (up, down):
                if not task.done():
                    task.cancel()
            await asyncio.gather(up, down, return_exceptions=True)
    finally:
        await room.disconnect()
        logger.info(
            f"[voice] bridge exit room={room_name} duration={time.monotonic() - t_start:.1f}s"
        )


async def _pump_actor_to_room(actor_ws: WebSocket, source: rtc.AudioSource) -> None:
    """Binary PCM16 frames from the actor → the bridge's mic track.

    Frames are passed through as-is — ``AudioSource`` accepts variable-size
    frames, and the room IO on the worker side resamples for Deepgram.
    """
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            data = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(data)
            samples = len(data) // (2 * NUM_CHANNELS)
            await source.capture_frame(rtc.AudioFrame(data, SAMPLE_RATE_HZ, NUM_CHANNELS, samples))
            if n_frames == 1:
                logger.info(f"[a->r] first frame bytes={len(data)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->r] forwarded {n_frames} frames ({n_bytes} bytes)")
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->r] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->r] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_room_to_actor(actor_ws: WebSocket, agent_audio: asyncio.Future[rtc.Track]) -> None:
    """The agent's audio track → 24 kHz PCM16 frames back to the actor.

    ``AudioStream`` emits 10 ms frames (480 bytes at 24 kHz mono); the actor
    expects 20 ms, so the stream is buffered and re-sliced to ``FRAME_BYTES``
    boundaries. The pump ends when the actor hangs up or the track goes away —
    the worker's session closing unpublishes it, which is how an agent-side
    hang-up reaches the actor.
    """
    try:
        track = await asyncio.wait_for(asyncio.shield(agent_audio), AGENT_AUDIO_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"agent audio track never subscribed within {AGENT_AUDIO_TIMEOUT_S}s — "
            "was the worker dispatched into the room?"
        ) from None

    stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE_HZ, num_channels=NUM_CHANNELS)
    buf = bytearray()
    n_frames = 0
    n_bytes = 0
    in_frames = 0
    try:
        async for event in stream:
            data = bytes(event.frame.data)
            if in_frames == 0:
                logger.info(
                    f"[r->a] first AudioStream frame bytes={len(data)} "
                    f"(re-slicing to {FRAME_BYTES}-byte frames)"
                )
            in_frames += 1
            buf.extend(data)
            while len(buf) >= FRAME_BYTES:
                chunk = bytes(buf[:FRAME_BYTES])
                del buf[:FRAME_BYTES]
                await actor_ws.send_bytes(chunk)
                n_frames += 1
                n_bytes += len(chunk)
                if n_frames == 1:
                    logger.info("[r->a] emitted first 20 ms frame to actor")
                elif n_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[r->a] forwarded {n_frames} 20-ms frames ({n_bytes} bytes) "
                        f"from {in_frames} AudioStream frames"
                    )
        if buf:
            pad = FRAME_BYTES - len(buf)
            await actor_ws.send_bytes(bytes(buf) + bytes(pad))
            logger.info(f"[r->a] flushed final partial frame ({len(buf)} real bytes + {pad} silence)")
        logger.info(f"[r->a] agent track ended after {n_frames} frames ({n_bytes} bytes)")
    except WebSocketDisconnect:
        logger.info(f"[r->a] actor WS closed during pump ({n_frames} frames, {n_bytes} bytes)")
    finally:
        await stream.aclose()


if __name__ == "__main__":
    cli.run_app(server)
