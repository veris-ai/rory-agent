"""Rory on the Fluxions voice-agent API — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Fluxions ``/v1/realtime`` websocket,
declares Rory's 16 tools as bring-your-own functions in ``session.update``, and
bridges audio and tool calls in both directions. Fluxions runs the whole voice
loop server-side — ASR, an end-of-utterance model, a vLLM-served chat model,
VUI TTS, barge-in — and asks this process to run a function when the agent
decides it needs one. The agent speaks first.

Everything below the transport is shared with the other Rory candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. The candidates exist to be compared, so the transport is the only
thing allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. Fluxions
takes the caller at 16 kHz and speaks at 24 kHz, so only the inbound leg is
resampled; agent audio goes back to the actor untouched.
"""

from __future__ import annotations

import asyncio
import audioop
import io
import json
import os
import time
import wave
from contextlib import asynccontextmanager

import httpx
import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context
from rory_tools.session import require_credentials

from .tools import TOOLS


# A voice id from GET /voices.
VOICE = os.environ.get("FLUXIONS_VOICE", "maeve")

# The actor's rate both ways, and Fluxions's output rate. Fluxions listens at
# 16 kHz — strict, and audio at any other rate transcribes badly rather than
# being rejected — so caller audio is downsampled on the way in.
ACTOR_RATE_HZ = 24000
MIC_RATE_HZ = 16000

# Fluxions delivers agent audio ~6× faster than realtime. Everything sent to
# the actor is already committed — nothing claws it back — so audio is paced
# out no more than this far ahead of playback. That is what makes the server's
# `audio.flush` on barge-in actually cut the reply, instead of the actor
# playing out seconds it already holds.
PLAYBACK_LEAD_S = 1.0

# The greeting is one rendered buffer; it is sliced this fine so the pacer's
# lead applies to it too.
PLAYBACK_CHUNK_S = 0.5

# How often to tell Fluxions how much of its audio the caller has heard
# (`playback.pos`), which is what its barge-in timing runs on.
PLAYBACK_POS_S = 0.25

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()

_host: str  # FLUXIONS_HOST, required; the realtime socket and the REST paths hang off it
_greeting_pcm: bytes
_token: str | None  # FLUXIONS_API_KEY when the host requires auth, else None


def _soul() -> str:
    """The system prompt Fluxions runs with.

    The date context is appended rather than baked into ``agent_desc.txt``
    because it is resolved per call, off the frozen clock the attempt restored
    with its snapshot.
    """
    return f"{AGENT_PROMPT}\n\n{today_context()}"


def _session_update() -> dict:
    """The session config, sent as the first message on the realtime socket.

    Fluxions builds the session from this at connect; anything sent later is
    ignored, so every knob is pinned here. The credential rides as `token`
    only on a host that asked for one (see ``vendors``). The opening line is not left to the
    model (`greet: false`): the shared greeting is spoken from a rendered
    buffer, like Pipecat's, and the prompt already tells the model it happened.
    `echo_guard` is off because the actor injects PCM — there is no acoustic
    loopback to guard against. The agent may not end the call itself, and its
    idle check-in ("still there?") and idle hangup — on by default at ~15 s of
    caller silence — are pushed out to an hour: no other candidate prompts or
    hangs up on a quiet caller, and a scored call ends when the caller's side
    does.
    """
    update = {
        "type": "session.update",
        "soul": _soul(),
        "voice": VOICE,
        "tools": TOOLS,
        "greet": False,
        "echo_guard": False,
        "allow_hangup": False,
        "idle_check_s": 3600,
        "idle_hangup_s": 3600,
    }
    if _token:
        update["token"] = _token
    return update


def _pcm_from_wav(data: bytes) -> bytes:
    """Frames of a mono s16le WAV at the actor's rate, or a loud failure."""
    with wave.open(io.BytesIO(data)) as wav:
        fmt = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        if fmt != (1, 2, ACTOR_RATE_HZ):
            raise RuntimeError(
                f"Fluxions TTS returned (channels, width, rate)={fmt}, "
                f"expected (1, 2, {ACTOR_RATE_HZ})"
            )
        return wav.readframes(wav.getnframes())


@asynccontextmanager
async def vendors():
    """Check the deployment can carry Rory, then render the greeting once.

    Entered from the web lifespan, so the bench's readiness probe only passes
    once a call can actually be served. The host's `GET /config` says whether
    it takes credentials: when `auth_required` is set, `FLUXIONS_API_KEY` is
    required and rides as a bearer on REST and as `token` on the socket; when
    it is not, nothing is sent — a deployment that runs anonymous rejects any
    bearer it did not issue, so an unneeded key is not merely ignored, it is a
    401 on the first authenticated route. Two more things fail the boot here
    rather than every call: a soul longer than the host's `max_soul_chars`,
    which it would otherwise truncate silently and run Rory on a fraction of
    the policy; and a voice the host does not have. The greeting is the same
    bytes for every call, so it is rendered once — which also wakes the TTS
    tier before the first caller dials.
    """
    global _host, _greeting_pcm, _token
    _host = os.environ["FLUXIONS_HOST"]
    async with httpx.AsyncClient(
        base_url=f"https://{_host}",
        # Generous enough to sit through a cold start being held at the gateway.
        timeout=180.0,
    ) as http:
        config = (await http.get("/config")).raise_for_status().json()
        if config["auth_required"]:
            require_credentials("FLUXIONS_API_KEY")
            _token = os.environ["FLUXIONS_API_KEY"]
            http.headers["Authorization"] = f"Bearer {_token}"
        else:
            _token = None
        soul_chars = len(_soul())
        if soul_chars > config["max_soul_chars"]:
            raise RuntimeError(
                f"Rory's prompt is {soul_chars} chars but {_host} caps the "
                f"soul at {config['max_soul_chars']} — the host would truncate it silently"
            )
        voices = [v["voice_id"] for v in (await http.get("/voices")).raise_for_status().json()["voices"]]
        if VOICE not in voices:
            raise RuntimeError(
                f"voice {VOICE!r} is not on {_host}; available: {', '.join(sorted(voices))}"
            )
        t0 = time.monotonic()
        resp = await http.post("/v1/tts", json={"voice": VOICE, "input": GREETING})
        _greeting_pcm = _pcm_from_wav(resp.raise_for_status().content)
        logger.info(
            f"[startup] fluxions host={_host} model={config['model']} voice={VOICE} "
            f"auth={'token' if _token else 'anonymous'} "
            f"soul={soul_chars}/{config['max_soul_chars']} chars; greeting rendered "
            f"({len(_greeting_pcm) / 2 / ACTOR_RATE_HZ:.1f}s) in {time.monotonic() - t0:.1f}s"
        )
    yield


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Fluxions realtime session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other candidates
    build one per call: it holds the account ``verify_caller`` matched and dies
    with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")

    t_start = time.monotonic()
    try:
        async with websockets.connect(f"wss://{_host}/v1/realtime", max_size=None) as fx_ws:
            await fx_ws.send(json.dumps(_session_update()))
            logger.info(
                f"[voice] Fluxions connected host={_host} voice={VOICE}; "
                f"session.update sent: {len(TOOLS)} tools, soul={len(AGENT_PROMPT)} chars"
            )
            await _Call(actor_ws, fx_ws).run()
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


class _Call:
    """The bridge for one call: caller audio up, agent audio and tool calls down.

    Four tasks run against this state: the actor pump (resample and forward),
    the Fluxions receive loop (events, tool dispatch, audio into the playback
    queue), the playback pacer (queue → actor at ~realtime), and the
    ``playback.pos`` ticker. The call ends when either socket does or the
    agent hangs up.
    """

    def __init__(self, actor_ws: WebSocket, fx_ws) -> None:
        self.actor_ws = actor_ws
        self.fx_ws = fx_ws
        self.session = CallSession()
        self._playback: asyncio.Queue[bytes] = asyncio.Queue()
        self._flushes = 0          # bumped by audio.flush; the pacer drops what it was holding
        self._sent_s = 0.0         # cumulative audio handed to the actor
        self._play_until = 0.0     # monotonic time the actor's buffer runs dry
        self._greeting_s = len(_greeting_pcm) / 2 / ACTOR_RATE_HZ

    async def run(self) -> None:
        # Rory speaks first. Queued before the pumps start so it is the first
        # audio out, through the same pacer as the agent's own replies.
        chunk = int(PLAYBACK_CHUNK_S * ACTOR_RATE_HZ) * 2
        for i in range(0, len(_greeting_pcm), chunk):
            self._playback.put_nowait(_greeting_pcm[i:i + chunk])
        logger.info("[voice] greeting queued (agent greets first)")

        tasks = [
            asyncio.create_task(self._pump_actor(), name="actor->fx"),
            asyncio.create_task(self._pump_fluxions(), name="fx->actor"),
            asyncio.create_task(self._pump_playback(), name="playback"),
            asyncio.create_task(self._report_playback(), name="playback.pos"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()

    # -- caller → Fluxions ---------------------------------------------------

    async def _pump_actor(self) -> None:
        """Binary PCM16 frames from the actor (24 kHz) → Fluxions mic (16 kHz).

        Forwarded frame by frame as they arrive, silence included: the actor
        streams at realtime, and Fluxions's endpointing needs that continuous
        timeline to hear the caller stop.
        """
        n_frames = 0
        n_bytes = 0
        resample_state = None  # carried across ratecv calls so the resample is continuous
        try:
            while True:
                frame = await self.actor_ws.receive_bytes()
                n_frames += 1
                n_bytes += len(frame)
                pcm16k, resample_state = audioop.ratecv(
                    frame, 2, 1, ACTOR_RATE_HZ, MIC_RATE_HZ, resample_state
                )
                if n_frames == 1:
                    logger.info(f"[a->fx] first frame bytes={len(frame)} (->{len(pcm16k)} @16kHz)")
                elif n_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(f"[a->fx] forwarded {n_frames} frames ({n_bytes} bytes @24kHz)")
                await self.fx_ws.send(pcm16k)
        except WebSocketDisconnect as exc:
            logger.info(
                f"[a->fx] actor disconnected after {n_frames} frames "
                f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
            )
        except KeyError as exc:
            # Starlette raises KeyError('bytes') when it got a text frame instead of
            # a binary one — a protocol mismatch on the actor side, not a bug here.
            logger.error(
                f"[a->fx] non-binary frame after {n_frames} binary frames — "
                f"actor protocol mismatch? ({exc})"
            )
            raise

    # -- Fluxions → caller ---------------------------------------------------

    async def _pump_fluxions(self) -> None:
        """Fluxions events → playback queue and tool dispatch.

        Binary frames are agent speech and go to the pacer. ``tool.call`` is
        answered inline, which keeps tool calls sequential — they share account
        state — and blocks nothing the caller can hear, since playback runs on
        its own task. Returns when the agent hangs up or Fluxions closes.
        """
        n_audio_chunks = 0
        n_audio_bytes = 0
        try:
            async for msg in self.fx_ws:
                if isinstance(msg, bytes):
                    n_audio_chunks += 1
                    n_audio_bytes += len(msg)
                    if n_audio_chunks == 1:
                        logger.info(f"[fx->a] first audio chunk bytes={len(msg)}")
                    elif n_audio_chunks % LOG_EVERY_N_FRAMES == 0:
                        logger.info(
                            f"[fx->a] received {n_audio_chunks} audio chunks ({n_audio_bytes} bytes)"
                        )
                    self._playback.put_nowait(msg)
                    continue

                event = json.loads(msg)
                etype = event["type"]

                if etype == "session.created":
                    rates = (event["mic_sample_rate"], event["audio_sample_rate"])
                    if rates != (MIC_RATE_HZ, ACTOR_RATE_HZ):
                        raise RuntimeError(
                            f"Fluxions session rates {rates}, expected ({MIC_RATE_HZ}, {ACTOR_RATE_HZ})"
                        )
                    logger.info(
                        f"[fx->a] session.created id={event['session_id']} "
                        f"model={event['model']} voice={event['voice']}"
                    )

                elif etype == "tool.call":
                    await self._run_tool(event)

                elif etype == "audio.flush":
                    # Barge-in: drop what the caller has not heard yet. What the
                    # pacer already sent — at most PLAYBACK_LEAD_S — still plays.
                    dropped = 0
                    while not self._playback.empty():
                        dropped += len(self._playback.get_nowait())
                    self._flushes += 1
                    logger.info(
                        f"[fx->a] audio.flush — dropped {dropped / 2 / ACTOR_RATE_HZ:.1f}s queued audio"
                    )

                elif etype == "user.transcript":
                    logger.info(f"[fx->a] caller_said: {event['text']}")

                elif etype == "agent.sentence":
                    logger.info(f"[fx->a] rory_said (#{event['idx']}): {event['text']}")

                elif etype == "agent.done":
                    logger.info(
                        f"[fx->a] agent.done sentences={event['n_sentences']} "
                        f"cancelled={event['cancelled']} llm_ttft_ms={event['llm_ttft_ms']} "
                        f"tts_ttfb_ms={event['tts_ttfb_ms']}"
                    )

                elif etype in ("vad", "vad_start", "barge_in"):
                    logger.info(f"[fx->a] {etype}: {event.get('state') or event.get('reason', '')}")

                elif etype == "agent.hangup":
                    logger.info("[fx->a] agent.hangup — the agent ended the call")
                    return

                elif etype == "error":
                    logger.error(f"[fx->a] Fluxions error: {event}")
                    raise RuntimeError(f"Fluxions error: {event.get('message')}")

                elif etype != "user.partial":
                    logger.info(f"[fx->a] unhandled event: {json.dumps(event)}")
        except websockets.ConnectionClosed as exc:
            logger.info(
                f"[fx->a] Fluxions WS closed code={exc.code} reason={exc.reason!r}; "
                f"{n_audio_bytes} audio bytes"
            )

    async def _run_tool(self, event: dict) -> None:
        """Dispatch one tool call and answer it.

        Fluxions gives the result 12 s to land before it tells the caller the
        action did not happen. Every vendor client is blocking, and this
        coroutine shares the loop with the actor pump — called inline, a
        ten-second vendor timeout is ten seconds of dead air on the mic leg,
        the same reason the other candidates hop threads.
        """
        name = event["name"]
        args = event["arguments"]
        logger.info(f"[tool] {name} ({event['call_id']}) args={args}")
        t0 = time.monotonic()
        result = await asyncio.to_thread(dispatch, self.session, name, args)
        logger.info(
            f"[tool] {name} -> {json.dumps(result, default=str)} in {time.monotonic() - t0:.2f}s"
        )
        await self.fx_ws.send(json.dumps({
            "type": "tool.result",
            "call_id": event["call_id"],
            "output": json.dumps(result, default=str),
        }))

    # -- playback ------------------------------------------------------------

    def _played_s(self, now: float) -> float:
        """Seconds of audio the actor has played by ``now``, greeting included."""
        return self._sent_s - max(0.0, self._play_until - now)

    async def _pump_playback(self) -> None:
        """Queue → actor, never more than PLAYBACK_LEAD_S ahead of playback.

        The actor plays what it is given at realtime; this models that buffer
        so the lead can be held and ``playback.pos`` reported honestly. A chunk
        taken before a flush and released after it belongs to the cancelled
        reply, so it is dropped rather than sent.
        """
        while True:
            chunk = await self._playback.get()
            flushes = self._flushes
            ahead = self._play_until - time.monotonic()
            if ahead > PLAYBACK_LEAD_S:
                await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
                if self._flushes != flushes:
                    continue
            await self.actor_ws.send_bytes(chunk)
            now = time.monotonic()
            seconds = len(chunk) / 2 / ACTOR_RATE_HZ
            self._play_until = max(now, self._play_until) + seconds
            self._sent_s += seconds

    async def _report_playback(self) -> None:
        """Tell Fluxions how much of *its* audio has been heard.

        The greeting is not Fluxions's audio — it was rendered here — so it is
        subtracted; the server reconciles the position against what it sent.
        """
        reported = None
        while True:
            await asyncio.sleep(PLAYBACK_POS_S)
            played = round(max(0.0, self._played_s(time.monotonic()) - self._greeting_s), 3)
            if played != reported:
                reported = played
                await self.fx_ws.send(json.dumps({"type": "playback.pos", "played_s": played}))
