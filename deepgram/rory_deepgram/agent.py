"""Rory on Deepgram's Voice Agent API — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Voice Agent WebSocket session
(``wss://agent.deepgram.com/v1/agent/converse``), sends the Settings message
carrying Rory's prompt, the 16 tools, the STT/LLM/TTS providers and the
greeting, and bridges audio and function calls in both directions. The agent
speaks first via ``agent.greeting``.

Like Gemini Live and unlike the Pipecat cascade, there is no pipeline to
assemble here: Deepgram runs STT, LLM and TTS server-side inside one session.
What this module adds is the ``voice_ws`` bridging, the per-call
``CallSession``, and the hop onto a worker thread for the blocking vendor calls.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher.
The candidates exist to be compared, so the transport is the only thing
allowed to differ.

Audio is raw binary in both directions — the Voice Agent API takes PCM16 on
the socket with no JSON envelope and no base64, and emits its TTS the same way.
Both legs are configured at the actor's 24 kHz, so this is a byte passthrough
with no resampling and no re-framing. The socket is mixed: binary frames are
audio, text frames are the JSON event stream.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import FUNCTIONS

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# The Veris actor speaks and listens at 24 kHz; Deepgram accepts and emits
# linear16 at that rate, so neither leg is resampled.
SAMPLE_RATE_HZ = 24000

# Speech-to-text. nova-3 on the v1 provider: the same model the Pipecat
# cascade runs for STT. v1 exposes no endpointing control through Voice Agent
# Settings — turn-taking is Deepgram's built-in, not the ~0.8 s end-of-turn
# silence the Pipecat and Gemini Live candidates pin.
DEEPGRAM_LISTEN_MODEL = os.environ.get("DEEPGRAM_LISTEN_MODEL", "nova-3")
# Text-to-speech. Aura-2 is Deepgram's current-generation voice line.
DEEPGRAM_VOICE = os.environ.get("DEEPGRAM_VOICE", "aura-2-thalia-en")
# The LLM runs on Deepgram's managed OpenAI access — ``think.provider.type:
# open_ai`` with no endpoint needs no key of our own — so gpt-4.1-mini matches
# the Pipecat candidate while DEEPGRAM_API_KEY stays the only vendor credential.
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _settings() -> dict:
    """The Settings message — the whole session config, sent once on connect.

    The date context is appended to the prompt rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock
    the attempt restored with its snapshot.
    """
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": SAMPLE_RATE_HZ},
            # container "none" keeps the downstream bytes raw PCM16, which is
            # exactly what the actor's voice_ws expects; anything else would
            # wrap them in a header the actor would play as noise.
            "output": {
                "encoding": "linear16",
                "sample_rate": SAMPLE_RATE_HZ,
                "container": "none",
            },
        },
        "agent": {
            "listen": {"provider": {"type": "deepgram", "model": DEEPGRAM_LISTEN_MODEL}},
            "think": {
                "provider": {"type": "open_ai", "model": LLM_MODEL},
                "prompt": f"{AGENT_PROMPT}\n\n{today_context()}",
                # No ``endpoint`` on any entry — that is what makes these
                # client-side, so calls come back here as FunctionCallRequest.
                "functions": FUNCTIONS,
            },
            "speak": {"provider": {"type": "deepgram", "model": DEEPGRAM_VOICE}},
            # Spoken at session start — the agent greets first, straight to TTS
            # without a round trip through the LLM, so the wording is verbatim.
            "greeting": GREETING,
        },
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Deepgram Voice Agent session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other agents build
    one per call: it holds the account ``verify_caller`` matched and dies with
    the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}

    t_start = time.monotonic()
    try:
        async with websockets.connect(DEEPGRAM_AGENT_URL, additional_headers=headers) as dg_ws:
            logger.info(
                f"[voice] Deepgram Voice Agent connected listen={DEEPGRAM_LISTEN_MODEL} "
                f"think={LLM_MODEL} speak={DEEPGRAM_VOICE} rate={SAMPLE_RATE_HZ}Hz"
            )
            await dg_ws.send(json.dumps(_settings()))
            logger.info(
                f"[voice] Settings sent: {len(FUNCTIONS)} functions, "
                f"prompt={len(AGENT_PROMPT)} chars, greeting={GREETING!r}"
            )

            # Actor audio is gated on SettingsApplied — audio sent before the
            # server has applied Settings is discarded.
            ready = asyncio.Event()

            up = asyncio.create_task(_pump_actor_to_dg(actor_ws, dg_ws, ready), name="actor->dg")
            down = asyncio.create_task(
                _pump_dg_to_actor(dg_ws, actor_ws, session_state, ready), name="dg->actor"
            )
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
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_dg(actor_ws: WebSocket, dg_ws, ready: asyncio.Event) -> None:
    """Binary PCM16 frames from the actor → the Voice Agent socket, verbatim.

    Frames that arrive before SettingsApplied are dropped: they are silence,
    since the actor only speaks after it hears the greeting.
    """
    n_frames = 0
    n_bytes = 0
    n_dropped = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            if not ready.is_set():
                n_dropped += 1
                continue
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(
                    f"[a->dg] first frame forwarded bytes={len(frame)} "
                    f"({n_dropped} pre-ready frames dropped)"
                )
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->dg] forwarded {n_frames} frames ({n_bytes} bytes)")
            await dg_ws.send(frame)
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->dg] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->dg] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_dg_to_actor(
    dg_ws, actor_ws: WebSocket, session_state: CallSession, ready: asyncio.Event
) -> None:
    """Voice Agent messages → audio bytes back to the actor; run function calls.

    Binary frames are TTS audio and go straight through to the actor as they
    arrive; text frames are the JSON event stream.
    """
    n_audio_frames = 0
    n_audio_bytes = 0
    n_turns = 0
    latency: dict = {}  # partial LatencyReport fields for the in-flight turn
    try:
        async for raw in dg_ws:
            if isinstance(raw, bytes):
                n_audio_frames += 1
                n_audio_bytes += len(raw)
                if n_audio_frames == 1:
                    logger.info(f"[dg->a] first audio chunk bytes={len(raw)}")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[dg->a] forwarded {n_audio_frames} audio chunks ({n_audio_bytes} bytes)"
                    )
                await actor_ws.send_bytes(raw)
                continue

            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "Welcome":
                logger.info(f"[dg->a] Welcome request_id={evt.get('request_id')}")

            elif etype == "SettingsApplied":
                ready.set()
                logger.info("[dg->a] SettingsApplied — actor audio now forwarding")

            elif etype == "ConversationText":
                role = evt.get("role", "?")
                text = (evt.get("content") or "").strip()
                who = "rory_said" if role == "assistant" else "caller_said"
                logger.info(f"[dg->a] {who}: {text}")

            elif etype == "UserStartedSpeaking":
                # Barge-in. Audio is forwarded frame-by-frame with no buffer
                # here, so there is nothing of ours to flush — Deepgram simply
                # stops producing.
                logger.info("[dg->a] UserStartedSpeaking (barge-in)")

            elif etype == "LatencyReport":
                # Deepgram's own timing for the turn. Each report carries a
                # single field, so they accumulate until total_latency arrives
                # and closes the turn out.
                latency.update({k: v for k, v in evt.items() if k != "type"})
                if "total_latency" in latency:
                    n_turns += 1
                    logger.info(
                        f"[dg->a] turn #{n_turns} latency total={latency['total_latency']:.2f}s "
                        f"ttt={latency.get('ttt_text_latency', 0.0):.2f}s "
                        f"tts={latency.get('tts_latency', 0.0):.2f}s"
                    )
                    latency = {}

            elif etype == "AgentAudioDone":
                logger.info(
                    f"[dg->a] AgentAudioDone ({n_audio_frames} audio chunks / "
                    f"{n_audio_bytes} bytes this session)"
                )

            elif etype == "FunctionCallRequest":
                await _handle_function_calls(dg_ws, session_state, evt.get("functions") or [])

            elif etype == "Error":
                logger.error(
                    f"[dg->a] Error code={evt.get('code')} description={evt.get('description')}"
                )
                raise RuntimeError(f"Deepgram Voice Agent error: {evt.get('description')}")

            elif etype == "Warning":
                logger.warning(
                    f"[dg->a] Warning code={evt.get('code')} description={evt.get('description')}"
                )
    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[dg->a] Deepgram WS closed code={exc.code} reason={exc.reason!r}; "
            f"{n_turns} turns, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        logger.info("[dg->a] actor WS closed; ending Deepgram receive loop")


async def _handle_function_calls(dg_ws, session_state: CallSession, functions: list[dict]) -> None:
    """Run each client-side call and send its FunctionCallResponse back.

    One request can carry several calls; they run in order because they share
    verification and payment state. Server-side entries (``client_side:
    false``) are informational — Deepgram runs those itself — so they are
    logged and skipped; every function this agent declares is client-side.
    """
    for fn in functions:
        name = fn.get("name", "")
        call_id = fn.get("id", "")
        if not fn.get("client_side", True):
            logger.info(f"[dg->a] server-side call {name} — Deepgram runs it, skipping")
            continue
        args = json.loads(fn.get("arguments") or "{}")
        # Every vendor client is blocking, and this coroutine is carrying the
        # caller's audio. Called inline, a ten-second vendor timeout is ten
        # seconds of dead air — the same reason the other candidates hop threads.
        result = await asyncio.to_thread(dispatch, session_state, name, args)
        logger.info(f"[dg->a] tool {name} -> {json.dumps(result, default=str)}")
        await dg_ws.send(
            json.dumps(
                {
                    "type": "FunctionCallResponse",
                    "id": call_id,
                    "name": name,
                    # ``content`` is a string — the model reads it as the call's result.
                    "content": json.dumps(result, default=str),
                }
            )
        )
