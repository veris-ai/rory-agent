"""Rory on Phonic's speech-to-speech API — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Phonic conversation websocket, sends one
``config`` carrying the shared prompt, greeting and Rory's 16 tools, and bridges
audio and tool calls in both directions. Phonic speaks the greeting first.

Phonic is a speech-to-speech model over its own websocket protocol, not an
OpenAI-Realtime-compatible one. Its messages are ``audio_chunk`` (base64 audio
plus its text), ``input_text`` (the caller's transcript), ``tool_call`` and
``tool_call_output`` for tools.

Everything below the transport is shared with the other candidates: same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. The transport is the only thing allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and Phonic
takes and emits ``pcm_24000``, so neither leg is resampled.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools import clock
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

PHONIC_URL = "wss://api.phonic.ai/v1/sts/ws"
PHONIC_VOICE = os.environ.get("PHONIC_VOICE", "sabrina")

# The Veris actor and Phonic both run PCM16 at 24 kHz.
AUDIO_FORMAT = "pcm_24000"

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _config() -> dict:
    """The conversation config, pinned so a run is reproducible.

    The date context is appended to the system prompt rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot.
    """
    return {
        "type": "config",
        "system_prompt": f"{AGENT_PROMPT}\n\n{today_context()}",
        "welcome_message": GREETING,
        "voice_id": PHONIC_VOICE,
        "input_format": AUDIO_FORMAT,
        "output_format": AUDIO_FORMAT,
        # Turn-taking parity with the other candidates: 800 ms of end-of-turn
        # silence. Phonic's other VAD knobs stay on its defaults.
        "vad_min_silence_duration_ms": 800,
        "tools": TOOLS,
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Phonic conversation with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other transports build
    one per call: it holds the account ``verify_caller`` matched and dies with
    the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    headers = {"Authorization": f"Bearer {os.environ['PHONIC_API_KEY']}"}

    t_start = time.monotonic()
    try:
        async with websockets.connect(PHONIC_URL, additional_headers=headers) as ph_ws:
            logger.info(f"[voice] Phonic websocket connected voice={PHONIC_VOICE} format={AUDIO_FORMAT}")
            # Phonic sends nothing until the config arrives; the config is the first message.
            await ph_ws.send(json.dumps(_config()))
            logger.info(f"[voice] config sent: {len(TOOLS)} tools, system_prompt={len(AGENT_PROMPT)} chars")
            ready = json.loads(await ph_ws.recv())
            if ready["type"] != "ready_to_start_conversation":
                raise RuntimeError(f"Phonic answered the config with {ready['type']}: {ready}")

            up = asyncio.create_task(_pump_actor_to_phonic(actor_ws, ph_ws), name="actor->phonic")
            down = asyncio.create_task(
                _pump_phonic_to_actor(ph_ws, actor_ws, session_state), name="phonic->actor"
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


async def _pump_actor_to_phonic(actor_ws: WebSocket, ph_ws) -> None:
    """Binary PCM16 frames from the actor → Phonic audio_chunk messages."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->ph] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->ph] forwarded {n_frames} frames ({n_bytes} bytes)")
            chunk = {
                "type": "audio_chunk",
                "audio": base64.b64encode(frame).decode(),
            }
            if n_frames == 1:
                # Phonic requires the call's start time on the first chunk only, as
                # UTC with millisecond precision (e.g. 2025-04-17T02:48:52.708Z).
                chunk["iso_date_time"] = clock.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            await ph_ws.send(json.dumps(chunk))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->ph] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->ph] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_phonic_to_actor(ph_ws, actor_ws: WebSocket, session_state: CallSession) -> None:
    """Phonic messages → audio bytes back to the actor; dispatch tool calls."""
    n_audio_frames = 0
    n_audio_bytes = 0

    try:
        async for raw in ph_ws:
            evt = json.loads(raw)
            etype = evt["type"]

            if etype == "audio_chunk":
                audio = base64.b64decode(evt["audio"])
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[ph->a] first audio chunk bytes={len(audio)}")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[ph->a] streamed {n_audio_frames} audio chunks "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if audio:
                    await actor_ws.send_bytes(audio)
                logger.info(f"[ph->a] rory_said: {evt['text']}")

            elif etype == "input_text":
                logger.info(f"[ph->a] caller_said: {evt['text']}")

            elif etype == "user_started_speaking":
                logger.info("[ph->a] vad: user_started_speaking")

            elif etype == "user_finished_speaking":
                logger.info("[ph->a] vad: user_finished_speaking")

            elif etype == "interrupted_response":
                logger.info(f"[ph->a] interrupted after: {evt['text']}")

            elif etype == "tool_call":
                name = evt["tool_name"]
                # Every vendor client is blocking, and this coroutine is carrying
                # the caller's audio. Called inline, a slow vendor call is dead
                # air — the same reason the other transports hop threads. Awaiting
                # here also keeps tool calls sequential: they share account state.
                result = await asyncio.to_thread(dispatch, session_state, name, evt["parameters"])
                logger.info(f"[ph->a] tool {name} ({evt['tool_call_id']}) -> {json.dumps(result, default=str)}")
                await ph_ws.send(json.dumps({
                    "type": "tool_call_output",
                    "tool_call_id": evt["tool_call_id"],
                    "output": result,
                }))

            elif etype == "tool_call_interrupted":
                logger.warning(f"[ph->a] tool {evt['tool_name']} ({evt['tool_call_id']}) interrupted")

            elif etype == "conversation_created":
                logger.info(f"[ph->a] conversation_created id={evt['conversation_id']}")

            elif etype == "assistant_ended_conversation":
                logger.info("[ph->a] assistant ended the conversation")

            elif etype == "warning":
                logger.warning(f"[ph->a] Phonic warning: {evt['warning']}")

            elif etype == "error":
                raise RuntimeError(f"Phonic realtime error: {evt['error']}")

    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[ph->a] Phonic WS closed code={exc.code} reason={exc.reason}; "
            f"{n_audio_frames} audio chunks, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        logger.info("[ph->a] actor WS closed; ending Phonic receive loop")
