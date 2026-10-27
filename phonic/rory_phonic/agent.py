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

from .tools import BUILT_IN_TOOLS, TOOLS

PHONIC_URL = "wss://api.phonic.ai/v1/sts/ws"
PHONIC_VOICE = os.environ.get("PHONIC_VOICE", "sabrina")
PHONIC_MODEL = os.environ.get("PHONIC_MODEL", "phonic-1")

# The Veris actor and Phonic both run PCM16 at 24 kHz.
AUDIO_FORMAT = "pcm_24000"

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

# The close code of a conversation that ended on purpose. Any other close from
# Phonic is a failure and must reach the caller of the handler as one.
NORMAL_CLOSE = 1000
# RFC 6455: the connection closed without a close frame.
CLOSED_WITHOUT_FRAME = 1006

AGENT_PROMPT = load_agent_prompt()


def _close(exc: websockets.ConnectionClosed) -> tuple[int, str]:
    """The close code and reason Phonic sent, or 1006 when it sent no close frame."""
    if exc.rcvd is None:
        return CLOSED_WITHOUT_FRAME, ""
    return exc.rcvd.code, exc.rcvd.reason


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
        "phonic_model": PHONIC_MODEL,
        "intelligence_level": "high",
        "input_format": AUDIO_FORMAT,
        "output_format": AUDIO_FORMAT,
        # Turn-taking parity with the other candidates: 800 ms of end-of-turn
        # silence. Phonic's other VAD knobs stay on its defaults.
        "vad_min_silence_duration_ms": 800,
        # The shared 16 tools, then Phonic's built-in turn-taking tool. The built-in
        # is Phonic's addition, not one of the shared tools, so it is listed here and
        # kept out of the parity surface in tools.py.
        "tools": TOOLS + BUILT_IN_TOOLS,
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

            tool_calls: asyncio.Queue = asyncio.Queue()
            interrupted: set[str] = set()
            tasks = [
                asyncio.create_task(_pump_actor_to_phonic(actor_ws, ph_ws), name="actor->phonic"),
                asyncio.create_task(
                    _pump_phonic_to_actor(ph_ws, actor_ws, tool_calls, interrupted), name="phonic->actor"
                ),
                asyncio.create_task(
                    _run_tool_calls(ph_ws, session_state, tool_calls, interrupted), name="tool-calls"
                ),
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_phonic(actor_ws: WebSocket, ph_ws) -> None:
    """Binary PCM16 frames from the actor → Phonic audio_chunk messages.

    The actor hanging up is the call ending. Phonic closing it is a failure unless
    it closed normally.
    """
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
    except websockets.ConnectionClosed as exc:
        code, _ = _close(exc)
        if code != NORMAL_CLOSE:
            raise
        logger.info(f"[a->ph] Phonic closed normally during the call after {n_frames} frames")
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->ph] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_phonic_to_actor(
    ph_ws, actor_ws: WebSocket, tool_calls: asyncio.Queue, interrupted: set[str]
) -> None:
    """Phonic messages → audio bytes back to the actor; queue tool calls for the worker.

    This loop never waits on a tool, so audio and interruptions keep flowing while
    a vendor call runs. A Phonic close with any code but ``NORMAL_CLOSE`` raises.
    """
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
                await tool_calls.put(evt)

            elif etype == "tool_call_interrupted":
                # Phonic cut the call off; its output would be stale, so the worker
                # drops it instead of sending it.
                interrupted.add(evt["tool_call_id"])
                logger.warning(f"[ph->a] tool {evt['tool_name']} ({evt['tool_call_id']}) interrupted")

            elif etype == "conversation_created":
                logger.info(f"[ph->a] conversation_created id={evt['conversation_id']}")

            elif etype == "assistant_ended_conversation":
                logger.info("[ph->a] assistant ended the conversation")

            elif etype == "assistant_chose_not_to_respond":
                logger.info("[ph->a] assistant chose not to respond this turn")

            elif etype == "warning":
                logger.warning(f"[ph->a] Phonic warning: {evt['warning']}")

            elif etype == "error":
                raise RuntimeError(f"Phonic realtime error: {evt['error']}")

    except websockets.ConnectionClosed as exc:
        code, reason = _close(exc)
        if code != NORMAL_CLOSE:
            logger.error(
                f"[ph->a] Phonic WS closed abnormally code={code} reason={reason}; "
                f"{n_audio_frames} audio chunks, {n_audio_bytes} audio bytes"
            )
            raise
        logger.info(
            f"[ph->a] Phonic WS closed normally; {n_audio_frames} audio chunks, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        logger.info("[ph->a] actor WS closed; ending Phonic receive loop")


async def _run_tool_calls(
    ph_ws, session_state: CallSession, tool_calls: asyncio.Queue, interrupted: set[str]
) -> None:
    """Run Phonic's tool calls one at a time and send each output back.

    The vendor clients are blocking and share account state, so calls run in order
    on a worker thread. The receive loop keeps consuming Phonic events meanwhile.
    """
    while True:
        evt = await tool_calls.get()
        try:
            await _run_tool_call(ph_ws, session_state, evt, interrupted)
        finally:
            tool_calls.task_done()


async def _run_tool_call(ph_ws, session_state: CallSession, evt: dict, interrupted: set[str]) -> None:
    """Run one tool call and send its output, unless Phonic interrupted it."""
    call_id = evt["tool_call_id"]
    name = evt["tool_name"]
    if call_id in interrupted:
        logger.info(f"[ph->a] tool {name} ({call_id}) was interrupted before it ran; not run")
        return
    result = await asyncio.to_thread(dispatch, session_state, name, evt["parameters"])
    logger.info(f"[ph->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}")
    if call_id in interrupted:
        logger.info(f"[ph->a] tool {name} ({call_id}) interrupted while running; output dropped")
        return
    await ph_ws.send(json.dumps({
        "type": "tool_call_output",
        "tool_call_id": call_id,
        "output": result,
    }))
