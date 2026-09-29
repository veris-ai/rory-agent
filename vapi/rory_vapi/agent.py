"""Rory on Vapi — Acme Energy's voice support agent on a hosted orchestrator.

Vapi runs the whole voice loop on its side — the transcriber, the LLM, TTS
and turn-taking — for a call created over its API. What this module adds is
the call: an inline assistant carrying the shared prompt, opening line and
tool declarations, created with a WebSocket transport, and per ``/voice``
connection the bridge that pumps PCM16 between the Veris actor and the call's
socket. Tool calls do not travel over that socket; Vapi POSTs them to the
``/tool`` webhook in ``rory_vapi.web``, which resolves them through
``rory_tools.dispatch`` against the ``CallSession`` registered here.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher.
The candidates exist to be compared, so the transport is the only thing
allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and the
call is created with a 24 kHz ``pcm_s16le`` transport, so audio passes through
untouched in both directions.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict

import httpx
import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import build_tools

VAPI_API_URL = "https://api.vapi.ai/call"

VAPI_MODEL_PROVIDER = os.environ.get("VAPI_MODEL_PROVIDER", "openai")
VAPI_MODEL = os.environ.get("VAPI_MODEL", "gpt-4.1-mini")
VAPI_VOICE_PROVIDER = os.environ.get("VAPI_VOICE_PROVIDER", "11labs")
VAPI_VOICE_ID = os.environ.get("VAPI_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
VAPI_VOICE_MODEL = os.environ.get("VAPI_VOICE_MODEL", "eleven_flash_v2")
VAPI_TRANSCRIBER_PROVIDER = os.environ.get("VAPI_TRANSCRIBER_PROVIDER", "deepgram")
VAPI_TRANSCRIBER_MODEL = os.environ.get("VAPI_TRANSCRIBER_MODEL", "nova-3")

# The Veris actor speaks and listens at 24 kHz; the call is created to match
# on both legs, so neither direction is resampled.
ACTOR_RATE_HZ = 24000

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


class LiveCall:
    """One Vapi call as the webhook sees it: its session and its ordering lock."""

    def __init__(self) -> None:
        self.session = CallSession()
        self.lock = asyncio.Lock()


# Vapi call id -> the call the /tool webhook should resolve against. The
# bench runs one call per pod, but the envelope names its call, so this is
# keyed rather than global — two calls on one process cannot share a session.
LIVE_CALLS: Dict[str, LiveCall] = {}


def build_call_payload(tool_webhook_url: str) -> Dict[str, Any]:
    """The inline assistant, pinned so a run is reproducible.

    Per call rather than a stored assistant: the date context is resolved
    off the frozen clock the attempt restored with its snapshot, and the
    tool webhook URL is this pod's tunnel, so both are only known here.
    """
    return {
        "transport": {
            "provider": "vapi.websocket",
            "audioFormat": {"format": "pcm_s16le", "container": "raw", "sampleRate": ACTOR_RATE_HZ},
        },
        "assistant": {
            "firstMessage": GREETING,
            "firstMessageMode": "assistant-speaks-first",
            "model": {
                "provider": VAPI_MODEL_PROVIDER,
                "model": VAPI_MODEL,
                "temperature": 0.3,
                # Vapi's default of 250 truncates longer tool results and
                # model turns.
                "maxTokens": 500,
                "messages": [{"role": "system", "content": f"{AGENT_PROMPT}\n\n{today_context()}"}],
                "tools": build_tools(tool_webhook_url),
            },
            "voice": {"provider": VAPI_VOICE_PROVIDER, "voiceId": VAPI_VOICE_ID, "model": VAPI_VOICE_MODEL},
            "transcriber": {
                "provider": VAPI_TRANSCRIBER_PROVIDER,
                "model": VAPI_TRANSCRIBER_MODEL,
                "language": "en",
            },
            # Benchmark turn-taking standard: ~0.8 s end-of-turn silence.
            # transcriptionEndpointingPlan is what commits the turn, and its
            # countdown starts once the transcript is received (~250 ms after
            # speech stops), so 0.6 s on punctuated turns lands total
            # endpointing at or above ~0.8 s. waitSeconds is only the minimum
            # time before reply audio may flow, not the limiter. This only
            # holds for a transcriber without its own end-of-turn model; one
            # with it makes Vapi ignore transcriptionEndpointingPlan.
            "startSpeakingPlan": {
                "waitSeconds": 0.8,
                "transcriptionEndpointingPlan": {
                    "onPunctuationSeconds": 0.6,
                    "onNoPunctuationSeconds": 1.2,
                    "onNumberSeconds": 0.7,
                },
            },
            # Barge-in needs 2 confidently transcribed words, except the
            # listed phrases, which interrupt immediately.
            "stopSpeakingPlan": {
                "numWords": 2,
                "voiceSeconds": 0.2,
                "backoffSeconds": 1,
                "interruptionPhrases": ["stop", "hold on"],
            },
            # Vapi denoises by default; off for parity with the other
            # candidates — background noise is not part of the exam.
            "backgroundSpeechDenoisingPlan": {
                "smartDenoisingPlan": {"enabled": False},
                "fourierDenoisingPlan": {"enabled": False},
            },
            "silenceTimeoutSeconds": 60,
            "maxDurationSeconds": 1800,
            # conversation-update carries the history with each server tool's
            # result; a trace reader takes results from it, since
            # tool-calls-result was never seen on this transport.
            "clientMessages": [
                "transcript",
                "speech-update",
                "status-update",
                "tool-calls",
                "tool-calls-result",
                "conversation-update",
            ],
        },
    }


async def create_call(tool_webhook_url: str) -> Dict[str, Any]:
    """POST the inline assistant to Vapi; returns the call, including its transport URL."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            VAPI_API_URL,
            json=build_call_payload(tool_webhook_url),
            headers={"Authorization": f"Bearer {os.environ['VAPI_API_KEY']}"},
        )
    if response.status_code >= 400:
        logger.error(f"[call] Vapi /call failed status={response.status_code} body={response.text[:1000]}")
        response.raise_for_status()
    return response.json()


async def run_voice_ws_bot(actor_ws: WebSocket, tool_webhook_url: str) -> None:
    """One actor connection ↔ one Vapi WebSocket call with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other transports
    build one per call: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument. It is
    registered under the Vapi call id before any audio flows, because the
    first tool call can arrive as soon as the model has heard one turn.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    t_start = time.monotonic()

    call = await create_call(tool_webhook_url)
    call_id = call["id"]
    vapi_ws_url = call["transport"]["websocketCallUrl"]
    LIVE_CALLS[call_id] = LiveCall()
    logger.info(f"[voice] Vapi call {call_id} created url={vapi_ws_url}")

    try:
        async with websockets.connect(vapi_ws_url, max_size=None) as vapi_ws:
            logger.info(f"[voice] Vapi WS connected call={call_id}")
            up = asyncio.create_task(_pump_actor_to_vapi(actor_ws, vapi_ws), name="actor->vapi")
            # Completes when Vapi ends the call — a status-update or its
            # socket closing — so a call the platform hangs up on does not
            # sit waiting for actor frames that will never come. Leaving the
            # ``async with`` closes our side, which ends the call.
            down = asyncio.create_task(_pump_vapi_to_actor(vapi_ws, actor_ws), name="vapi->actor")
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
        LIVE_CALLS.pop(call_id, None)
        logger.info(f"[voice] handler exit call={call_id} duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_vapi(actor_ws: WebSocket, vapi_ws) -> None:
    """Binary PCM16 frames from the actor → the call's socket."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->v] first actor frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->v] forwarded {n_frames} frames ({n_bytes} bytes)")
            await vapi_ws.send(frame)
    except WebSocketDisconnect as exc:
        logger.info(f"[a->v] actor disconnected after {n_frames} frames ({n_bytes} bytes): code={exc.code}")
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(f"[a->v] non-binary frame after {n_frames} binary frames — actor protocol mismatch? ({exc})")
        raise


async def _pump_vapi_to_actor(vapi_ws, actor_ws: WebSocket) -> None:
    """The call's socket → the actor: audio forwarded, control messages logged.

    Vapi sends binary PCM16 at the configured rate and text JSON events.
    Tool calls are only *notified* here; they are executed through the HTTP
    webhook. A ``status-update`` of ``ended`` is the platform hanging up.
    """
    n_frames = 0
    n_bytes = 0
    try:
        async for raw in vapi_ws:
            if isinstance(raw, (bytes, bytearray)):
                n_frames += 1
                n_bytes += len(raw)
                if n_frames == 1:
                    logger.info(f"[v->a] first audio chunk bytes={len(raw)}")
                elif n_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(f"[v->a] streamed {n_frames} chunks ({n_bytes} bytes)")
                await actor_ws.send_bytes(bytes(raw))
                continue

            event = json.loads(raw)
            message = event.get("message", event)
            kind = message.get("type", "?")
            if kind == "transcript":
                if message.get("transcriptType") == "final":
                    who = "rory_said" if message.get("role") == "assistant" else "caller_said"
                    logger.info(f"[v->a] {who}: {message.get('transcript', '')}")
            elif kind == "status-update":
                status = message.get("status")
                logger.info(f"[v->a] status-update: {status}")
                if status == "ended":
                    return
            elif kind == "tool-calls":
                names = [call["function"]["name"] for call in message.get("toolCallList", [])]
                logger.info(f"[v->a] tool-calls notified: {names} (resolved via /tool webhook)")
            elif kind == "error":
                logger.error(f"[v->a] Vapi error: {json.dumps(message, default=str)}")
            else:
                logger.debug(f"[v->a] {kind}")
    except websockets.ConnectionClosed as exc:
        logger.info(f"[v->a] Vapi WS closed code={exc.code} reason={exc.reason!r} after {n_frames} chunks")
