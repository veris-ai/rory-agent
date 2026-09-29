"""Rory on Cartesia Managed Agents — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
At startup the pod creates Rory's 16 tools as Cartesia webhook tools, once. For
each connection it creates a managed agent on them (Ink speech recognition, the
LLM, Sonic speech), opens the agent's WebSocket and relays audio in both
directions. The agent speaks first. The agent is deleted when the call ends,
the tools when the pod shuts down.

Cartesia runs the model. It executes a tool by POSTing the arguments to the
tool's URL — ``/tool/<tool name>`` on this pod, reached through the attempt's
public endpoint (``PUBLIC_BASE_URL``) — with a bearer secret, and hands the
response to the model. URL and secret are fixed when a tool is created, so both
are per pod: ``rory_cartesia.web`` checks the bearer and resolves the webhook
against the pod's one live call.

Why once per pod: an account may store at most 100 tools. Creating the 16 per
call would fail every call past the sixth concurrent one with
``tool_limit_reached``. Sixteen per pod caps an account at six pods.

Why one call at a time: the tool URL is the only thing that could name a call,
and it is fixed at creation. Each concurrent call needs its own pod and public
endpoint; a second concurrent connection is refused rather than answered
against another caller's session.

Why Managed Agents rather than the Line SDK: Cartesia stops hosting Line agents
on December 1, 2026, and ``cartesia-line`` pins ``websockets<14`` and
``starlette<1``, which the shared lockfile cannot hold beside the transports that
run websockets 15.

Everything below the transport is shared with the other candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. Vendor limits that shape the comparison: the LLM catalog has no
``gpt-4.1-mini``, so the model is ``gpt-4.1``; webhook responses are truncated
after 4 KiB; and the body schema has no ``minimum`` (see ``tools.py``).

Sample-rate note: the session runs ``pcm_24000`` in both directions — the
actor's format — so nothing is resampled.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import httpx
import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import build_tools

CARTESIA_API = "https://api.cartesia.ai"
CARTESIA_WS = "wss://api.cartesia.ai"
CARTESIA_VERSION = "2026-08-14"
CARTESIA_MODEL = os.environ.get("CARTESIA_MODEL", "gpt-4.1")
# "Skylar – Friendly Guide", an approachable American voice.
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")

# The Veris actor speaks and listens at 24 kHz PCM16 mono.
AUDIO_FORMAT = "pcm_24000"

# Roughly one heartbeat per second at 20–40 ms chunks.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


@dataclass
class LiveCall:
    """What the /tool webhook needs to resolve the pod's call.

    The lock keeps one call's tools from running at once: they share account
    state, and a webhook can land while the previous one is still running.
    """

    session: CallSession = field(default_factory=CallSession)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# The call this pod is on, which its webhooks resolve against; None between calls.
LIVE_CALL: LiveCall | None = None


def cartesia_api() -> httpx.AsyncClient:
    """The REST client the pod holds for its lifetime: the tools at startup, an agent per call."""
    return httpx.AsyncClient(
        base_url=CARTESIA_API,
        headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": CARTESIA_VERSION},
        timeout=30.0,
    )


async def run_tool_call(live: LiveCall, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run one webhook's tool through the shared gate, off the event loop.

    Every vendor client is blocking, and this loop is also carrying the
    caller's audio.
    """
    async with live.lock:
        return await asyncio.to_thread(dispatch, live.session, name, args)


def agent_config(tool_ids: List[str]) -> Dict[str, Any]:
    """The managed agent's configuration, pinned so a run is reproducible.

    The date context is resolved per call, off the frozen clock the attempt
    restored with its snapshot.
    """
    return {
        "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
        "initial_message": GREETING,
        "model": {"id": CARTESIA_MODEL},
        "language": {"primary": "en"},
        "audio": {
            # Background noise is not part of the benchmark; the other hosted
            # transports turn their denoising off too.
            "input": {"noise_suppression": "off"},
            "output": {"voice_id": CARTESIA_VOICE_ID},
        },
        "tools": [{"id": tool_id} for tool_id in tool_ids],
    }


async def run_voice_ws_bot(actor_ws: WebSocket, api: httpx.AsyncClient, tool_ids: List[str]) -> None:
    """One actor connection ↔ one call on a freshly created managed agent over the pod's tools."""
    global LIVE_CALL
    if LIVE_CALL is not None:
        raise RuntimeError("this pod is already on a call; its webhook URL names one call at a time")
    LIVE_CALL = LiveCall()
    t_start = time.monotonic()
    try:
        response = await api.post("/v1/agents", json={"name": "Rory", "config": agent_config(tool_ids)})
        response.raise_for_status()
        agent_id = response.json()["id"]
        try:
            await _relay(actor_ws, agent_id, os.environ["CARTESIA_API_KEY"])
        finally:
            (await api.delete(f"/v1/agents/{agent_id}")).raise_for_status()
    finally:
        LIVE_CALL = None
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def create_tools(api: httpx.AsyncClient, base_url: str, secret: str) -> List[str]:
    """Create the 16 webhook tools concurrently; on any failure delete the rest and raise."""
    definitions = build_tools(base_url, secret)
    responses = await asyncio.gather(
        *(api.post("/v1/agents/tools", json=definition) for definition in definitions), return_exceptions=True
    )
    created = [r.json()["id"] for r in responses if isinstance(r, httpx.Response) and r.status_code == 201]
    if len(created) != len(definitions):
        await delete_tools(api, created)
        failure = next(r for r in responses if not (isinstance(r, httpx.Response) and r.status_code == 201))
        detail = failure if isinstance(failure, Exception) else f"HTTP {failure.status_code} {failure.text}"
        raise RuntimeError(f"Cartesia tool creation failed: {detail}")
    logger.info(f"[voice] created {len(created)} webhook tools model={CARTESIA_MODEL} voice={CARTESIA_VOICE_ID}")
    return created


async def delete_tools(api: httpx.AsyncClient, tool_ids: List[str]) -> None:
    """Delete the pod's tools. A tool cannot be deleted while an agent references it."""
    responses = await asyncio.gather(*(api.delete(f"/v1/agents/tools/{tool_id}") for tool_id in tool_ids))
    for response in responses:
        response.raise_for_status()


async def _relay(actor_ws: WebSocket, agent_id: str, api_key: str) -> None:
    """Open the agent's WebSocket at the actor's format and pump until either side ends."""
    url = f"{CARTESIA_WS}/v1/agents/websocket/{agent_id}?cartesia_version={CARTESIA_VERSION}"
    async with websockets.connect(url, additional_headers={"X-API-Key": api_key}, max_size=None) as ct_ws:
        await ct_ws.send(
            json.dumps({"type": "session_create", "audio": {"input_format": AUDIO_FORMAT, "output_delivery": "speaking_pace"}})
        )
        ready = json.loads(await ct_ws.recv())
        if ready["type"] != "session_ready":
            raise RuntimeError(f"Cartesia refused the session: {ready}")
        logger.info(f"[voice] Cartesia session ready call={ready['call_id']} agent={agent_id}")

        up = asyncio.create_task(_pump_actor_to_cartesia(actor_ws, ct_ws), name="actor->cartesia")
        down = asyncio.create_task(_pump_cartesia_to_actor(ct_ws, actor_ws), name="cartesia->actor")
        try:
            done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (up, down):
                if not task.done():
                    task.cancel()
            await asyncio.gather(up, down, return_exceptions=True)


async def _pump_actor_to_cartesia(actor_ws: WebSocket, ct_ws) -> None:
    """Binary PCM16 frames from the actor → base64 ``audio_input`` events.

    The actor streams continuously, silence included, which is also what keeps
    Cartesia's 120-second inactivity timer from closing the session.
    """
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->c] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->c] forwarded {n_frames} frames ({n_bytes} bytes)")
            await ct_ws.send(json.dumps({"type": "audio_input", "audio": base64.b64encode(frame).decode()}))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->c] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except websockets.ConnectionClosed as exc:
        logger.info(f"[a->c] Cartesia WS closed after {n_frames} frames: code={exc.code} reason={exc.reason}")
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->c] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_cartesia_to_actor(ct_ws, actor_ws: WebSocket) -> None:
    """Cartesia events → audio bytes back to the actor; transcript and errors → the log.

    ``speaking_pace`` delivery means Cartesia paces the audio itself, so each
    chunk is forwarded as it arrives.
    """
    n_chunks = 0
    n_bytes = 0
    try:
        async for raw in ct_ws:
            evt = json.loads(raw)
            etype = evt["type"]
            if etype == "audio_output":
                audio = base64.b64decode(evt["audio"])
                n_chunks += 1
                n_bytes += len(audio)
                if n_chunks == 1:
                    logger.info(f"[c->a] first audio chunk bytes={len(audio)}")
                elif n_chunks % LOG_EVERY_N_FRAMES == 0:
                    logger.info(f"[c->a] forwarded {n_chunks} audio chunks ({n_bytes} bytes)")
                await actor_ws.send_bytes(audio)
            elif etype == "turn_ended":
                who = "rory_said" if evt["role"] == "assistant" else "caller_said"
                logger.info(f"[c->a] {who}: {evt['text']}")
                for call in evt.get("tool_calls") or []:
                    logger.info(f"[c->a] tool call in turn {evt['turn']}: {json.dumps(call, default=str)}")
            elif etype == "audio_output_clear":
                # Barge-in. Audio already sent to the actor cannot be recalled.
                logger.info("[c->a] audio_output_clear (caller barged in)")
            elif etype == "error":
                if evt.get("fatal"):
                    raise RuntimeError(f"Cartesia session error: {evt}")
                logger.warning(f"[c->a] Cartesia recoverable error: {evt}")
            elif etype not in ("turn_started", "turn_output_text_delta"):
                logger.info(f"[c->a] event {etype}: {json.dumps(evt)[:300]}")
    except websockets.ConnectionClosed as exc:
        logger.info(f"[c->a] Cartesia WS closed code={exc.code} reason={exc.reason}; {n_chunks} chunks / {n_bytes} bytes")
    except WebSocketDisconnect:
        logger.info("[c->a] actor WS closed; ending Cartesia receive loop")
