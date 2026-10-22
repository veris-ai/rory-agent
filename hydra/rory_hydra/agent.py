"""Rory on Smallest AI's Hydra speech-to-speech model — Acme Energy's voice
support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Hydra ``/waves/v1/s2s`` websocket,
registers Rory's 16 tools in ``session.configure``, and bridges audio and
function calls in both directions. The agent speaks first.

Like Grok Voice and Gemini Live, Hydra *is* the whole voice loop — a single
speech-to-speech model, no STT/LLM/TTS pipeline to assemble, and turn-taking
and barge-in are owned server-side. What this module adds is the ``voice_ws``
bridging, the per-call ``CallSession``, and the hop onto a worker thread for
the blocking vendor calls.

Hydra's wire protocol reuses OpenAI-Realtime-shaped event names
(``response.function_call_arguments.done``, ``conversation.item.create``,
``response.create``, ``response.done``) for tool calls and turn generation,
but the session handshake is its own: the client must wait for
``session.created`` then send exactly one ``session.configure``, and must not
stream audio before ``session.configured`` answers it — frames sent earlier
are silently dropped.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. Hydra's
input is strictly PCM16 mono at 16 kHz regardless of model version, so the
inbound leg is always resampled. Hydra's output rate is fixed per model
version and announced back in ``session.configured``; at ``hydra-v1.1`` (the
default here) that is 24 kHz, matching the actor exactly, so the outbound leg
is resampled only if a different ``HYDRA_MODEL`` is selected.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import time
from urllib.parse import urlencode

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

HYDRA_MODEL = os.environ.get("HYDRA_MODEL", "hydra-v1.1")
HYDRA_VOICE = os.environ.get("HYDRA_VOICE", "wren")

# The actor's rate both ways. Hydra's input is pinned at 16 kHz no matter the
# model version — the strict leg, documented as PCM16/mono/16 kHz or
# `invalid_audio` errors.
ACTOR_RATE_HZ = 24000
HYDRA_INPUT_RATE_HZ = 16000

# Output rate is fixed per model version rather than negotiated, and announced
# in session.configured.session.output_audio_sample_rate. Pinned here so a
# mismatch between what the docs say and what the model actually answers with
# fails the boot loudly instead of shipping distorted audio.
HYDRA_OUTPUT_RATE_HZ = {
    "hydra-v1.1": 24000,
    "hydra-v1.0": 48000,
    "hydra": 48000,  # deprecated alias of hydra-v1.0
}

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _hydra_url() -> str:
    """Built per call so a missing credential fails at connect, not at import."""
    return "wss://api.smallest.ai/waves/v1/s2s?" + urlencode(
        {"model": HYDRA_MODEL, "api_key": os.environ["SMALLEST_API_KEY"]}
    )


def _session_configure() -> dict:
    """The one ``session.configure`` Hydra allows, sent right after ``session.created``.

    Persona, voice and audio format are frozen at handshake — only ``tools``
    can change later via ``session.update`` — so every knob that matters for a
    reproducible run is pinned here. ``generate_initial_response`` is left off
    (the documented default): the shared prompt tells the model it already
    greeted the caller, which would make the vendor's own speak-first flag
    paraphrase an opening line that contradicts its own system instructions.
    The greeting is instead forced open with a per-response override, the same
    way the Grok Voice transport does it — see ``_greeting_trigger``.

    The date context is appended rather than baked into ``agent_desc.txt``
    because it is resolved per call, off the frozen clock the attempt restored
    with its snapshot.
    """
    return {
        "type": "session.configure",
        "session": {
            "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
            "voice": HYDRA_VOICE,
            "tools": TOOLS,
            "generate_initial_response": False,
        },
    }


def _greeting_trigger() -> dict:
    """Forces the opening line despite the session instructions' own claim.

    Hydra's turn-taking is entirely server-driven with no documented way to
    seed an assistant turn or send a text conversation item (there is no
    transcript on the wire — audio is the payload), so there is no Pipecat- or
    Gemini-Live-style way to pre-render or textually request the greeting.
    ``response.create`` is the one documented lever: Smallest's tool-calling
    docs use it to make Hydra narrate a tool result on demand, so it is a real,
    client-triggerable event in this protocol. Whether its ``response.instructions``
    override is honoured the way Grok Voice's is has not been confirmed against a
    live session; if Hydra ignores it, the caller hears Hydra's own persona-driven
    opening rather than a broken call.
    """
    return {
        "type": "response.create",
        "response": {
            "instructions": (
                "Open the call by greeting the caller now. Say exactly: "
                f'"{GREETING}" Then stop and wait for the caller to respond.'
            ),
        },
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection <-> one Hydra session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other speech-to-speech
    candidates build one per call: it holds the account ``verify_caller``
    matched and dies with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()

    t_start = time.monotonic()
    try:
        async with websockets.connect(_hydra_url(), max_size=None) as hy_ws:
            created = json.loads(await hy_ws.recv())
            if created.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, got {created.get('type')!r}")
            logger.info(f"[voice] Hydra connected model={HYDRA_MODEL} voice={HYDRA_VOICE}")

            await hy_ws.send(json.dumps(_session_configure()))
            configured = json.loads(await hy_ws.recv())
            if configured.get("type") != "session.configured":
                raise RuntimeError(f"expected session.configured, got {configured.get('type')!r}")
            output_rate = configured["session"]["output_audio_sample_rate"]
            expected_rate = HYDRA_OUTPUT_RATE_HZ[HYDRA_MODEL]
            if output_rate != expected_rate:
                raise RuntimeError(
                    f"Hydra announced output_audio_sample_rate={output_rate} for "
                    f"model={HYDRA_MODEL}, expected {expected_rate}"
                )
            logger.info(
                f"[voice] session.configured: {len(TOOLS)} tools, "
                f"instructions={len(AGENT_PROMPT)} chars, output_rate={output_rate}Hz"
            )

            await hy_ws.send(json.dumps(_greeting_trigger()))
            logger.info("[voice] sent greeting trigger (agent greets first)")

            up = asyncio.create_task(_pump_actor_to_hydra(actor_ws, hy_ws), name="actor->hydra")
            down = asyncio.create_task(
                _pump_hydra_to_actor(hy_ws, actor_ws, session_state, output_rate),
                name="hydra->actor",
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


async def _pump_actor_to_hydra(actor_ws: WebSocket, hy_ws) -> None:
    """Binary PCM16 frames from the actor (24 kHz) -> Hydra's input buffer (16 kHz)."""
    n_frames = 0
    n_bytes = 0
    resample_state = None  # carried across ratecv calls so the resample is continuous
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            pcm16k, resample_state = audioop.ratecv(
                frame, 2, 1, ACTOR_RATE_HZ, HYDRA_INPUT_RATE_HZ, resample_state
            )
            if n_frames == 1:
                logger.info(f"[a->hy] first frame bytes={len(frame)} (->{len(pcm16k)} @16kHz)")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->hy] forwarded {n_frames} frames ({n_bytes} bytes @24kHz)")
            await hy_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16k).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->hy] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->hy] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_hydra_to_actor(
    hy_ws, actor_ws: WebSocket, session_state: CallSession, output_rate: int
) -> None:
    """Hydra events -> audio bytes back to the actor; dispatch tool calls.

    Audio deltas are forwarded as they stream, resampled to the actor's 24 kHz
    only if ``output_rate`` differs from it (a no-op at the default
    ``hydra-v1.1``, which already outputs 24 kHz).

    A function call's output goes back as a ``function_call_output`` item; the
    model does not resume on its own, so ``response.done`` after a completed
    tool round kicks the next ``response.create``. A response cancelled by a
    caller barge-in (``status: "cancelled"``) is not resumed — the caller is
    already mid-turn, and the next natural response covers it.
    """
    pending_tool_response = False
    n_audio_frames = 0
    n_audio_bytes = 0
    n_responses = 0
    resample_state = None  # carried across ratecv calls on the outbound leg, if used

    try:
        async for raw in hy_ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "response.output_audio.delta":
                audio = base64.b64decode(evt.get("delta", ""))
                if audio and output_rate != ACTOR_RATE_HZ:
                    audio, resample_state = audioop.ratecv(
                        audio, 2, 1, output_rate, ACTOR_RATE_HZ, resample_state
                    )
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[hy->a] first audio delta bytes={len(audio)}")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[hy->a] streamed {n_audio_frames} audio deltas "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if audio:
                    await actor_ws.send_bytes(audio)

            elif etype == "response.output_audio.done":
                continue  # already streamed delta-by-delta

            elif etype == "input_audio_buffer.speech_started":
                logger.info("[hy->a] vad: speech_started")

            elif etype == "input_audio_buffer.speech_stopped":
                logger.info("[hy->a] vad: speech_stopped")

            elif etype == "response.created":
                n_responses += 1
                logger.info(f"[hy->a] response.created #{n_responses}")

            elif etype == "response.function_call_arguments.done":
                name = evt["name"]
                call_id = evt["call_id"]
                raw_args = evt.get("arguments") or "{}"
                # A barge-in can cancel a response mid-generation, including
                # while a function call's arguments are still streaming; the
                # event still arrives with the arguments cut off wherever
                # generation stopped. That call was never completed by the
                # model, so it is not dispatched — the same failure the Grok
                # Voice and OpenAI Realtime transports handle.
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[hy->a] tool {name} ({call_id}) cancelled mid-generation, "
                        f"arguments truncated — not dispatched: {raw_args!r}"
                    )
                    continue
                pending_tool_response = True
                # Every vendor client is blocking, and this coroutine is
                # carrying the caller's audio. Called inline, a vendor timeout
                # is dead air on the mic leg — the same reason every other
                # speech-to-speech candidate hops threads. Awaiting here also
                # keeps tool calls sequential: they share account state.
                result = await asyncio.to_thread(dispatch, session_state, name, args)
                logger.info(
                    f"[hy->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}"
                )
                await hy_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, default=str),
                    },
                }))

            elif etype == "response.done":
                status = (evt.get("response") or {}).get("status")
                if status == "completed" and pending_tool_response:
                    pending_tool_response = False
                    logger.info("[hy->a] response.done after tool — kicking next response.create")
                    await hy_ws.send(json.dumps({"type": "response.create"}))
                elif status == "cancelled":
                    pending_tool_response = False
                    reason = ((evt.get("response") or {}).get("status_details") or {}).get("reason")
                    logger.info(f"[hy->a] response.done cancelled (reason={reason}) — barge-in")
                else:
                    pending_tool_response = False
                    logger.info(
                        f"[hy->a] response.done status={status} — session totals "
                        f"{n_audio_frames} frames / {n_audio_bytes} bytes"
                    )

            elif etype == "error":
                logger.error(f"[hy->a] Hydra error: {evt.get('error')}")
                raise RuntimeError(f"Hydra error: {evt.get('error')}")
    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[hy->a] Hydra WS closed code={exc.code} reason={exc.reason}; "
            f"{n_responses} responses, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        logger.info("[hy->a] actor WS closed; ending Hydra receive loop")
