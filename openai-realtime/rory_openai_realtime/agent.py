"""Rory on OpenAI Realtime — Acme Energy's voice support agent, speech-to-speech.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own OpenAI Realtime websocket, registers
Rory's 16 tools in ``session.update``, and bridges audio and function calls in
both directions. The agent speaks first.

Like Gemini Live, Realtime *is* the whole voice loop — there is no STT/LLM/TTS
pipeline to assemble. What this module adds is the ``voice_ws`` bridging, the
per-call ``CallSession``, and the hop onto a worker thread for the blocking
vendor calls.

Everything below the transport is shared with the other candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. They exist to be compared, so the transport is the only thing
allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and
Realtime accepts and emits 24 kHz PCM16, so neither leg is resampled.
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
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

# The full model by default; the mini is a per-env override.
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime-2")
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
REALTIME_VOICE = os.environ.get("REALTIME_VOICE", "alloy")

# The Veris actor and Realtime both run PCM16 at 24 kHz.
SAMPLE_RATE_HZ = 24000
PCM16_BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1000

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _session_update() -> dict:
    """The session config, pinned so a run is reproducible.

    The date context is appended to the instructions rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot.
    """
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    # Turn-taking parity with the other candidates: ~0.8 s of
                    # end-of-turn silence. server_vad is pure silence duration
                    # (not semantic); threshold and prefix_padding_ms are
                    # pinned so they are not silently defaulted.
                    "turn_detection": {
                        "type": "server_vad",
                        "silence_duration_ms": 800,
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    "voice": REALTIME_VOICE,
                },
            },
            "tools": TOOLS,
        },
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one OpenAI Realtime session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the Pipecat agent builds
    one per ``PipelineTask``: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    t_start = time.monotonic()
    try:
        async with websockets.connect(REALTIME_URL, additional_headers=headers) as oa_ws:
            logger.info(
                f"[voice] OA Realtime connected model={REALTIME_MODEL} "
                f"voice={REALTIME_VOICE} rate={SAMPLE_RATE_HZ}Hz"
            )
            await oa_ws.send(json.dumps(_session_update()))
            logger.info(
                f"[voice] session.update sent: {len(TOOLS)} tools, "
                f"instructions={len(AGENT_PROMPT)} chars"
            )
            # Agent speaks first. The system prompt tells the model the greeting
            # already happened, so force it on this opening turn with
            # per-response instructions, which override the session
            # instructions for that one response.
            await oa_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "instructions": (
                        "Open the call by greeting the caller now. Say exactly: "
                        f'"{GREETING}" Then stop and wait for the caller to respond.'
                    ),
                },
            }))
            logger.info("[voice] sent greeting trigger (agent greets first)")

            up = asyncio.create_task(_pump_actor_to_oa(actor_ws, oa_ws), name="actor->oa")
            down = asyncio.create_task(
                _pump_oa_to_actor(oa_ws, actor_ws, session_state), name="oa->actor"
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


async def _pump_actor_to_oa(actor_ws: WebSocket, oa_ws) -> None:
    """Binary PCM16 frames from the actor → Realtime input buffer."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->oa] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->oa] forwarded {n_frames} frames ({n_bytes} bytes)")
            await oa_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->oa] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->oa] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_oa_to_actor(oa_ws, actor_ws: WebSocket, session_state: CallSession) -> None:
    """Realtime events → audio bytes back to the actor; dispatch tool calls.

    The first output item of each response is forwarded delta-by-delta as the
    model generates it, so the actor hears the reply onset immediately and
    response latency reflects the model rather than this bridge (whole-turn
    buffering was measured at ~3 s extra on Riley).

    Any *additional* items in the same response are buffered and deduped by
    transcript before forwarding: Realtime has been observed emitting the same
    item twice within one response (greeting audio played twice). The common
    case is one item per response, which streams with zero buffering.

    A function call's output goes back as a ``function_call_output`` item; the
    model does not resume on its own, so ``response.done`` after a tool turn
    kicks the next ``response.create``.

    When the caller speaks over the live item, server VAD cancels the response
    but keeps the item's full generated transcript in the model's context. The
    bridge sends ``conversation.item.truncate`` at the audio it had forwarded
    by then, so the server drops the audio and transcript past that point and
    the model does not later refer to words the caller never heard.
    """
    pending_tool_response = False
    n_audio_frames = 0
    n_audio_bytes = 0
    n_responses = 0

    # Per-response state — reset on response.created.
    live_item_id: str | None = None  # first audio item this response; streamed live
    live_item_bytes = 0  # PCM16 bytes of the live item forwarded to the actor
    items: dict[str, dict] = {}  # buffered ADDITIONAL items (dedup path)
    forwarded_transcripts: set[str] = set()
    forwarded_count = 0

    def _new_item() -> dict:
        return {
            "audio": bytearray(),
            "transcript": "",
            "audio_done": False,
            "transcript_done": False,
            "decision": None,
        }

    async def _maybe_forward(item_id: str) -> None:
        nonlocal forwarded_count
        item = items[item_id]
        if not (item["audio_done"] and item["transcript_done"]):
            return
        if item["decision"] is not None:
            return
        text = item["transcript"]
        if text and text in forwarded_transcripts:
            item["decision"] = "drop"
            logger.warning(
                f"[oa->a] dropped duplicate item {item_id} "
                f"(same transcript as earlier item this response): {text}"
            )
            return
        item["decision"] = "forward"
        forwarded_transcripts.add(text)
        forwarded_count += 1
        audio_bytes = bytes(item["audio"])
        if audio_bytes:
            await actor_ws.send_bytes(audio_bytes)
        logger.info(f"[oa->a] rory_said (item {item_id}, {len(audio_bytes)} bytes): {text}")

    try:
        async for raw in oa_ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "response.output_audio.delta":
                item_id = evt.get("item_id", "default")
                audio = base64.b64decode(evt.get("delta", ""))
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[oa->a] first audio delta bytes={len(audio)} (item {item_id})")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[oa->a] streamed {n_audio_frames} audio deltas "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if live_item_id is None:
                    live_item_id = item_id
                    forwarded_count += 1
                    logger.info(f"[oa->a] streaming live item {item_id}")
                if item_id == live_item_id:
                    if audio:
                        await actor_ws.send_bytes(audio)
                        live_item_bytes += len(audio)
                else:
                    items.setdefault(item_id, _new_item())["audio"].extend(audio)

            elif etype == "response.output_audio.done":
                item_id = evt.get("item_id", "default")
                if item_id == live_item_id:
                    continue  # already streamed delta-by-delta
                item = items.setdefault(item_id, _new_item())
                item["audio_done"] = True
                await _maybe_forward(item_id)

            elif etype == "response.output_audio_transcript.done":
                item_id = evt.get("item_id", "default")
                text = (evt.get("transcript") or "").strip()
                if item_id == live_item_id:
                    # Record the streamed transcript so a duplicate later item
                    # in this response is dropped by _maybe_forward.
                    if text:
                        forwarded_transcripts.add(text)
                        logger.info(f"[oa->a] rory_said (live item {item_id}): {text}")
                    continue
                item = items.setdefault(item_id, _new_item())
                item["transcript"] = text
                item["transcript_done"] = True
                await _maybe_forward(item_id)

            elif etype == "conversation.item.input_audio_transcription.completed":
                # If this never fires after the actor speaks, server VAD did
                # not commit and the actor's silence trailer is missing.
                text = (evt.get("transcript") or "").strip()
                if text:
                    logger.info(f"[oa->a] caller_said: {text}")

            elif etype == "input_audio_buffer.speech_started":
                logger.info("[oa->a] vad: speech_started")
                if live_item_id is not None:
                    # The live item is forwarded as fast as Realtime emits it
                    # and the actor plays out everything it received, so the
                    # forwarded bytes are what the caller hears of this item.
                    audio_end_ms = live_item_bytes // PCM16_BYTES_PER_MS
                    await oa_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": live_item_id,
                        "content_index": 0,
                        "audio_end_ms": audio_end_ms,
                    }))
                    logger.info(f"[a->oa] truncate item={live_item_id} audio_end_ms={audio_end_ms}")

            elif etype == "conversation.item.truncated":
                logger.info(
                    f"[oa->a] item {evt['item_id']} truncated at audio_end_ms={evt['audio_end_ms']}"
                )

            elif etype == "input_audio_buffer.speech_stopped":
                logger.info("[oa->a] vad: speech_stopped")

            elif etype == "input_audio_buffer.committed":
                logger.info("[oa->a] vad: buffer committed -> response generation")

            elif etype == "response.created":
                n_responses += 1
                live_item_id = None
                live_item_bytes = 0
                items.clear()
                forwarded_transcripts.clear()
                forwarded_count = 0
                logger.info(f"[oa->a] response.created #{n_responses}")

            elif etype == "response.function_call_arguments.done":
                name = evt["name"]
                call_id = evt["call_id"]
                raw_args = evt.get("arguments") or "{}"
                # Realtime sends arguments as a JSON string; a no-argument tool
                # can arrive with it empty. When server VAD cancels the response
                # mid-generation (caller or background noise spoke over it), the
                # event still arrives, carrying the arguments cut off wherever
                # generation stopped. That call was never completed by the model,
                # so it is not dispatched; the cancelled response's own
                # response.done follows and the model retries on the next turn.
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[oa->a] tool {name} ({call_id}) cancelled mid-generation, "
                        f"arguments truncated — not dispatched: {raw_args!r}"
                    )
                    continue
                pending_tool_response = True
                # Every vendor client is blocking, and this coroutine is
                # carrying the caller's audio. Called inline, a ten-second
                # vendor timeout is ten seconds of dead air — the same reason
                # the Pipecat handler hops threads. Awaiting here also keeps
                # tool calls sequential: they share account state.
                result = await asyncio.to_thread(dispatch, session_state, name, args)
                logger.info(
                    f"[oa->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}"
                )
                await oa_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, default=str),
                    },
                }))

            elif etype == "response.done":
                # A completed item was forwarded whole; a later barge-in has
                # nothing of it to truncate.
                live_item_id = None
                if pending_tool_response:
                    pending_tool_response = False
                    logger.info("[oa->a] response.done after tool — kicking next response.create")
                    await oa_ws.send(json.dumps({"type": "response.create"}))
                else:
                    logger.info(
                        f"[oa->a] response.done — {forwarded_count} item(s) forwarded; "
                        f"session totals {n_audio_frames} frames / {n_audio_bytes} bytes"
                    )

            elif etype == "error":
                error = evt.get("error") or {}
                if error.get("code") == "conversation_already_has_active_response":
                    # Server VAD started a response while the caller spoke during a
                    # tool round, and our post-tool response.create collided with it.
                    # The function_call_output items are already in the conversation,
                    # so the response in flight consumes them. Seen on live calls:
                    # the call was healthy, only the extra kick was rejected.
                    logger.warning("[oa->a] response.create collided with an active response — continuing")
                    continue
                logger.error(f"[oa->a] OpenAI Realtime error: {error}")
                raise RuntimeError(f"OpenAI Realtime error: {error}")
    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[oa->a] OA WS closed code={exc.code} reason={exc.reason}; "
            f"{n_responses} responses, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        logger.info("[oa->a] actor WS closed; ending Realtime receive loop")
