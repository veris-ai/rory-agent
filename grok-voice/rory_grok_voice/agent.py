"""Rory on xAI's Grok speech-to-speech API — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Grok realtime websocket, registers Rory's
16 tools in ``session.update``, and bridges audio and function calls in both
directions. The agent speaks first.

Like OpenAI Realtime and Gemini Live, Grok *is* the whole voice loop — a single
speech-to-speech model, no STT/LLM/TTS pipeline to assemble. What this module
adds is the ``voice_ws`` bridging, the per-call ``CallSession``, and the hop
onto a worker thread for the blocking vendor calls.

The wire protocol is OpenAI-Realtime-compatible (``session.update``,
``input_audio_buffer.append``, ``response.output_audio.delta``,
``response.function_call_arguments.done``, ``response.done`` …), so this is the
Realtime transport with two deltas: ``voice``, ``instructions`` and
``turn_detection`` sit at the session's top level rather than under ``audio``,
and input transcription events are only emitted when
``audio.input.transcription.model`` names ``grok-transcribe``.

Everything below the transport is shared with the other candidates — same
greeting, same frozen-clock date context, same tools, same dispatcher, and the
same prompt apart from one sentence on how callers may say an account number
(``GROK_ACCOUNT_FORMAT``).

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and Grok
accepts and emits 24 kHz PCM16, so neither leg is resampled.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import os
import time

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

GROK_MODEL = os.environ.get("GROK_VOICE_MODEL", "grok-voice-3.0")
GROK_URL = f"wss://api.x.ai/v1/realtime?model={GROK_MODEL}"
GROK_VOICE = os.environ.get("GROK_VOICE", "eve")

# The Veris actor and Grok both run PCM16 at 24 kHz.
SAMPLE_RATE_HZ = 24000
PCM16_BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1000

# Agent audio goes to the actor in 20 ms frames at the rate it is spoken, kept
# this far ahead of real time so network jitter never starves playback.
PLAYOUT_CHUNK_BYTES = 20 * PCM16_BYTES_PER_MS
PLAYOUT_LEAD_S = 0.1

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

# After a tool round, how long server VAD gets to answer a caller who just
# finished speaking before the bridge asks for the response itself.
AUTO_RESPONSE_GRACE_S = 2.5
# The longest a finished tool round waits on a caller who is still talking.
TOOL_CONTINUE_TIMEOUT_S = 10.0

# Grok read the shared sentence as a rule for how callers must speak, and asked
# them to repeat correct numbers. The lookup still needs the dashed form.
SHARED_ACCOUNT_FORMAT = "an account number is 4 digits, a dash, then 5."
GROK_ACCOUNT_FORMAT = (
    "callers may read an account number with or without the dash; "
    "send it to verify_caller as 4 digits, a dash, then 5."
)


def _load_grok_prompt() -> str:
    prompt = load_agent_prompt()
    if SHARED_ACCOUNT_FORMAT not in prompt:
        raise RuntimeError("the shared prompt no longer has the account-format sentence")
    return prompt.replace(SHARED_ACCOUNT_FORMAT, GROK_ACCOUNT_FORMAT)


AGENT_PROMPT = _load_grok_prompt()


def _session_update() -> dict:
    """The session config, pinned so a run is reproducible.

    The date context is appended to the instructions rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot.
    """
    return {
        "type": "session.update",
        "session": {
            # Unlike OpenAI Realtime, Grok keeps voice, instructions and
            # turn_detection at the session's top level rather than under audio.
            "voice": GROK_VOICE,
            "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
            # xAI's own end-of-turn detection, with nothing pinned.
            "turn_detection": {"type": "server_vad"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    # Grok only emits input transcription events when this
                    # model is set; without it the caller's speech never
                    # reaches the container log or the graded trace.
                    "transcription": {"model": "grok-transcribe"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                },
            },
            "tools": TOOLS,
        },
    }


def _greeting() -> dict:
    """The opening line, spoken word for word without calling the model."""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "force_message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": GREETING}],
        },
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Grok realtime session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the Pipecat agent builds
    one per ``PipelineTask``: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    headers = {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"}

    t_start = time.monotonic()
    try:
        async with websockets.connect(GROK_URL, additional_headers=headers) as gk_ws:
            logger.info(
                f"[voice] Grok realtime connected model={GROK_MODEL} "
                f"voice={GROK_VOICE} rate={SAMPLE_RATE_HZ}Hz"
            )
            await gk_ws.send(json.dumps(_session_update()))
            logger.info(
                f"[voice] session.update sent: {len(TOOLS)} tools, "
                f"instructions={len(AGENT_PROMPT)} chars"
            )
            # Agent speaks first. force_message is its own turn, so no
            # response.create follows it.
            await gk_ws.send(json.dumps(_greeting()))
            logger.info("[voice] sent greeting as force_message")

            up = asyncio.create_task(_pump_actor_to_grok(actor_ws, gk_ws), name="actor->gk")
            down = asyncio.create_task(
                _pump_grok_to_actor(gk_ws, actor_ws, session_state), name="gk->actor"
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


async def _pump_actor_to_grok(actor_ws: WebSocket, gk_ws) -> None:
    """Binary PCM16 frames from the actor → Grok input buffer."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->gk] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->gk] forwarded {n_frames} frames ({n_bytes} bytes)")
            await gk_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->gk] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->gk] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


class _Playout:
    """Agent audio on its way to the actor, sent at the rate it is spoken.

    Grok generates audio faster than real time, and the voice_ws protocol cannot
    take audio back once it is sent. Holding it here is what lets a barge-in stop
    the words the caller has not heard yet.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self._actor_ws = actor_ws
        self._segments: collections.deque[tuple[str, bytearray]] = collections.deque()
        self._sent_bytes: collections.Counter[str] = collections.Counter()
        self._stopped_items: set[str] = set()
        self._has_audio = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()

    def write(self, item_id: str, audio: bytes) -> None:
        if not audio or item_id in self._stopped_items:
            return
        if self._segments and self._segments[-1][0] == item_id:
            self._segments[-1][1].extend(audio)
        else:
            self._segments.append((item_id, bytearray(audio)))
        self._drained.clear()
        self._has_audio.set()

    def interrupt(self) -> tuple[str, int] | None:
        """Drops unsent audio; returns the item cut off and the ms of it already sent."""
        if not self._segments:
            return None
        item_id = self._segments[0][0]
        self._stopped_items.update(segment[0] for segment in self._segments)
        self._segments.clear()
        self._drained.set()
        return item_id, self._sent_bytes[item_id] // PCM16_BYTES_PER_MS

    async def drained(self) -> None:
        await self._drained.wait()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        playhead = 0.0
        try:
            while True:
                if not self._segments:
                    self._drained.set()
                    self._has_audio.clear()
                    await self._has_audio.wait()
                    continue
                item_id, audio = self._segments[0]
                chunk = bytes(audio[:PLAYOUT_CHUNK_BYTES])
                del audio[:PLAYOUT_CHUNK_BYTES]
                if not audio:
                    self._segments.popleft()
                self._sent_bytes[item_id] += len(chunk)
                playhead = max(playhead, loop.time()) + len(chunk) / PCM16_BYTES_PER_MS / 1000
                await self._actor_ws.send_bytes(chunk)
                delay = playhead - PLAYOUT_LEAD_S - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
        except WebSocketDisconnect:
            logger.info("[gk->a] actor WS closed; ending playout")


class _TurnState:
    """What the Grok reader has seen, for the tool worker to act on."""

    def __init__(self) -> None:
        self.responses_started = 0
        self.response_active = False
        self.caller_speaking = False
        self.caller_stopped_at = 0.0
        self.changed = asyncio.Event()

    def touch(self) -> None:
        self.changed.set()


async def _run_tool(call: dict, session_state: CallSession) -> dict:
    """One tool call. Every call gets an output, or Grok waits on it forever."""
    try:
        args = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        logger.warning(
            f"[gk->a] tool {call['name']} ({call['call_id']}) has invalid JSON arguments: "
            f"{call['arguments']!r}"
        )
        return {"error": "the tool call's arguments were not valid JSON"}
    # Every vendor client is blocking, so it runs on a worker thread. Awaiting
    # it here keeps a round's calls sequential: they share account state.
    return await asyncio.to_thread(dispatch, session_state, call["name"], args)


async def _continue_after_tools(
    gk_ws, turn: _TurnState, rounds: asyncio.Queue, playout: _Playout
) -> None:
    """Ask for the next response once, unless another response will use the outputs.

    xAI's documented sequence: send the outputs, let the current turn's audio
    finish playing, then request the response. If the caller spoke during the tool round, server VAD starts its own response
    and that response reads the outputs. A second ``response.create`` would
    collide with it, so the bridge only asks when no such response arrives.
    """
    outputs_sent_at = turn.responses_started
    await playout.drained()
    deadline = time.monotonic() + TOOL_CONTINUE_TIMEOUT_S
    while True:
        if turn.responses_started > outputs_sent_at or not rounds.empty():
            return
        now = time.monotonic()
        timeout = None
        if not turn.response_active:
            if now >= deadline:
                break
            wake_at = deadline
            if not turn.caller_speaking:
                grace_ends = turn.caller_stopped_at + AUTO_RESPONSE_GRACE_S
                if now >= grace_ends:
                    break
                wake_at = min(deadline, grace_ends)
            timeout = wake_at - now
        turn.changed.clear()
        try:
            await asyncio.wait_for(turn.changed.wait(), timeout)
        except TimeoutError:
            pass
    logger.info("[gk->a] tool outputs sent — response.create")
    await gk_ws.send(json.dumps({"type": "response.create"}))


async def _run_tool_rounds(
    rounds: asyncio.Queue, gk_ws, session_state: CallSession, turn: _TurnState, playout: _Playout
) -> None:
    """Runs tool rounds off the reader, so a slow tool never stalls Grok's events."""
    try:
        while True:
            calls = await rounds.get()
            for call in calls:
                result = await _run_tool(call, session_state)
                logger.info(
                    f"[gk->a] tool {call['name']} ({call['call_id']}) -> "
                    f"{json.dumps(result, default=str)}"
                )
                await gk_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result, default=str),
                    },
                }))
            await _continue_after_tools(gk_ws, turn, rounds, playout)
    except websockets.ConnectionClosed as exc:
        logger.info(f"[gk->a] Grok WS closed during a tool round: code={exc.code}")


async def _pump_grok_to_actor(gk_ws, actor_ws: WebSocket, session_state: CallSession) -> None:
    """Reads Grok's events and runs its tool calls until either side ends."""
    turn = _TurnState()
    rounds: asyncio.Queue = asyncio.Queue()
    playout = _Playout(actor_ws)
    tasks = {
        asyncio.create_task(_read_grok(gk_ws, playout, turn, rounds), name="gk-reader"),
        asyncio.create_task(
            _run_tool_rounds(rounds, gk_ws, session_state, turn, playout), name="gk-tools"
        ),
        asyncio.create_task(playout.run(), name="gk-playout"),
    }
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _read_grok(gk_ws, playout: _Playout, turn: _TurnState, rounds: asyncio.Queue) -> None:
    """Grok events → audio to the playout; tool calls → the worker.

    The first output item of each response goes to the playout delta-by-delta as
    the model generates it, so the actor hears the reply onset immediately and
    response latency reflects the model rather than this bridge.

    Any *additional* items in the same response are buffered and deduped by
    transcript before forwarding — the guard the Realtime transport grew after
    seeing one item emitted twice within a response. Grok has not been seen
    doing it; keeping the guard costs nothing on the one-item-per-response path,
    which streams with zero buffering.

    A function call is queued only once its response completes. A response the
    caller cancelled never runs its tools.
    """
    calls_by_response: dict[str | None, list[dict]] = {}
    n_audio_frames = 0
    n_audio_bytes = 0
    n_responses = 0

    # Per-response state — reset on response.created.
    live_item_id: str | None = None  # first audio item this response; streamed live
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
                f"[gk->a] dropped duplicate item {item_id} "
                f"(same transcript as earlier item this response): {text}"
            )
            return
        item["decision"] = "forward"
        forwarded_transcripts.add(text)
        forwarded_count += 1
        audio_bytes = bytes(item["audio"])
        playout.write(item_id, audio_bytes)
        logger.info(f"[gk->a] rory_said (item {item_id}, {len(audio_bytes)} bytes): {text}")

    try:
        async for raw in gk_ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "response.output_audio.delta":
                item_id = evt.get("item_id", "default")
                audio = base64.b64decode(evt.get("delta", ""))
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[gk->a] first audio delta bytes={len(audio)} (item {item_id})")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[gk->a] streamed {n_audio_frames} audio deltas "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if live_item_id is None:
                    live_item_id = item_id
                    forwarded_count += 1
                    logger.info(f"[gk->a] streaming live item {item_id}")
                if item_id == live_item_id:
                    playout.write(item_id, audio)
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
                        logger.info(f"[gk->a] rory_said (live item {item_id}): {text}")
                    continue
                item = items.setdefault(item_id, _new_item())
                item["transcript"] = text
                item["transcript_done"] = True
                await _maybe_forward(item_id)

            elif etype == "conversation.item.input_audio_transcription.completed":
                # Cumulative transcript of the caller's audio; Grok may emit it
                # more than once per utterance as the transcript refines. If it
                # never fires after the actor speaks, server VAD did not commit
                # and the actor's silence trailer is missing.
                text = (evt.get("transcript") or "").strip()
                if text:
                    logger.info(f"[gk->a] caller_said: {text}")

            elif etype == "input_audio_buffer.speech_started":
                logger.info("[gk->a] vad: speech_started")
                turn.caller_speaking = True
                turn.touch()
                interrupted = playout.interrupt()
                if interrupted:
                    interrupted_item, audio_end_ms = interrupted
                    logger.info(f"[gk->a] barge-in: truncate {interrupted_item} at {audio_end_ms} ms")
                    await gk_ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": interrupted_item,
                        "content_index": 0,
                        "audio_end_ms": audio_end_ms,
                    }))

            elif etype == "input_audio_buffer.speech_stopped":
                logger.info("[gk->a] vad: speech_stopped")
                turn.caller_speaking = False
                turn.caller_stopped_at = time.monotonic()
                turn.touch()

            elif etype == "input_audio_buffer.committed":
                logger.info("[gk->a] vad: buffer committed -> response generation")

            elif etype == "response.created":
                n_responses += 1
                turn.responses_started += 1
                turn.response_active = True
                turn.touch()
                live_item_id = None
                items.clear()
                forwarded_transcripts.clear()
                forwarded_count = 0
                logger.info(
                    f"[gk->a] response.created #{n_responses} id={(evt.get('response') or {}).get('id')}"
                )

            elif etype == "response.function_call_arguments.done":
                logger.info(
                    f"[gk->a] tool args done {evt['name']} ({evt['call_id']}) "
                    f"response={evt.get('response_id')}"
                )
                calls_by_response.setdefault(evt.get("response_id"), []).append({
                    "name": evt["name"],
                    "call_id": evt["call_id"],
                    "arguments": evt.get("arguments"),
                })

            elif etype == "conversation.item.truncated":
                logger.info(
                    f"[gk->a] item {evt.get('item_id')} truncated at "
                    f"audio_end_ms={evt.get('audio_end_ms')}"
                )

            elif etype == "response.done":
                response = evt.get("response") or {}
                status = response.get("status")
                output_types = [item.get("type") for item in response.get("output") or []]
                logger.info(
                    f"[gk->a] response.done id={response.get('id')} status={status} "
                    f"output={output_types} — {forwarded_count} item(s) forwarded; "
                    f"session totals {n_audio_frames} frames / {n_audio_bytes} bytes"
                )
                calls = calls_by_response.pop(response.get("id"), [])
                if status == "completed":
                    if len(calls) != output_types.count("function_call"):
                        logger.warning(
                            f"[gk->a] response {response.get('id')} listed "
                            f"{output_types.count('function_call')} function calls, "
                            f"received arguments for {len(calls)}"
                        )
                    if calls:
                        rounds.put_nowait(calls)
                elif calls:
                    logger.info(f"[gk->a] {len(calls)} tool call(s) not run: response {status}")
                turn.response_active = False
                turn.touch()

            elif etype == "error":
                # xAI documents most errors as recoverable, with the session
                # left open. A session that is really over closes its websocket,
                # which ends this loop.
                logger.error(f"[gk->a] Grok realtime error: {evt.get('error')}")
    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[gk->a] Grok WS closed code={exc.code} reason={exc.reason}; "
            f"{n_responses} responses, {n_audio_bytes} audio bytes"
        )
