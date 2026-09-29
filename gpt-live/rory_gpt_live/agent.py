"""Rory on OpenAI GPT-Live — Acme Energy's voice support agent, full duplex.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own GPT-Live websocket
(``wss://api.openai.com/v1/live/sessions``), starts a session with Rory's 16
tools registered on the Responses backend the live model delegates to, and
bridges audio and function calls in both directions. The agent speaks first.

GPT-Live is not the Realtime API. The live model listens and speaks at the same
time and runs no tools of its own: anything that needs a tool or deeper
reasoning is *delegated* to a backend Responses model configured in
``session.start``. That backend's function calls come back to this pod inside
``response.event`` envelopes, are run through the shared dispatcher, and are
answered with ``response.item.create`` + ``response.create``. The live model
keeps talking while the backend works, so tool calls are answered off the
receive loop rather than inline, one at a time because Rory's tools share
account state.

Everything below the transport is shared with the other candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. They exist to be compared, so the transport is the only thing
allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and the
GPT-Live websocket accepts and emits 24 kHz PCM16, so neither leg is resampled.
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

LIVE_MODEL = os.environ.get("LIVE_MODEL", "gpt-live-1")
LIVE_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_VOICE = os.environ.get("LIVE_VOICE", "marin")
# The Responses model GPT-Live delegates tool use and reasoning to; the
# benchmark run used gpt-5.6-terra.
LIVE_BACKEND_MODEL = os.environ.get("LIVE_BACKEND_MODEL", "gpt-5.6-terra")

# The Veris actor and GPT-Live both run PCM16 at 24 kHz.
SAMPLE_RATE_HZ = 24000

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

# How long to wait for session.closed (and its final usage) after the actor hangs up.
CLOSE_GRACE_S = 5.0

# A speaker's fragments are one line until the other speaker's next fragment
# starts this long after them on the session timeline. Shorter gaps are
# overlap — the two talking at once — not a turn.
TURN_GAP_MS = 1000

AGENT_PROMPT = load_agent_prompt()

# The backend never hears the call: it reads the transcript GPT-Live hands it.
# This preamble on the backend prompt says so.
BACKEND_PREAMBLE = (
    "## Voice conversation context\n"
    "You are the reasoning and tool-use backend of a live voice agent. "
    "Transcripts can contain mistakes, unfinished phrases, and later "
    "corrections. Use the latest context and the tool results you receive. "
    "Report an action as complete only after its tool confirms it.\n\n"
    "## Task instructions\n"
)


def _session_start() -> dict:
    """The session config, pinned so a run is reproducible.

    The date context is appended to the instructions rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot. The live model and the backend read the
    same prompt: the backend is what applies Rory's policy through the tools,
    and the live model is what says it out loud.
    """
    prompt = f"{AGENT_PROMPT}\n\n{today_context()}"
    return {
        "type": "session.start",
        "session": {
            "model": LIVE_MODEL,
            "instructions": prompt,
            "audio": {"output": {"voice": LIVE_VOICE}},
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": LIVE_BACKEND_MODEL,
                    "instructions": f"{BACKEND_PREAMBLE}{prompt}",
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    # Rory's tools share account state and run one at a time.
                    "parallel_tool_calls": False,
                },
            },
        },
    }


def _greeting_instruction() -> dict:
    """Agent speaks first. GPT-Live has no response.create; a trusted
    instruction append is how an application makes it say something now."""
    return {
        "type": "session.instructions.append",
        "delegation_id": None,
        "content": (
            "Open the call by greeting the caller now. Say exactly: "
            f'"{GREETING}" Then stop and wait for the caller to respond.'
        ),
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one GPT-Live session with Rory's tools.

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
        async with websockets.connect(LIVE_URL, additional_headers=headers) as live_ws:
            logger.info(
                f"[voice] GPT-Live connected model={LIVE_MODEL} backend={LIVE_BACKEND_MODEL} "
                f"voice={LIVE_VOICE} rate={SAMPLE_RATE_HZ}Hz"
            )
            await live_ws.send(json.dumps(_session_start()))
            logger.info(
                f"[voice] session.start sent: {len(TOOLS)} tools, "
                f"instructions={len(AGENT_PROMPT)} chars"
            )

            up = asyncio.create_task(_pump_actor_to_live(actor_ws, live_ws), name="actor->live")
            down = asyncio.create_task(
                _pump_live_to_actor(live_ws, actor_ws, session_state), name="live->actor"
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
            if not (down.done() and not down.cancelled() and down.exception() is None
                    and down.result()):
                # The actor hung up, whichever pump noticed first. Close the
                # session gracefully so session.closed reports the billed seconds.
                await _close_session(live_ws)
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_live(actor_ws: WebSocket, live_ws) -> None:
    """Binary PCM16 frames from the actor → GPT-Live input audio."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->gl] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->gl] forwarded {n_frames} frames ({n_bytes} bytes)")
            await live_ws.send(json.dumps({
                "type": "session.input_audio.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->gl] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->gl] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _close_session(live_ws) -> None:
    """``session.close``, then read until ``session.closed`` for the final usage."""
    try:
        await live_ws.send(json.dumps({"type": "session.close"}))

        async def drain() -> None:
            async for raw in live_ws:
                evt = json.loads(raw)
                if evt.get("type") == "session.closed":
                    logger.info(
                        f"[gl->a] session.closed reason={evt.get('reason')} "
                        f"usage seconds={(evt.get('usage') or {}).get('seconds')}"
                    )
                    return

        await asyncio.wait_for(drain(), CLOSE_GRACE_S)
    except asyncio.TimeoutError:
        logger.warning(f"[voice] no session.closed within {CLOSE_GRACE_S:.0f}s; usage unconfirmed")
    except websockets.ConnectionClosed as exc:
        logger.info(f"[voice] GPT-Live WS closed before session.closed code={exc.code}")


class _Transcript:
    """Timed fragments from both speakers, logged as one line per utterance.

    GPT-Live emits no turn boundary: fragments follow audio cadence and the two
    speakers overlap. A speaker's line closes when the other speaker's next
    fragment starts ``TURN_GAP_MS`` after it on the session timeline; anything
    closer is the two talking at once and stays on the line it belongs to.
    Lines leave in the order they started.
    """

    def __init__(self) -> None:
        self._text = {"caller": "", "rory": ""}
        self._start = {"caller": 0, "rory": 0}
        self._end = {"caller": 0, "rory": 0}

    def add(self, speaker: str, delta: str, start_ms: int, end_ms: int) -> None:
        other = "rory" if speaker == "caller" else "caller"
        if self._text[other] and start_ms >= self._end[other] + TURN_GAP_MS:
            self._flush(other)
        if not self._text[speaker]:
            self._start[speaker] = start_ms
        self._text[speaker] += delta
        self._end[speaker] = max(self._end[speaker], end_ms)

    def flush(self) -> None:
        for speaker in sorted(("caller", "rory"), key=self._start.__getitem__):
            self._flush(speaker)

    def _flush(self, speaker: str) -> None:
        other = "rory" if speaker == "caller" else "caller"
        if self._text[other] and self._start[other] < self._start[speaker]:
            self._emit(other)
        self._emit(speaker)

    def _emit(self, speaker: str) -> None:
        text = self._text[speaker].strip()
        if text:
            label = "caller_said" if speaker == "caller" else "rory_said"
            logger.info(f"[gl->a] {label}: {text}")
        self._text[speaker] = ""


async def _pump_live_to_actor(live_ws, actor_ws: WebSocket, session_state: CallSession) -> bool:
    """GPT-Live events → audio bytes back to the actor; answer delegated tool calls.

    Output audio is forwarded delta by delta as it arrives; there is no output
    item to buffer and no barge-in to handle here, because the live model is
    full duplex and stops itself when the caller talks over it.

    A delegated Responses run arrives as ``response.event`` envelopes. Its
    function calls are complete at ``response.output_item.done`` and the run
    ends at ``response.completed``; the calls are then answered from a
    separate task so audio keeps flowing while a vendor client blocks, and
    sequentially, because calls share account state. Every result goes back
    as a ``function_call_output`` item, then one ``response.create`` continues
    the backend.

    Returns True when GPT-Live reported ``session.closed`` (its own hang-up),
    False when the loop ended another way — the actor's socket dropped or the
    vendor's did — and the session may still need closing.
    """
    transcript = _Transcript()
    pending: dict[str, list[dict]] = {}  # delegation_id -> completed function calls
    answering: set[asyncio.Task] = set()
    tool_lock = asyncio.Lock()
    n_audio_frames = 0
    n_audio_bytes = 0
    n_delegations = 0

    async def _answer(delegation_id: str, calls: list[dict]) -> None:
        async with tool_lock:
            for call in calls:
                name, call_id = call["name"], call["call_id"]
                args = json.loads(call.get("arguments") or "{}")
                # Every vendor client is blocking; a ten-second vendor timeout
                # awaited on the receive loop would be ten seconds of dead air.
                result = await asyncio.to_thread(dispatch, session_state, name, args)
                logger.info(
                    f"[gl->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}"
                )
                await live_ws.send(json.dumps({
                    "type": "response.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, default=str),
                    },
                }))
            await live_ws.send(json.dumps({"type": "response.create"}))
            logger.info(
                f"[gl->a] {len(calls)} result(s) sent for {delegation_id} — response.create"
            )

    try:
        async for raw in live_ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "session.output_audio.delta":
                audio = base64.b64decode(evt.get("delta", ""))
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[gl->a] first audio delta bytes={len(audio)}")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[gl->a] streamed {n_audio_frames} audio deltas "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if audio:
                    await actor_ws.send_bytes(audio)

            elif etype == "session.input_transcript.delta":
                transcript.add("caller", evt.get("delta", ""), evt["start_ms"], evt["end_ms"])

            elif etype == "session.output_transcript.delta":
                transcript.add("rory", evt.get("delta", ""), evt["start_ms"], evt["end_ms"])

            elif etype == "session.started":
                session = evt.get("session") or {}
                logger.info(
                    f"[gl->a] session.started id={session.get('id')} "
                    f"model={session.get('model')} expires_at={session.get('expires_at')}"
                )
                await live_ws.send(json.dumps(_greeting_instruction()))
                logger.info("[voice] sent greeting instruction (agent greets first)")

            elif etype == "session.delegation.created":
                n_delegations += 1
                delegation = evt.get("delegation") or {}
                logger.info(
                    f"[gl->a] delegation #{n_delegations} id={delegation.get('id')} "
                    f"target={delegation.get('target')} response={delegation.get('response_id')}"
                )

            elif etype == "response.event":
                delegation_id = evt.get("delegation_id") or "?"
                nested = evt.get("event") or {}
                ntype = nested.get("type", "")
                if ntype == "response.output_item.done":
                    item = nested.get("item") or {}
                    if item.get("type") == "function_call":
                        pending.setdefault(delegation_id, []).append(item)
                        logger.info(
                            f"[gl->a] backend call {item.get('name')} ({item.get('call_id')}) "
                            f"args={item.get('arguments')}"
                        )
                elif ntype == "response.completed":
                    usage = (nested.get("response") or {}).get("usage") or {}
                    logger.info(
                        f"[gl->a] backend response completed for {delegation_id} "
                        f"tokens in={usage.get('input_tokens')} out={usage.get('output_tokens')}"
                    )
                    calls = pending.pop(delegation_id, [])
                    if calls:
                        task = asyncio.create_task(_answer(delegation_id, calls))
                        answering.add(task)
                        task.add_done_callback(answering.discard)
                elif ntype == "error":
                    logger.error(f"[gl->a] backend error for {delegation_id}: {nested}")
                    raise RuntimeError(f"GPT-Live backend error: {nested}")

            elif etype == "session.usage.updated":
                logger.info(
                    f"[gl->a] usage seconds={(evt.get('usage') or {}).get('seconds')} "
                    f"context={(evt.get('context_window') or {}).get('usage_ratio')}"
                )

            elif etype == "session.closed":
                transcript.flush()
                logger.info(
                    f"[gl->a] session.closed reason={evt.get('reason')} "
                    f"usage seconds={(evt.get('usage') or {}).get('seconds')}; "
                    f"{n_delegations} delegations, {n_audio_bytes} audio bytes"
                )
                return True

            elif etype == "error":
                logger.error(f"[gl->a] GPT-Live error: {evt.get('error')}")
                raise RuntimeError(f"GPT-Live error: {evt.get('error')}")
    except websockets.ConnectionClosed as exc:
        transcript.flush()
        logger.info(
            f"[gl->a] GPT-Live WS closed code={exc.code} reason={exc.reason}; "
            f"{n_delegations} delegations, {n_audio_bytes} audio bytes"
        )
    except WebSocketDisconnect:
        transcript.flush()
        logger.info("[gl->a] actor WS closed; ending GPT-Live receive loop")
    finally:
        for task in answering:
            task.cancel()
        if answering:
            await asyncio.gather(*answering, return_exceptions=True)
    return False
