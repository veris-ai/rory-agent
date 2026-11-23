"""Rory on Smallest AI's Atoms platform — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor
and bridges it to one Atoms realtime session per call, over the unified
websocket endpoint:

    wss://api.smallest.ai/atoms/v1/agent/connect

Atoms is a platform, not one speech-to-speech model under the hood: the
agent behind ``ATOMS_AGENT_ID`` runs a cascaded STT/LLM/TTS pipeline Atoms
assembles internally. This bridge does not need to know or care: the wire
protocol over ``/agent/connect`` is one audio-in/audio-out websocket either
way, turn-taking and barge-in are owned server-side, and what this module
adds is the ``voice_ws`` bridging, the per-call ``CallSession``, pacing
output audio to a bounded lookahead, declaring Rory's tools fresh on every
call, and the hop onto a worker thread for the blocking vendor calls.

Prompt, greeting and turn-taking settings live on the Atoms agent itself.
Rory's 16 tools do not — they are declared per session, right after
``session.created``, with a ``session.update`` this bridge sends and
resends on a timer until ``session.updated`` lists every one of them.
Caller audio is dropped rather than forwarded until that handshake
finishes, and any greeting audio Atoms streams during the handshake itself
is buffered and played once the actor connection is live, rather than lost.
``AGENT_PROMPT``/``GREETING`` below are imported only so this transport
asserts the same parity contract as every other candidate
(``rory_tools.parity.assert_shares_prompt_and_greeting``); they are read,
never transmitted.

The Veris actor speaks and listens at 24 kHz PCM16; both legs of this
connection are opened at ``pcm_24000`` too, so no resampling runs in either
direction.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from urllib.parse import urlencode

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.exceptions import InvalidStatus

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt

from .tools import TOOLS

ATOMS_BASE_URL = "wss://api.smallest.ai/atoms/v1/agent/connect"
ATOMS_AGENT_ID = os.environ.get("ATOMS_AGENT_ID", "")

# The actor's rate both ways, requested explicitly on the connection query
# string rather than negotiated — Atoms documents this as one of a fixed set
# of pcm_{8000,16000,22050,24000,44100,48000} values, and 24000 is in it, so
# both legs run at the actor's native rate with no resampling.
ACTOR_RATE_HZ = 24000
AUDIO_FORMAT = f"pcm_{ACTOR_RATE_HZ}"
BYTES_PER_SAMPLE = 2  # PCM16 mono

# webcall is Atoms' audio mode (the other documented mode, "chat", is text
# over input_text.send/transcript — not this bridge).
CONNECT_MODE = "webcall"

# Never let queued-but-unsent output audio get more than this far ahead of
# when it should actually be heard.
OUTPUT_LOOKAHEAD_S = 0.3

# session.closed reasons that mean the call ended the way it was supposed
# to — anything else (a dropped connection, a setup failure) is this
# bridge's own infrastructure failing, not the agent hanging up on the
# caller, and is raised rather than treated as a normal end.
NORMAL_CLOSE_REASONS = {"client_requested", "ended"}

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _atoms_url() -> str:
    """Built per call so a missing agent id fails at connect, not at import."""
    return ATOMS_BASE_URL + "?" + urlencode({
        "agent_id": ATOMS_AGENT_ID,
        "mode": CONNECT_MODE,
        "input_audio_format": AUDIO_FORMAT,
        "output_audio_format": AUDIO_FORMAT,
    })


class _PacedOutput:
    """Forwards audio to the actor at real-time pace, never more than
    ``OUTPUT_LOOKAHEAD_S`` ahead of when it should actually play.

    A barge-in means everything still queued is audio the caller will never
    hear and should not hear — ``drop()`` discards it outright rather than
    draining it. A normal end of call is the opposite: ``drain()`` lets
    whatever is left (the agent's own goodbye) finish before the connection
    closes.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self._actor_ws = actor_ws
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="paced-output")

    def push(self, pcm: bytes) -> None:
        self._queue.put_nowait(pcm)

    def drop(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        # started + sent_s is how far playback has been scheduled out to; a
        # silence gap means real time has already passed that point, and the
        # clock has to restart there rather than keep compounding off a
        # start time from before the gap — otherwise everything after the
        # first pause gets scheduled too early and drains out at 2x.
        started: float | None = None
        sent_s = 0.0
        while True:
            pcm = await self._queue.get()
            try:
                now = time.monotonic()
                if started is None or now >= started + sent_s:
                    started, sent_s = now, 0.0
                await self._actor_ws.send_bytes(pcm)
                sent_s += len(pcm) / BYTES_PER_SAMPLE / ACTOR_RATE_HZ
                ahead = sent_s - (time.monotonic() - started)
                if ahead > OUTPUT_LOOKAHEAD_S:
                    await asyncio.sleep(ahead - OUTPUT_LOOKAHEAD_S)
            finally:
                self._queue.task_done()


class _AtomsGate:
    """Lets the actor->Atoms pump start draining the actor's stream before
    the Atoms connection even exists.

    Without this, caller audio sent during the connect+handshake window
    (which can take several seconds) just sits unread in the actor
    connection's own buffer, then floods into Atoms the instant the pump
    finally starts — a burst that looks nothing like real-time speech.
    Reading it continuously and dropping it until ``open()`` is called
    avoids that burst ever building up.
    """

    def __init__(self) -> None:
        self.ws = None
        self.ready = asyncio.Event()

    def open(self, ws) -> None:
        self.ws = ws
        self.ready.set()


class _SetupError(RuntimeError):
    """A connect/handshake failure, with enough context to log and to
    decide whether retrying is worth it at all."""

    def __init__(self, message: str, *, call_id: str | None = None, code: str | None = None,
                 retryable: bool = True) -> None:
        super().__init__(message)
        self.call_id = call_id
        self.code = code
        self.retryable = retryable


# Rejections that mean retrying would just fail the same way again:
# tools stored on the agent refuse a per-session declaration outright.
NON_RETRYABLE_TOOL_ERROR_CODES = {"client_tools_managed_by_config"}

# How long a session.update declaring tools may go unacknowledged before it
# is resent, and the overall deadline — covering both session.created and
# every tool getting acknowledged — across every attempt. The send cap is
# one under the gateway's own pre-worker control-message buffer; one more
# send than that is refused with a too_many_updates error.
TOOLS_RESEND_S = 3.0
TOOLS_READY_TIMEOUT_S = 8.0
MAX_TOOL_UPDATE_SENDS = 7

# A failed connect or handshake is retried once, after a jittered delay
# rather than immediately — a transient failure on Atoms' side is more
# likely to have cleared a couple of seconds later than instantly.
RETRY_JITTER_MIN_S = 2.0
RETRY_JITTER_MAX_S = 4.0

_TOOL_NAMES = frozenset(t["name"] for t in TOOLS)


def _retry_delay() -> float:
    return random.uniform(RETRY_JITTER_MIN_S, RETRY_JITTER_MAX_S)


async def _recv_json(ws, deadline: float, what: str) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _SetupError(f"no {what} before the connect deadline")
    try:
        raw = await asyncio.wait_for(ws.recv(), remaining)
    except asyncio.TimeoutError as exc:
        raise _SetupError(f"no {what} within the connect deadline") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _SetupError(f"{what}: Atoms sent non-JSON: {raw!r}") from exc


async def _declare_tools(atoms_ws, deadline: float, call_id: str | None) -> list[bytes]:
    """Send ``session.update`` declaring Rory's tools; resend on a timer
    until every one is acked or the deadline passes.

    Resending is purely time-gated (``now - last_send >= TOOLS_RESEND_S``):
    earlier this resent on every incoming event instead, including routine
    greeting audio streamed during the handshake, which could burn through
    the whole send budget in under a second and fail setup long before a
    real (if slow) acknowledgment would have arrived. A duplicate send is
    harmless either way — Atoms applies a declaration with replace
    semantics, not by accumulating.

    Any ``output_audio.delta`` seen here — the agent's greeting, which can
    start before tools are even acknowledged — is buffered and returned
    rather than dropped, so the caller still hears it once the actor
    connection goes live. A failed attempt's buffer is simply discarded
    with it; a retry gets its own, from its own fresh session.created.
    """
    buffered_audio: list[bytes] = []
    sends = 0
    last_send = 0.0
    while True:
        now = time.monotonic()
        if sends < MAX_TOOL_UPDATE_SENDS and now - last_send >= TOOLS_RESEND_S:
            await atoms_ws.send(json.dumps({"type": "session.update", "tools": TOOLS}))
            sends += 1
            last_send = now
        remaining = deadline - now
        if remaining <= 0:
            raise _SetupError(
                f"no session.updated for all tools after {sends} session.update(s)", call_id=call_id
            )
        try:
            raw = await asyncio.wait_for(atoms_ws.recv(), min(TOOLS_RESEND_S, remaining))
        except asyncio.TimeoutError:
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "session.updated":
            if _TOOL_NAMES.issubset(evt.get("tools") or []):
                return buffered_audio
            continue  # an ack for a resend, or a partial list — keep waiting
        if etype == "error":
            code = evt.get("code")
            raise _SetupError(
                f"Atoms rejected the tool declaration: {code} {evt.get('message')}",
                call_id=call_id, code=code, retryable=code not in NON_RETRYABLE_TOOL_ERROR_CODES,
            )
        if etype == "output_audio.delta":
            audio = base64.b64decode(evt.get("audio", ""))
            if audio:
                buffered_audio.append(audio)
        # Anything else (transcript, agent_start_talking, etc.) during
        # setup is not actionable here and is dropped.


async def _connect_once():
    try:
        atoms_ws = await websockets.connect(
            _atoms_url(), max_size=None, additional_headers=_auth_headers()
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        raise _SetupError(
            f"Atoms upgrade failed: HTTP {status}", code=str(status), retryable=not (400 <= status < 500)
        ) from exc
    except Exception as exc:
        raise _SetupError(f"Atoms connect failed: {type(exc).__name__}: {exc}") from exc

    call_id = None
    try:
        deadline = time.monotonic() + TOOLS_READY_TIMEOUT_S
        created = await _recv_json(atoms_ws, deadline, "session.created")
        if created.get("type") != "session.created":
            raise _SetupError(f"expected session.created, got {created.get('type')!r}")
        sample_rate = created.get("sample_rate")
        call_id = created.get("call_id")
        if sample_rate != ACTOR_RATE_HZ:
            raise _SetupError(
                f"Atoms announced sample_rate={sample_rate}, expected {ACTOR_RATE_HZ} "
                f"(requested via input_audio_format/output_audio_format={AUDIO_FORMAT})",
                call_id=call_id,
            )
        early_audio = await _declare_tools(atoms_ws, deadline, call_id)
        return atoms_ws, sample_rate, call_id, early_audio
    except _SetupError as exc:
        if exc.call_id is None:
            exc.call_id = call_id
        await atoms_ws.close()
        raise
    except Exception as exc:
        await atoms_ws.close()
        raise _SetupError(str(exc), call_id=call_id) from exc


async def _connect_with_retry():
    """One retry on a failed connect or handshake, after a jittered delay —
    a single transient failure should not fail the whole call when trying
    again costs only a few seconds. Not retried when the failure is one a
    second attempt cannot fix: a 4xx on the upgrade itself, or tools
    rejected because they are stored on the agent instead."""
    try:
        return await _connect_once()
    except _SetupError as exc:
        logger.warning(f"[voice] connect/handshake failed call_id={exc.call_id} code={exc.code}: {exc}")
        if not exc.retryable:
            raise
        await asyncio.sleep(_retry_delay())
        return await _connect_once()


def _auth_headers() -> dict:
    """Built per call so a missing key fails at connect, not at import."""
    return {"Authorization": f"Bearer {os.environ['ATOMS_API_KEY']}"}


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection <-> one Atoms session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other
    speech-to-speech candidates build one per call: it holds the account
    ``verify_caller`` matched and dies with the call, so identity never
    becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()

    t_start = time.monotonic()
    atoms_ws = None
    # Starts draining the actor's stream immediately, dropping frames until
    # gate.open() — so nothing piles up during connect+handshake only to
    # burst into Atoms all at once the moment it's ready.
    gate = _AtomsGate()
    up = asyncio.create_task(_pump_actor_to_atoms(actor_ws=actor_ws, gate=gate), name="actor->atoms")
    try:
        try:
            atoms_ws, sample_rate, call_id, early_audio = await _connect_with_retry()
        except Exception:
            up.cancel()
            await asyncio.gather(up, return_exceptions=True)
            raise
        gate.open(atoms_ws)
        logger.info(
            f"[voice] Atoms connected agent_id={ATOMS_AGENT_ID} call_id={call_id} "
            f"sample_rate={sample_rate} tools={len(TOOLS)}"
        )

        down = asyncio.create_task(
            _pump_atoms_to_actor(atoms_ws, actor_ws, session_state, early_audio),
            name="atoms->actor",
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
        if atoms_ws is not None:
            await atoms_ws.close()
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_atoms(actor_ws: WebSocket, gate: _AtomsGate) -> None:
    """Binary PCM16 frames from the actor -> Atoms' input buffer, both at 24 kHz.

    Frames arriving before ``gate`` opens are read (so they never queue up
    behind us) and dropped (there is nowhere ready to forward them yet)."""
    n_frames = 0
    n_bytes = 0
    n_dropped = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            if not gate.ready.is_set():
                n_dropped += 1
                continue
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->at] first frame bytes={len(frame)} dropped_before_ready={n_dropped}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->at] forwarded {n_frames} frames ({n_bytes} bytes)")
            await gate.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->at] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes, {n_dropped} dropped before ready): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead
        # of a binary one — a protocol mismatch on the actor side, not a bug
        # here.
        logger.error(
            f"[a->at] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _handle_function_call(
    atoms_ws, session_state: CallSession, evt: dict, tool_lock: asyncio.Lock
) -> None:
    """Run one tool call and always answer it, even if it came in broken.

    Dispatch itself is serialized behind ``tool_lock`` — calls share one
    ``CallSession``, and running two at once would race its mutation of
    account/attempt state — but acquiring that lock does not block the
    caller: each call runs as its own task, so the receive loop keeps
    reading audio and interruption events the whole time one is in flight.
    ``asyncio.Lock`` serves waiters in the order they asked for it, which
    is the order calls arrived in.

    A barge-in cutting generation off mid-argument is possible on any
    realtime tool-calling wire; a call whose arguments never finished
    streaming is answered with an error result instead of being left
    hanging, since the model has no way to recover from a function call it
    never heard back from.
    """
    name = evt["name"]
    call_id = evt["call_id"]
    raw_args = evt.get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        logger.warning(f"[at->a] tool {name} ({call_id}) arguments truncated: {raw_args!r}")
        result = {"error": "the arguments for this call were cut off and could not be read"}
    else:
        async with tool_lock:
            # Every vendor client is blocking, so this still hops to a
            # worker thread rather than running inline on the event loop —
            # the same reason every other speech-to-speech candidate in
            # this repo does it.
            result = await asyncio.to_thread(dispatch, session_state, name, args)
    output = json.dumps(result, default=str, ensure_ascii=False)
    logger.info(f"[at->a] tool {name} ({call_id}) -> {output}")
    await atoms_ws.send(json.dumps({
        "type": "function_call.result",
        "call_id": call_id,
        "output": output,
    }))


async def _pump_atoms_to_actor(
    atoms_ws, actor_ws: WebSocket, session_state: CallSession, early_audio: list[bytes]
) -> None:
    """Atoms events -> paced audio back to the actor; dispatch tool calls."""
    n_audio_frames = 0
    n_audio_bytes = 0
    paced = _PacedOutput(actor_ws)
    for audio in early_audio:
        n_audio_frames += 1
        n_audio_bytes += len(audio)
        paced.push(audio)
    tool_lock = asyncio.Lock()
    pending_calls: set[asyncio.Task] = set()
    normal_end = False

    try:
        try:
            async for raw in atoms_ws:
                evt = json.loads(raw)
                etype = evt.get("type", "")

                if etype == "output_audio.delta":
                    audio = base64.b64decode(evt.get("audio", ""))
                    n_audio_frames += 1
                    n_audio_bytes += len(audio)
                    if n_audio_frames == 1:
                        logger.info(f"[at->a] first audio delta bytes={len(audio)}")
                    elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                        logger.info(
                            f"[at->a] streamed {n_audio_frames} audio deltas "
                            f"({n_audio_bytes} bytes) so far this session"
                        )
                    if audio:
                        paced.push(audio)

                elif etype == "agent_start_talking":
                    logger.info("[at->a] agent_start_talking")

                elif etype == "agent_stop_talking":
                    logger.info("[at->a] agent_stop_talking")

                elif etype == "interruption":
                    logger.info("[at->a] interruption (barge-in) — dropping queued output")
                    paced.drop()

                elif etype in ("transcript", "transcript.delta"):
                    continue  # logged at debug volume only; not needed to bridge the call

                elif etype == "function_call":
                    task = asyncio.create_task(
                        _handle_function_call(atoms_ws, session_state, evt, tool_lock)
                    )
                    pending_calls.add(task)
                    task.add_done_callback(pending_calls.discard)

                elif etype == "session.closed":
                    reason = evt.get("reason")
                    if reason in NORMAL_CLOSE_REASONS:
                        logger.info(f"[at->a] session.closed reason={reason} — normal end of call")
                        normal_end = True
                        break
                    raise RuntimeError(f"Atoms session closed abnormally: reason={reason!r}")

                elif etype == "error":
                    # Setup-time errors (a bad connect, a rejected tool
                    # declaration) are raised in _connect_once/_declare_tools,
                    # before this loop ever starts. Everything reaching here
                    # is scoped to one call or one tool call — tool_timeout,
                    # unknown_call_id, result_too_large and the like — and
                    # the call is still live, so it is logged rather than
                    # torn down over.
                    logger.warning(f"[at->a] Atoms error: {evt.get('code')} {evt.get('message')}")
        except websockets.ConnectionClosed:
            # Raised for a genuinely abnormal disconnect (no close frame at
            # all); handled uniformly below, same as the silent case.
            pass
        except WebSocketDisconnect:
            logger.info("[at->a] actor WS closed; ending Atoms receive loop")
            normal_end = True

        # Falls through to here whenever the loop above did not already
        # explicitly raise: a normal break (normal_end already True), the
        # atoms_ws iterator stopping with NO exception at all — which is
        # what websockets 15 does for close codes 1000, 1001 and 1005, not
        # just abnormal ones — or a ConnectionClosed caught and swallowed
        # just above (an abrupt reset, no close frame received). None of
        # those numeric codes says the agent ended the call on its own;
        # only an actual session.closed(ended) we saw does, and if we are
        # here without one, this is this bridge's own connection failing.
        if not normal_end:
            close_code = getattr(atoms_ws, "close_code", None)
            close_reason = getattr(atoms_ws, "close_reason", None)
            logger.error(
                f"[at->a] Atoms WS ended without a prior session.closed(ended); "
                f"code={close_code} reason={close_reason!r}"
            )
            raise RuntimeError(f"Atoms connection closed abnormally: code={close_code}")
    finally:
        if normal_end:
            await paced.drain()
        await paced.aclose()
        # Let in-flight calls finish and answer rather than cancelling them —
        # "answer every tool call" has to hold even for one still running
        # when the session ends, not just the common case.
        if pending_calls:
            await asyncio.gather(*pending_calls, return_exceptions=True)
