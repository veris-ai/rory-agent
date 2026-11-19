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
``session.created``, with one ``session.update`` this bridge sends and
waits on ``session.updated`` for before any audio moves. ``AGENT_PROMPT``/
``GREETING`` below are imported only so this transport asserts the same
parity contract as every other candidate
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
import time
from urllib.parse import urlencode

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

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

# Atoms errors that are about one call or one tool call, not the session
# itself — the round that hit one can't be redone, but the call is still
# live and worth continuing rather than tearing down over. Anything else is
# treated as a setup/session-level failure and raised.
NON_FATAL_ERROR_CODES = {"tool_timeout", "unknown_call_id", "result_too_large"}

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
    try:
        atoms_ws, sample_rate, call_id = await _connect_with_retry()
        logger.info(
            f"[voice] Atoms connected agent_id={ATOMS_AGENT_ID} call_id={call_id} "
            f"sample_rate={sample_rate} tools={len(TOOLS)}"
        )

        up = asyncio.create_task(_pump_actor_to_atoms(actor_ws, atoms_ws), name="actor->atoms")
        down = asyncio.create_task(
            _pump_atoms_to_actor(atoms_ws, actor_ws, session_state),
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


def _auth_headers() -> dict:
    """Built per call so a missing key fails at connect, not at import."""
    return {"Authorization": f"Bearer {os.environ['ATOMS_API_KEY']}"}


async def _connect_once():
    atoms_ws = await websockets.connect(
        _atoms_url(), max_size=None, additional_headers=_auth_headers()
    )
    try:
        created = json.loads(await atoms_ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"expected session.created, got {created.get('type')!r}")
        sample_rate = created.get("sample_rate")
        if sample_rate != ACTOR_RATE_HZ:
            raise RuntimeError(
                f"Atoms announced sample_rate={sample_rate}, expected {ACTOR_RATE_HZ} "
                f"(requested via input_audio_format/output_audio_format={AUDIO_FORMAT})"
            )
        call_id = created.get("call_id")

        # Rory's 16 tools are declared here, per session, rather than stored
        # on the agent — nothing waits on this but the handshake itself, so
        # no audio can cross the wire before the model actually has a tool
        # list to call into.
        await atoms_ws.send(json.dumps({"type": "session.update", "session": {"tools": TOOLS}}))
        updated = json.loads(await atoms_ws.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"expected session.updated, got {updated.get('type')!r}")

        return atoms_ws, sample_rate, call_id
    except Exception:
        await atoms_ws.close()
        raise


async def _connect_with_retry():
    """One retry on a failed connect or handshake — a single transient
    failure should not fail the whole call when trying again costs nothing
    but a few seconds."""
    try:
        return await _connect_once()
    except Exception as exc:
        logger.warning(f"[voice] connect/handshake failed, retrying once: {exc}")
        return await _connect_once()


async def _pump_actor_to_atoms(actor_ws: WebSocket, atoms_ws) -> None:
    """Binary PCM16 frames from the actor -> Atoms' input buffer, both at 24 kHz."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->at] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->at] forwarded {n_frames} frames ({n_bytes} bytes)")
            await atoms_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->at] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
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


async def _handle_function_call(atoms_ws, session_state: CallSession, evt: dict) -> None:
    """Run one tool call and always answer it, even if it came in broken.

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
        # Every vendor client is blocking, so this still hops to a worker
        # thread rather than running inline on the event loop — the same
        # reason every other speech-to-speech candidate in this repo does
        # it. But the await here is inline, not its own task: calls share
        # CallSession, and running two concurrently on the same session
        # races its mutation of account/attempt state. One at a time, in
        # the order they arrived, is what every other candidate that shares
        # this state does too.
        result = await asyncio.to_thread(dispatch, session_state, name, args)
    logger.info(f"[at->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}")
    await atoms_ws.send(json.dumps({
        "type": "function_call.result",
        "call_id": call_id,
        "output": json.dumps(result, default=str),
    }))


async def _pump_atoms_to_actor(
    atoms_ws, actor_ws: WebSocket, session_state: CallSession
) -> None:
    """Atoms events -> paced audio back to the actor; dispatch tool calls."""
    n_audio_frames = 0
    n_audio_bytes = 0
    paced = _PacedOutput(actor_ws)
    normal_end = False

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
                await _handle_function_call(atoms_ws, session_state, evt)

            elif etype == "session.closed":
                reason = evt.get("reason")
                if reason in NORMAL_CLOSE_REASONS:
                    logger.info(f"[at->a] session.closed reason={reason} — normal end of call")
                    normal_end = True
                    break
                raise RuntimeError(f"Atoms session closed abnormally: reason={reason!r}")

            elif etype == "error":
                code = evt.get("code")
                if code in NON_FATAL_ERROR_CODES:
                    # Scoped to one call or one tool call, not the session —
                    # that round is lost, but the call itself is still live
                    # and worth continuing rather than tearing down over.
                    logger.warning(f"[at->a] Atoms error (non-fatal): {code} {evt.get('message')}")
                    continue
                logger.error(f"[at->a] Atoms error: {code} {evt.get('message')}")
                raise RuntimeError(f"Atoms error: {code} {evt.get('message')}")
    except websockets.ConnectionClosed as exc:
        # Reaching this except at all means the socket closed without ever
        # delivering a session.closed we recognized as a normal end — the
        # loop above already breaks out on one before iteration can raise
        # here. So this is always this bridge's own connection failing, not
        # the agent ending the call, regardless of the numeric close code
        # (1000/1001/1005 included — websockets 15 deprecates .code/.reason,
        # so the close frame is read off .rcvd instead).
        close = exc.rcvd
        logger.error(
            f"[at->a] Atoms WS closed without a prior session.closed(ended); "
            f"code={close.code if close else None} reason={close.reason if close else None!r}"
        )
        raise RuntimeError(
            f"Atoms connection closed abnormally: code={close.code if close else None}"
        ) from exc
    except WebSocketDisconnect:
        logger.info("[at->a] actor WS closed; ending Atoms receive loop")
        normal_end = True
    finally:
        if normal_end:
            await paced.drain()
        await paced.aclose()
