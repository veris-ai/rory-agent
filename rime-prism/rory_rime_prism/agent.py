"""Rory on Rime's Prism speech-to-speech model — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens one Prism session through the Rime Python SDK,
with the shared prompt and Rory's 16 tools, and runs it with the SDK's
conversation runner: the actor's audio is the runner's source, and a paced
player that writes to the actor is its playback.

Prism owns turn-taking. It decides when the caller has finished, and caller
speech over a reply interrupts it. Its turn settings stay on Prism's defaults
(eagerness ``auto``, a 5 s end-of-turn timeout), as Rime recommends.

Everything below the transport is shared with the other candidates: same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. The transport is the only thing allowed to differ.

Sample-rate note: the actor speaks and listens at 24 kHz PCM16. Prism listens at
16 kHz, and the SDK resamples the actor's audio itself. Prism replies at 24 kHz,
so the reply leg is not resampled.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from collections.abc import Sequence

from loguru import logger
from rimelabs_sdk import Rime, RimeInputError
from rimelabs_sdk.realtime import (
    AudioChunk,
    FaultEvent,
    InputReady,
    ItemTruncated,
    OutputRef,
    PCMFormat,
    RealtimeAdmissionTimeout,
    RealtimeSession,
    ResponseEnded,
    ResponseStarted,
    RimeRealtimeError,
    SessionEvent,
    SpeechStarted,
    SpeechStopped,
    TextDone,
    ToolCall,
    ToolCallRef,
    TranscriptFinal,
    run_conversation,
)
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

PRISM_VOICE = os.environ.get("PRISM_VOICE", "aurora")

# The actor's rate both ways; Prism replies at the same rate.
ACTOR_RATE_HZ = 24000
PCM16_BYTES_PER_MS = ACTOR_RATE_HZ * 2 // 1000

# Prism sends a reply faster than it plays. Audio handed to the actor is
# committed (nothing claws it back), so the player sends it in slices and stays
# at most PLAYBACK_LEAD_S ahead of real-time playback. That keeps a barge-in
# able to cut the reply, and keeps the heard figure in Prism's playback reports
# close to what the caller heard.
PLAYBACK_SLICE_BYTES = 100 * PCM16_BYTES_PER_MS
PLAYBACK_LEAD_S = 0.5

# Prism refuses a tool continuation when the caller spoke first or a wait ran out.
SUPERSEDED_CODES = {"tool_continuation_unavailable", "response_superseded"}

# Rime's slow-tool pattern: a few seconds of silence after the tool's phrase
# sounds like a dropped line, so a lookup still running after 4 s gets a short
# line, and another 10 s after that. Only create_payment_arrangement (~9 s) runs
# that long.
STILL_WORKING = "Tell the caller in one short sentence that you are still working on it."
STILL_WORKING_WAITS_S = (4.0, 10.0)

# Replies that answer the caller with everything in the history: a spoken turn,
# or the continuation of a tool round. The pod's own nudges are left out.
ANSWERING_CAUSES = {"speech", "tools"}

AGENT_PROMPT = load_agent_prompt()

# Prism has no welcome message: a nudge makes it speak first, with the same
# wording the other speech-to-speech candidates send. Prism generates the audio
# itself, so like them it can paraphrase.
GREETING_INSTRUCTION = (
    "Open the call by greeting the caller now. Say exactly: "
    f'"{GREETING}" Then stop and wait for the caller to respond.'
)


def _connect_options() -> dict:
    """The session settings, pinned so a run is reproducible.

    The date context is appended to the system prompt rather than baked into
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot. Interruption and turn detection are left
    out so Prism uses its own defaults.
    """
    return {
        "voice": PRISM_VOICE,
        "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
        "tools": TOOLS,
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Prism session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other transports build
    one per call: it holds the account ``verify_caller`` matched and dies with
    the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    call_session = CallSession()
    t_start = time.monotonic()
    try:
        async with Rime() as client, client.realtime.connect(**_connect_options()) as session:
            info = session.info
            logger.info(
                f"[voice] Prism session_id={info.session_id} request_id={info.request_id} "
                f"voice={info.voice} turn_detection={info.turn_detection} "
                f"interrupt_on_speech={info.interrupt_on_speech} "
                f"tool_waits={info.tool_result_timeout_s}/{info.tool_continuation_timeout_s}s "
                f"tools={len(TOOLS)}"
            )
            rounds = ToolRounds(session, call_session)
            conversation = asyncio.create_task(
                run_conversation(
                    session,
                    source=ActorSource(actor_ws),
                    playback=ActorPlayback(actor_ws),
                    on_event=lambda event: _on_event(event, rounds),
                ),
                name="conversation",
            )
            control = asyncio.create_task(_control(session, rounds), name="control")
            tasks = [conversation, control]
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


def _on_event(event: SessionEvent, rounds: ToolRounds) -> None:
    """The runner's per-event callback: log, and hand tool work to the control task.

    It runs inline in the runner's events loop, so it must not block.
    """
    payload = event.payload
    rounds.notice(payload)
    if isinstance(payload, ResponseEnded):
        logger.info(
            f"[prism] response {payload.response.response_id} ended "
            f"status={payload.status} reason={payload.reason}"
        )
    elif isinstance(payload, ResponseStarted):
        logger.info(f"[prism] response {payload.response.response_id} started cause={payload.cause}")
    elif isinstance(payload, SpeechStarted):
        logger.info("[prism] vad: speech_started")
    elif isinstance(payload, SpeechStopped):
        logger.info(f"[prism] vad: speech_stopped end_reason={payload.end_reason}")
    elif isinstance(payload, TranscriptFinal):
        logger.info(f"[prism] caller_said: {payload.text}")
    elif isinstance(payload, TextDone):
        logger.info(f"[prism] rory_said: {payload.text}")
    elif isinstance(payload, ItemTruncated):
        logger.info(f"[prism] interrupted reply keeps: {payload.kept_text!r}")
    elif isinstance(payload, FaultEvent):
        # Most faults do not end the session (busy refusals arrive here while the SDK
        # retries). A session-scope fault ends it, and the runner raises that.
        fault = payload.error
        logger.warning(
            f"[prism] error {fault.code} scope={fault.scope} "
            f"correlation_id={fault.correlation_id}: {fault.message}"
        )


async def _control(session: RealtimeSession, rounds: ToolRounds) -> None:
    """Speak the greeting, then run every tool round for the rest of the call."""
    try:
        await session.request_reply(instruction=GREETING_INSTRUCTION)
    except RimeRealtimeError as exc:
        # The caller spoke first; Prism answers them instead.
        logger.info(f"[prism] greeting skipped: {exc}")
    await rounds.run()


class ToolRounds:
    """Rory's tool rounds on one Prism session, run the way Rime's tool guides run them.

    The vendor clients are blocking and share account state, so calls run in order
    on a worker thread. The runner keeps reading Prism's events meanwhile. A reply's
    ``tool.call`` events arrive before its ``response.ended``, so by the time this
    loop reads the end of a reply, every call it made has its result, and the reply
    is continued.

    A round can close before it uses its result: the caller spoke first (even an
    "okay"), the reply that made the call was cancelled, or a still-working line
    closed it. Prism keeps the result but mentions it only if asked, so the caller
    would sit in silence. Once Prism is idle with no newer round open, a nudge
    reports the result, unless Prism has already answered the caller since the
    result went in. A nudge closes any open round, so one taken after Prism has
    started answering the caller closes the round that answer opened, and the
    caller hears answers twice. Prism takes a nudge only once it is idle, so every
    nudge (a report or a still-working line) is withdrawn if Prism starts
    answering the caller first. A call that has run has acted, so it is never run
    again here.
    """

    def __init__(self, session: RealtimeSession, call_session: CallSession) -> None:
        self._session = session
        self._call_session = call_session
        self._events: asyncio.Queue[ToolCall | ResponseEnded | InputReady] = asyncio.Queue()
        self._answers = 0  # replies so far that answered with the whole history
        # Each round with every result in: its parent reply -> its last call, and
        # the answer count when that result went in.
        self._open: dict[str, tuple[ToolCallRef, int]] = {}
        self._unreported: tuple[ToolCallRef, int] | None = None
        self._nudges: set[asyncio.Task] = set()  # nudges waiting for Prism to be idle

    def notice(self, payload: object) -> None:
        """Take one session event, inline in the runner's events loop."""
        if isinstance(payload, ResponseStarted) and payload.cause in ANSWERING_CAUSES:
            self._answers += 1
            for nudge in self._nudges:
                nudge.cancel()
        elif isinstance(payload, (ToolCall, ResponseEnded, InputReady)):
            self._events.put_nowait(payload)

    async def run(self) -> None:
        while True:
            payload = await self._events.get()
            if isinstance(payload, ToolCall):
                await self._run_call(payload)
            elif isinstance(payload, ResponseEnded):
                await self._end_round(payload)
            elif self._unreported is not None and not self._open and self._events.empty():
                # Prism is idle, and nothing newer is waiting.
                await self._report()

    async def _run_call(self, call: ToolCall) -> None:
        lines = asyncio.create_task(self._say_still_working())
        try:
            result = await asyncio.to_thread(dispatch, self._call_session, call.name, dict(call.arguments))
        finally:
            lines.cancel()
        logger.info(f"[prism] tool {call.name} ({call.call.call_id}) -> {json.dumps(result)}")
        await self._session.submit_tool_result(call.call, json.dumps(result))
        self._open[call.call.response_id] = (call.call, self._answers)

    async def _end_round(self, ended: ResponseEnded) -> None:
        response_id = ended.response.response_id
        last = self._open.pop(response_id, None)
        if last is None:
            return
        if ended.status != "completed":
            logger.info(f"[prism] round of {response_id} dropped: the reply was {ended.status}")
        elif await _continue(self._session, ended):
            return
        self._unreported = last

    async def _report(self) -> None:
        call, answers_then = self._unreported
        self._unreported = None
        if self._answers > answers_then:
            logger.info(f"[prism] result of {call.call_id} not reported: Prism has answered since")
            return
        if await self._nudge(tool_call=call):
            logger.info(f"[prism] reported the result of {call.call_id}")

    async def _say_still_working(self) -> None:
        """Fill a slow lookup with short lines. The first one closes the tool round."""
        for wait in STILL_WORKING_WAITS_S:
            await asyncio.sleep(wait)
            if not await self._nudge(instruction=STILL_WORKING):
                return
            logger.info("[prism] said a still-working line")

    async def _nudge(self, **request) -> bool:
        """Ask Prism for a reply once it is idle. True when the reply started.

        False when the caller spoke first (Prism answers them with every result in
        its history), the wait ran out, or Prism started answering the caller
        before it took the nudge, which withdraws it.
        """
        nudge = asyncio.ensure_future(self._session.request_reply(**request))
        self._nudges.add(nudge)
        try:
            await asyncio.wait([nudge])
        finally:
            self._nudges.discard(nudge)
            nudge.cancel()
        if nudge.cancelled():
            logger.info(f"[prism] nudge withdrawn, Prism is answering the caller: {request}")
            return False
        error = nudge.exception()
        if isinstance(error, (RimeRealtimeError, RealtimeAdmissionTimeout)):
            logger.info(f"[prism] nudge refused: {error}")
            return False
        if error is not None:
            raise error
        return True


async def _continue(session: RealtimeSession, ended: ResponseEnded) -> bool:
    """Continue the reply that made the calls. False when the round has closed."""
    try:
        await session.continue_reply(ended.response)
    except (RimeInputError, RealtimeAdmissionTimeout) as exc:
        logger.info(f"[prism] round of {ended.response.response_id} dropped: {exc}")
        return False
    except RimeRealtimeError as exc:
        if exc.fault.code not in SUPERSEDED_CODES:
            raise
        logger.info(f"[prism] round of {ended.response.response_id} dropped: {exc}")
        return False
    return True


class ActorSource:
    """The actor's binary PCM16 frames as the runner's audio source."""

    def __init__(self, actor_ws: WebSocket) -> None:
        self._ws = actor_ws
        self._frames = 0

    async def read(self) -> AudioChunk | None:
        try:
            frame = await self._ws.receive_bytes()
        except WebSocketDisconnect as exc:
            # The actor hanging up is the call ending; the runner closes the session.
            logger.info(f"[a->prism] actor disconnected after {self._frames} frames: code={exc.code}")
            return None
        self._frames += 1
        if self._frames == 1:
            logger.info(f"[a->prism] first frame bytes={len(frame)}")
        return AudioChunk(data=frame, format=PCMFormat(sample_rate=ACTOR_RATE_HZ))


class _Output:
    """One reply message's audio, queued for the actor in the order the runner opened it."""

    def __init__(self, playback: ActorPlayback, ref: OutputRef) -> None:
        self._playback = playback
        self.ref = ref
        self.buffer = bytearray()
        self.sealed = False
        self.sent_bytes = 0
        self.played: asyncio.Future[float] = asyncio.get_running_loop().create_future()

    def write(self, chunk: AudioChunk) -> None:
        self.buffer += chunk.data
        self._playback.wake()

    def end(self) -> None:
        self.sealed = True
        self._playback.wake()

    async def completion(self) -> float:
        return await asyncio.shield(self.played)

    def resolve(self) -> None:
        """Report everything sent as heard: the actor plays all it was handed."""
        if not self.played.done():
            self.played.set_result(self.sent_bytes / PCM16_BYTES_PER_MS)


class ActorPlayback:
    """The runner's playback: reply audio to the actor at real-time pace.

    The actor is the caller's ear, so the playback clock is the audio sent to it.
    A reply finishes playing once its last slice has had time to play. An
    interrupted reply drops its queued audio, and the caller heard what was sent.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self._ws = actor_ws
        self._queue: deque[_Output] = deque()
        self._open: dict[OutputRef, _Output] = {}
        self._wakeup = asyncio.Event()
        self._plays_until = 0.0
        self._pump = asyncio.create_task(self._send(), name="playback")

    def wake(self) -> None:
        self._wakeup.set()

    def open(self, output: OutputRef) -> _Output:
        stream = _Output(self, output)
        self._queue.append(stream)
        self._open[output] = stream
        stream.played.add_done_callback(lambda _: self._open.pop(output, None))
        self.wake()
        return stream

    def interrupt(self, outputs: Sequence[OutputRef]) -> None:
        for ref in outputs:
            stream = self._open.get(ref)
            if stream is None:
                continue
            stream.buffer.clear()
            stream.resolve()
            if stream in self._queue:
                self._queue.remove(stream)
        self.wake()

    async def stop(self) -> None:
        self._pump.cancel()
        try:
            await self._pump
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        for stream in list(self._open.values()):
            stream.buffer.clear()
            stream.resolve()
        self._queue.clear()

    async def _send(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._wakeup.clear()
            if not self._queue:
                await self._wakeup.wait()
                continue
            stream = self._queue[0]
            if stream.buffer:
                piece = bytes(stream.buffer[:PLAYBACK_SLICE_BYTES])
                del stream.buffer[:PLAYBACK_SLICE_BYTES]
                await self._ws.send_bytes(piece)
                stream.sent_bytes += len(piece)
                now = loop.time()
                self._plays_until = max(now, self._plays_until) + len(piece) / (PCM16_BYTES_PER_MS * 1000)
                ahead = self._plays_until - now
                if ahead > PLAYBACK_LEAD_S:
                    await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
            elif stream.sealed:
                self._queue.popleft()
                loop.call_at(max(loop.time(), self._plays_until), stream.resolve)
            else:
                await self._wakeup.wait()
