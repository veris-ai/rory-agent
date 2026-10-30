"""Rory on Amazon Connect's voice stack: the pod that stands where Connect would.

In the Connect pattern a call reaches Amazon Connect, Connect streams the caller's
audio to Lex, Lex transcribes each turn, the agent in AgentCore Runtime reasons on
Bedrock and calls its tools over MCP, and the reply streams back for Connect to
speak. A Lambda initializes the agent when the call connects.

The bench does not dial a phone, so this process plays Connect's part over the
Veris ``voice_ws`` socket:

- When the call connects it invokes the init Lambda, which writes the call's
  session record and warms its AgentCore session, while Rory speaks the greeting.
- It streams all of the caller's audio to Lex over one ``StartConversation`` stream
  for the whole call. Lex decides when each turn is over and sends its transcript.
- It sends each transcript to the AgentCore Runtime and reads the reply as it is
  generated, speaking it with Polly, the engine behind Connect's voices, one
  sentence at a time: the caller hears the first sentence while the rest is still
  being written.
- It owns barge-in, as the other cascades do: voice onset while Rory is speaking
  cuts the reply, and the next turn tells the runtime what the caller heard.

The runtime returns each turn as the model saw it; this process exports that as the
agent's trace (see ``trace``).

Sample-rate note: the actor speaks and listens at 24 kHz PCM16. Lex's streaming API
takes 8 kHz linear PCM, the telephone rate Connect sends it, and Polly's PCM output
is 16 kHz, so audio is resampled at both boundaries.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import boto3
from aws_sdk_lex_runtime_v2.client import AsyncLexRuntimeV2Client
from aws_sdk_lex_runtime_v2.config import AsyncLexRuntimeV2Config
from aws_sdk_lex_runtime_v2.models import (
    AudioInputEvent,
    ConfigurationEvent,
    PlaybackCompletionEvent,
    SessionState,
    StartConversationInput,
    StartConversationRequestEventStreamAudioInputEvent,
    StartConversationRequestEventStreamConfigurationEvent,
    StartConversationRequestEventStreamPlaybackCompletionEvent,
    StartConversationResponseEventStreamTextResponseEvent,
    StartConversationResponseEventStreamTranscriptEvent,
)
from loguru import logger
from smithy_aws_core.identity.components import AWSCredentialsIdentity
from smithy_aws_core.identity.static import StaticCredentialsResolver
from smithy_http.aio.crt import AWSCRTHTTPClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools.prompt import GREETING
from rory_tools.vad import SileroVad

from . import trace
from .contract import (
    ATTR_FINERACT_API_BASE,
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    ATTR_STRIPE_API_BASE,
    ATTR_STRIPE_API_KEY,
    ATTR_TOOL_PREAMBLE,
    CALL_CONNECTED,
    INTERRUPTED_MARKER,
)

LEX_LOCALE_ID = os.environ.get("LEX_LOCALE_ID", "en_US")
POLLY_VOICE_ID = os.environ.get("POLLY_VOICE_ID", "Joanna")
POLLY_ENGINE = os.environ.get("POLLY_ENGINE", "neural")
# Demo only: ask the model to say what it is doing before a tool call (see
# runtime.TOOL_PREAMBLE). Off on the bench, where every candidate gets the same prompt.
TOOL_PREAMBLE = os.environ.get("RORY_TOOL_PREAMBLE") == "1"

ACTOR_RATE_HZ = 24000
LEX_RATE_HZ = 8000
POLLY_RATE_HZ = 16000
BYTES_PER_S = ACTOR_RATE_HZ * 2
BYTES_PER_MS = BYTES_PER_S // 1000

LEX_AUDIO = f"audio/lpcm; sample-rate={LEX_RATE_HZ}; sample-size-bits=16; channel-count=1; is-big-endian=false"
LEX_SESSION_ATTRIBUTES = {
    # The silence that ends a caller's turn. Lex's default is over a second.
    "x-amz-lex:audio:end-timeout-ms:*:*": os.environ.get("LEX_END_TIMEOUT_MS", "400"),
    # How long Lex waits for speech before giving up on a turn. Its default of a
    # few seconds ends the stream while the caller listens to a long reply.
    "x-amz-lex:audio:start-timeout-ms:*:*": "900000",
}

# Lex ends a conversation at 15 minutes, a hard quota. The pod starts a new one
# on the same session before then, in a pause, so the call can run longer.
LEX_STREAM_MAX_S = 14 * 60

# Voice onset after this much silence counts as the caller starting to speak again.
END_OF_SPEECH_S = 0.8

# Replies go out in slices paced against playback, so a barge-in has something
# to cut. Sent as it lands, the whole reply would sit in the actor's buffer.
PLAYBACK_CHUNK_BYTES = 500 * BYTES_PER_MS
PLAYBACK_LEAD_S = 1.0

# The end of a sentence: where a reply is cut into pieces to speak. The first
# piece of a reply may end at a clause instead, once it is long enough to say.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
CLAUSE_END = re.compile(r"[,;:—]\s*")
FIRST_CLAUSE_MIN_CHARS = 20

# Set once per process by resolve_vendor, after the credential check.
_polly: Any
_lambda: Any
_agentcore: Any
_identity: AWSCredentialsIdentity
_twins: dict[str, str]


def resolve_vendor() -> None:
    """Build the AWS clients and read this attempt's twins, once per process.

    The AWS key has its own names rather than the SDK's defaults: on the bench it
    sits in candidate-env, which every candidate on the bench receives, and the
    default names would hand it to any other candidate's AWS SDK.
    """
    global _polly, _lambda, _agentcore, _identity, _twins
    session = boto3.Session(
        aws_access_key_id=os.environ["RORY_AGENTCORE_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )
    _polly = session.client("polly")
    _lambda = session.client("lambda")
    _agentcore = session.client("bedrock-agentcore")
    _identity = AWSCredentialsIdentity(
        access_key_id=os.environ["RORY_AGENTCORE_AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY"],
    )
    _twins = {
        ATTR_STRIPE_API_BASE: os.environ["STRIPE_API_BASE"],
        ATTR_STRIPE_API_KEY: os.environ["STRIPE_API_KEY"],
        ATTR_FINERACT_API_BASE: os.environ["FINERACT_API_BASE"],
    }
    if "RORY_REFERENCE_TIME" in os.environ:
        _twins[ATTR_REFERENCE_TIME] = os.environ["RORY_REFERENCE_TIME"]
    logger.info(
        f"[startup] lex={os.environ['LEX_BOT_ID']}/{os.environ['LEX_BOT_ALIAS_ID']} "
        f"polly={POLLY_VOICE_ID}/{POLLY_ENGINE} tool_preamble={TOOL_PREAMBLE}"
    )


async def _open_lex(session_id: str):
    """One Lex conversation stream for the whole call."""
    config = await AsyncLexRuntimeV2Config.resolve(
        region=os.environ["AWS_REGION"],
        transport=AWSCRTHTTPClient(),
        aws_credentials_identity_resolver=StaticCredentialsResolver(_identity),
    )
    stream = await AsyncLexRuntimeV2Client(config).start_conversation(StartConversationInput(
        bot_id=os.environ["LEX_BOT_ID"],
        bot_alias_id=os.environ["LEX_BOT_ALIAS_ID"],
        locale_id=LEX_LOCALE_ID,
        session_id=session_id,
        conversation_mode="AUDIO",
    ))
    await stream.input_stream.send(StartConversationRequestEventStreamConfigurationEvent(ConfigurationEvent(
        # Text back, not audio: the pod speaks the agent's reply itself. Playback
        # stays on, as Connect runs it: after each response Lex waits for the
        # client to say the prompt has played before it takes the next turn. With
        # playback disabled Lex loops on its own prompt and fails the stream.
        response_content_type="text/plain; charset=utf-8",
        session_state=SessionState(session_attributes=LEX_SESSION_ATTRIBUTES),
    )))
    return stream


def _init_agent(session_id: str) -> None:
    """The init Lambda: the call's session record, and a warm agent session."""
    resp = _lambda.invoke(
        FunctionName=os.environ["INIT_LAMBDA"],
        Payload=json.dumps({"session_id": session_id, ATTR_TOOL_PREAMBLE: TOOL_PREAMBLE, **_twins}).encode(),
    )
    body = resp["Payload"].read()
    if "FunctionError" in resp:
        raise RuntimeError(f"init Lambda failed: {body.decode()}")


def _invoke_turn(session_id: str, payload: dict[str, Any], emit) -> None:
    """One caller turn through the runtime, ``emit``-ing its events as they arrive."""
    resp = _agentcore.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENTCORE_RUNTIME_ARN"],
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="text/event-stream",
        payload=json.dumps(payload).encode(),
    )
    # read1 hands over what has arrived; a sized read would sit on the first
    # sentence until the buffer filled.
    raw = resp["response"]._raw_stream
    buf = b""
    while chunk := raw.read1(65536):
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            emit(json.loads(event.removeprefix(b"data: ")))


def _to_rate(pcm16: bytes, src_hz: int, dst_hz: int) -> bytes:
    converted, _ = audioop.ratecv(pcm16, 2, 1, src_hz, dst_hz, None)
    return converted


def _polly_audio(text: str) -> bytes:
    audio = _polly.synthesize_speech(
        Text=text, VoiceId=POLLY_VOICE_ID, Engine=POLLY_ENGINE, OutputFormat="pcm", SampleRate=str(POLLY_RATE_HZ),
    )["AudioStream"].read()
    return _to_rate(audio, POLLY_RATE_HZ, ACTOR_RATE_HZ)


def _polly_marks(text: str) -> list[dict]:
    body = _polly.synthesize_speech(
        Text=text, VoiceId=POLLY_VOICE_ID, Engine=POLLY_ENGINE, OutputFormat="json", SpeechMarkTypes=["word"],
    )["AudioStream"].read()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


@dataclass
class _Piece:
    """One sentence of a reply: where it sits in its model round's text, and its audio."""

    round: int
    start: int  # character offset in the round's text
    text: str
    synth: asyncio.Task  # (word marks, PCM16 at the actor's rate)
    sent_bytes: int = 0
    marks: list[dict] = field(default_factory=list)

    def heard(self) -> str:
        """The part of this sentence the caller heard before the cut.

        A word counts as heard once it has started playing. Polly's ``end`` is a
        byte offset into the UTF-8 text.
        """
        heard_ms = self.sent_bytes // BYTES_PER_MS
        heard = [mark for mark in self.marks if mark["time"] < heard_ms]
        return self.text.encode()[: heard[-1]["end"]].decode() if heard else ""


async def _synthesize(text: str) -> tuple[list[dict], bytes]:
    """Polly's audio and its word timings. One output format per request, so two, side by side."""
    marks, audio = await asyncio.gather(asyncio.to_thread(_polly_marks, text), asyncio.to_thread(_polly_audio, text))
    return marks, audio


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection, one call."""
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")

    t_start = time.monotonic()
    try:
        await _Call(actor_ws).run()
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


class _Call:
    """One call: actor audio to Lex, Lex's transcripts to the agent, its reply back through Polly.

    Three tasks run against this state: the actor pump, which forwards the caller's
    audio to Lex and watches it for barge-in; the Lex reader, which queues each
    transcript; and the turn worker, which takes the turns one at a time and speaks
    the replies.

    The conversation lives in the AgentCore Runtime, keyed by this call's session.
    This process keeps a copy in step with it, only to report the agent's trace,
    and a barge-in it has not yet reported: the reply as the caller heard it, sent
    with the next turn.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self.actor_ws = actor_ws
        # One Lex session and one AgentCore session per call, as one Connect contact.
        self.session_id = str(uuid.uuid4())
        self._heard: str | None = None  # last reply as heard, after a barge-in
        # The runtime's history, seeded the same way, for the trace.
        self._history: list[dict] = [
            {"role": "user", "content": [{"text": CALL_CONNECTED}]},
            {"role": "assistant", "content": [{"text": GREETING}]},
        ]

        self._vad = SileroVad(ACTOR_RATE_HZ)
        self._voiced = False
        self._silence_s = 0.0
        self._resample_state = None
        self._lex: Any = None
        self._lex_opened_at = 0.0
        self._lex_ready = asyncio.Event()  # clear while the stream is being replaced
        self._transcripts: asyncio.Queue[str] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None  # the reply in progress, once its audio has started
        self._play_end = 0.0  # when the audio handed to the actor so far finishes playing

    async def run(self) -> None:
        self._lex = await _open_lex(self.session_id)
        self._lex_opened_at = time.monotonic()
        self._lex_ready.set()
        tasks = [
            asyncio.create_task(self._pump_actor(), name="actor->lex"),
            asyncio.create_task(self._read_lex(), name="lex->turns"),
            asyncio.create_task(self._turn_worker(), name="turns"),
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
            await self._lex.close()

    async def _pump_actor(self) -> None:
        """Actor frames → Lex, and → VAD for barge-in. The actor streams silence too, which keeps the call alive."""
        n_frames = 0
        try:
            while True:
                frame = await self.actor_ws.receive_bytes()
                n_frames += 1
                if not self._voiced and time.monotonic() - self._lex_opened_at > LEX_STREAM_MAX_S:
                    await self._renew_lex()
                audio, self._resample_state = audioop.ratecv(
                    frame, 2, 1, ACTOR_RATE_HZ, LEX_RATE_HZ, self._resample_state
                )
                await self._lex.input_stream.send(StartConversationRequestEventStreamAudioInputEvent(
                    AudioInputEvent(content_type=LEX_AUDIO, audio_chunk=audio)
                ))
                if self._vad.push(frame):
                    if not self._voiced:
                        self._voiced = True
                        self._on_voice_onset()
                    self._silence_s = 0.0
                elif self._voiced:
                    self._silence_s += len(frame) / BYTES_PER_S
                    if self._silence_s >= END_OF_SPEECH_S:
                        self._voiced = False
        except WebSocketDisconnect as exc:
            logger.info(
                f"[voice] actor disconnected after {n_frames} frames: code={getattr(exc, 'code', '?')}"
            )

    async def _renew_lex(self) -> None:
        """A new Lex stream on the same session. The old one closes first: Lex refuses
        two streams on one session."""
        logger.info("[lex] renewing the conversation stream before Lex's 15-minute limit")
        self._lex_ready.clear()
        await self._lex.close()
        self._lex = await _open_lex(self.session_id)
        self._lex_opened_at = time.monotonic()
        self._lex_ready.set()

    def _on_voice_onset(self) -> None:
        logger.info("[vad] caller started speaking")
        if self._speaking is not None and not self._speaking.done():
            logger.info("[vad] barge-in — cutting the reply short")
            self._speaking.cancel()

    async def _read_lex(self) -> None:
        """Lex's transcripts, one per caller turn, onto the turn queue."""
        while True:
            await self._lex_ready.wait()
            stream = self._lex
            _, events = await stream.await_output()
            try:
                async for event in events:
                    if isinstance(event, StartConversationResponseEventStreamTranscriptEvent):
                        text = event.value.transcript or ""
                        logger.info(f"[lex] caller_said: {text!r}")
                        # Noise Lex took for speech arrives with nothing transcribed. There
                        # is no turn to answer, and Bedrock refuses a blank message.
                        if text.strip():
                            await self._transcripts.put(text)
                    elif isinstance(event, StartConversationResponseEventStreamTextResponseEvent):
                        # The Lex hook's placeholder prompt (see lex_hook). Nothing to
                        # play: tell Lex it has played, so it listens for the next turn.
                        await stream.input_stream.send(
                            StartConversationRequestEventStreamPlaybackCompletionEvent(PlaybackCompletionEvent())
                        )
                    elif type(event).__name__.endswith("Exception"):
                        raise RuntimeError(f"Lex stream failed: {event.value}")
            except Exception:
                if not self._replaced(stream):
                    raise
            if not self._replaced(stream):
                raise RuntimeError("Lex closed the conversation stream")

    def _replaced(self, stream: Any) -> bool:
        """Whether ``stream`` ended because the pod is replacing it."""
        return stream is not self._lex or not self._lex_ready.is_set()

    async def _turn_worker(self) -> None:
        """Greet while the agent initializes, then answer caller turns strictly one at a time."""
        # Rory speaks first, straight through Polly, so the opening words are
        # verbatim. The runtime seeds its history with the same greeting. It is not
        # interruptible: a barge-in during the greeting would leave silence.
        logger.info("[turn] speaking greeting (agent greets first)")
        init = asyncio.create_task(asyncio.to_thread(_init_agent, self.session_id))
        greeting = _Piece(0, 0, GREETING, asyncio.create_task(_synthesize(GREETING)))
        await self._play_piece(greeting)
        await init

        while True:
            await self._take_turn(await self._transcripts.get())

    async def _take_turn(self, text: str) -> None:
        """One caller turn: stream the agent's reply, speaking each sentence as it completes."""
        payload: dict[str, Any] = {"text": text}
        if self._heard is not None:
            payload[ATTR_INTERRUPTED_REPLY] = self._heard
            self._history[-1] = {"role": "assistant", "content": [{"text": self._heard}]}
            self._heard = None

        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict | None] = asyncio.Queue()

        def emit(event: dict | None) -> None:
            loop.call_soon_threadsafe(events.put_nowait, event)

        async def invoke() -> None:
            try:
                await asyncio.to_thread(_invoke_turn, self.session_id, payload, emit)
            finally:
                emit(None)

        t0 = time.monotonic()
        invoking = asyncio.create_task(invoke())
        pieces: asyncio.Queue[_Piece | None] = asyncio.Queue()
        player = asyncio.create_task(self._play(pieces, t0))
        played: list[_Piece] = []
        rounds: list[str] = []  # each model round's text so far
        spoken = 0  # characters of the current round already cut into pieces
        turn = None

        def cut(final: bool) -> None:
            """Queue the current round's finished sentences (all of it, if ``final``) to be spoken."""
            nonlocal spoken
            text = rounds[-1]
            ends = [m.end() for m in SENTENCE_END.finditer(text, spoken)]
            if not played and not ends:
                # The caller is waiting on the first words: a clause will do.
                clause = CLAUSE_END.search(text, spoken + FIRST_CLAUSE_MIN_CHARS)
                if clause:
                    ends = [clause.end()]
            if final and spoken < len(text):
                ends.append(len(text))
            for stop in ends:
                sentence = text[spoken:stop]
                # After a barge-in the rest of the reply is not spoken.
                if sentence.strip() and not player.done():
                    piece = _Piece(len(rounds) - 1, spoken, sentence, asyncio.create_task(_synthesize(sentence)))
                    played.append(piece)
                    pieces.put_nowait(piece)
                spoken = stop

        while (event := await events.get()) is not None:
            if event["type"] == "round":
                if rounds:
                    cut(final=True)
                rounds.append("")
                spoken = 0
            elif event["type"] == "text":
                if not rounds[-1]:
                    logger.info(f"[llm] first words after {time.monotonic() - t0:.2f}s")
                rounds[-1] += event["text"]
                cut(final=False)
            elif event["type"] == "tool":
                logger.info(f"[llm] tool {event['name']} after {time.monotonic() - t0:.2f}s")
                # The words before a tool call are complete: say them while it runs.
                cut(final=True)
            elif event["type"] == "turn":
                turn = event["turn"]
        await invoking  # raises if the runtime failed
        if turn is None:
            raise RuntimeError("the runtime's stream ended without the turn")
        cut(final=True)
        pieces.put_nowait(None)
        logger.info(f"[llm] rory_said: {rounds[-1]}")
        trace.report_turn(self._history, turn)

        try:
            await player
        except asyncio.CancelledError:
            # Barge-in cancels the player and the worker carries on to the caller's
            # next turn. A cancel aimed at the worker itself (the call ending
            # mid-reply) must propagate, or the worker never exits.
            if asyncio.current_task().cancelling():
                raise
            self._heard = self._heard_reply(played, rounds) + INTERRUPTED_MARKER
            logger.info(f'[voice] barge-in: caller heard "{self._heard[:80]}"')
        finally:
            self._speaking = None

    @staticmethod
    def _heard_reply(played: list[_Piece], rounds: list[str]) -> str:
        """The turn's reply (its last round's text) as far as the caller heard it."""
        last = len(rounds) - 1
        cut_at = next((p for p in reversed(played) if p.sent_bytes), None)
        if cut_at is None or cut_at.round != last:
            return ""
        return rounds[last][: cut_at.start] + cut_at.heard()

    async def _play(self, pieces: asyncio.Queue[_Piece | None], t0: float) -> None:
        """Speak pieces in order, as each one's audio is ready."""
        first = True
        while (piece := await pieces.get()) is not None:
            if first:
                await piece.synth
                logger.info(f"[tts] first audio after {time.monotonic() - t0:.2f}s")
                # From the first audio on, the caller's voice cuts this reply.
                self._speaking = asyncio.current_task()
                first = False
            await self._play_piece(piece)

    async def _play_piece(self, piece: _Piece) -> None:
        """One piece's audio to the actor, in slices paced against playback."""
        piece.marks, pcm16 = await piece.synth
        for i in range(0, len(pcm16), PLAYBACK_CHUNK_BYTES):
            ahead = self._play_end - time.monotonic()
            if ahead > PLAYBACK_LEAD_S:
                await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
            chunk = pcm16[i : i + PLAYBACK_CHUNK_BYTES]
            await self.actor_ws.send_bytes(chunk)
            piece.sent_bytes += len(chunk)
            self._play_end = max(self._play_end, time.monotonic()) + len(chunk) / BYTES_PER_S
        logger.info(f"[tts] spoke {len(pcm16) / BYTES_PER_S:.1f}s of audio")
