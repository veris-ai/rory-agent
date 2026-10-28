"""Rory on Amazon Connect's voice stack: the pod that stands where Connect would.

In the Connect pattern a call reaches Amazon Connect, Connect streams each caller
turn to Lex, Lex transcribes it and hands it to a fulfillment Lambda, the Lambda
invokes the agent in AgentCore Runtime, the agent reasons on Bedrock and calls
its tools over MCP, and the reply comes back through Lex for Connect to speak.

The bench does not dial a phone, so this process plays Connect's part over the
Veris ``voice_ws`` socket: it decides when the caller's turn is over, sends the
turn's audio to Lex, and speaks the reply Lex returns with Polly, the engine
behind Connect's voices. It owns turn-taking and barge-in, as the other cascades
do. Everything after Lex (the fulfillment Lambda, the runtime, the gateway, the
tools Lambda) runs in AWS; see ``init_lambda``, ``runtime`` and ``tools_lambda``.

Each Lex request carries session attributes: the twins the bench injected into
this pod, which the tools Lambda must use, the bench's frozen clock, and, after a
barge-in, the part of the last reply the caller actually heard.

Sample-rate note: the actor speaks and listens at 24 kHz PCM16. Lex accepts 16 kHz
linear PCM and Polly's PCM output is 16 kHz, so audio is resampled at both
boundaries.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import gzip
import json
import os
import time
import uuid
from typing import Any

import boto3
from loguru import logger
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
    CALL_CONNECTED,
    INTERRUPTED_MARKER,
)

LEX_LOCALE_ID = os.environ.get("LEX_LOCALE_ID", "en_US")
POLLY_VOICE_ID = os.environ.get("POLLY_VOICE_ID", "Joanna")
POLLY_ENGINE = os.environ.get("POLLY_ENGINE", "neural")

ACTOR_RATE_HZ = 24000
LEX_RATE_HZ = 16000
POLLY_RATE_HZ = 16000
BYTES_PER_S = ACTOR_RATE_HZ * 2
BYTES_PER_MS = BYTES_PER_S // 1000

# End of the caller's turn: this much silence, matching the 0.8 s the other
# cascades land at. Measured in audio time, not the wall clock, so a burst of
# frames arriving late cannot endpoint a turn early.
END_OF_TURN_S = 0.8

# Audio kept from just before voice onset, so Lex does not lose the first word.
PRE_ROLL_BYTES = 300 * BYTES_PER_MS

# Replies go out in slices paced against playback, so a barge-in has something
# to cut. Sent as it lands, the whole reply would sit in the actor's buffer.
PLAYBACK_CHUNK_BYTES = 500 * BYTES_PER_MS
PLAYBACK_LEAD_S = 1.0

# Set once per process by resolve_vendor, after the credential check.
_lex: Any
_polly: Any
_sessions: Any
_bot_id: str
_bot_alias_id: str
_twins: dict[str, str]


def resolve_vendor() -> None:
    """Build the AWS clients and read this attempt's twins, once per process.

    The AWS key has its own names rather than the SDK's defaults: on the bench it
    sits in candidate-env, which every candidate on the bench receives, and the
    default names would hand it to any other candidate's AWS SDK.
    """
    global _lex, _polly, _sessions, _bot_id, _bot_alias_id, _twins
    session = boto3.Session(
        aws_access_key_id=os.environ["RORY_AGENTCORE_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION"],
    )
    _lex = session.client("lexv2-runtime")
    _polly = session.client("polly")
    _sessions = session.resource("dynamodb").Table(os.environ["SESSIONS_TABLE"])
    _bot_id = os.environ["LEX_BOT_ID"]
    _bot_alias_id = os.environ["LEX_BOT_ALIAS_ID"]
    _twins = {
        ATTR_STRIPE_API_BASE: os.environ["STRIPE_API_BASE"],
        ATTR_STRIPE_API_KEY: os.environ["STRIPE_API_KEY"],
        ATTR_FINERACT_API_BASE: os.environ["FINERACT_API_BASE"],
    }
    if "RORY_REFERENCE_TIME" in os.environ:
        _twins[ATTR_REFERENCE_TIME] = os.environ["RORY_REFERENCE_TIME"]
    logger.info(f"[startup] lex={_bot_id}/{_bot_alias_id} polly={POLLY_VOICE_ID}/{POLLY_ENGINE}")


def _to_rate(pcm16: bytes, src_hz: int, dst_hz: int) -> bytes:
    converted, _ = audioop.ratecv(pcm16, 2, 1, src_hz, dst_hz, None)
    return converted


def _encode(value: Any) -> str:
    """Lex's compressed fields: JSON, gzipped, then base64."""
    return base64.b64encode(gzip.compress(json.dumps(value).encode())).decode()


def _decode(value: str) -> Any:
    return gzip.decompress(base64.b64decode(value)).decode()


def _recognize(session_id: str, pcm16_16k: bytes, attrs: dict[str, str]) -> tuple[str, str]:
    """One caller turn through Lex and the agent behind it: (what Lex heard, what Rory says)."""
    resp = _lex.recognize_utterance(
        botId=_bot_id,
        botAliasId=_bot_alias_id,
        localeId=LEX_LOCALE_ID,
        sessionId=session_id,
        sessionState=_encode({"sessionAttributes": attrs}),
        requestContentType=f"audio/l16; rate={LEX_RATE_HZ}; channels=1",
        # Text back: the pod speaks the reply itself, as Connect would.
        responseContentType="text/plain; charset=utf-8",
        inputStream=pcm16_16k,
    )
    resp["audioStream"].close()
    transcript = json.loads(_decode(resp["inputTranscript"]))
    messages = json.loads(_decode(resp["messages"])) if "messages" in resp else []
    return transcript, " ".join(m["content"] for m in messages).strip()


def _read_turn(session_id: str) -> tuple[int, dict[str, Any]]:
    """The last turn the agent took on this call, as the fulfillment Lambda recorded it."""
    item = _sessions.get_item(Key={"session_id": session_id}, ConsistentRead=True)["Item"]
    return int(item["turn_seq"]), json.loads(item["last_turn"])


def _synthesize(text: str) -> tuple[list[dict], bytes]:
    """Polly audio at the actor's rate, and the word timings that map cut audio back to text.

    Polly returns one output format per call, so the audio and its word speech
    marks are two requests.
    """
    audio = _polly.synthesize_speech(
        Text=text,
        VoiceId=POLLY_VOICE_ID,
        Engine=POLLY_ENGINE,
        OutputFormat="pcm",
        SampleRate=str(POLLY_RATE_HZ),
    )["AudioStream"].read()
    marks_body = _polly.synthesize_speech(
        Text=text,
        VoiceId=POLLY_VOICE_ID,
        Engine=POLLY_ENGINE,
        OutputFormat="json",
        SpeechMarkTypes=["word"],
    )["AudioStream"].read()
    marks = [json.loads(line) for line in marks_body.splitlines() if line.strip()]
    return marks, _to_rate(audio, POLLY_RATE_HZ, ACTOR_RATE_HZ)


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
    """One call: actor audio in, Lex turns out, Polly replies back.

    Two tasks run against this state: the actor pump, which does VAD and cuts
    the caller's audio into turns, and the turn worker, which sends them to Lex
    one at a time and speaks the replies.

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
        self._turn_seq = 0

        self._vad = SileroVad(ACTOR_RATE_HZ)
        self._voiced = False
        self._silence_s = 0.0
        self._utterance = bytearray()
        self._pre_roll = b""
        self._turns: asyncio.Queue[bytes] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None
        self._marks: list[dict] = []  # word timings of the reply in progress
        self._sent_bytes = 0  # PCM16 of the in-flight reply handed to the actor

    async def run(self) -> None:
        tasks = [
            asyncio.create_task(self._pump_actor(), name="actor->turns"),
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

    async def _pump_actor(self) -> None:
        """Actor frames → VAD → caller turns. The actor streams silence too, which keeps the call alive."""
        n_frames = 0
        try:
            while True:
                frame = await self.actor_ws.receive_bytes()
                n_frames += 1
                if self._vad.push(frame):
                    if not self._voiced:
                        self._open_turn()
                    self._silence_s = 0.0
                    self._utterance += frame
                elif self._voiced:
                    self._utterance += frame
                    self._silence_s += len(frame) / BYTES_PER_S
                    if self._silence_s >= END_OF_TURN_S:
                        await self._close_turn()
                else:
                    self._pre_roll = (self._pre_roll + frame)[-PRE_ROLL_BYTES:]
        except WebSocketDisconnect as exc:
            logger.info(
                f"[voice] actor disconnected after {n_frames} frames: code={getattr(exc, 'code', '?')}"
            )

    def _open_turn(self) -> None:
        """Voice onset: start the turn, and cut off any reply in progress."""
        self._voiced = True
        self._utterance = bytearray(self._pre_roll)
        logger.info("[vad] caller started speaking")
        if self._speaking is not None and not self._speaking.done():
            logger.info("[vad] barge-in — cutting the reply short")
            self._speaking.cancel()

    async def _close_turn(self) -> None:
        self._voiced = False
        self._silence_s = 0.0
        audio = bytes(self._utterance)
        self._utterance = bytearray()
        self._pre_roll = b""
        logger.info(f"[turn] endpoint — {len(audio) / BYTES_PER_S:.1f}s of caller audio")
        await self._turns.put(audio)

    async def _turn_worker(self) -> None:
        """Greet, then answer caller turns strictly one at a time."""
        # Rory speaks first, straight through Polly, so the opening words are
        # verbatim. The runtime seeds its history with the same greeting. It is not
        # interruptible: a barge-in during the greeting would leave silence.
        logger.info("[turn] speaking greeting (agent greets first)")
        await self._say(GREETING)

        while True:
            audio = await self._turns.get()
            attrs = dict(_twins)
            if self._heard is not None:
                attrs[ATTR_INTERRUPTED_REPLY] = self._heard
            t0 = time.monotonic()
            transcript, reply = await asyncio.to_thread(
                _recognize, self.session_id, _to_rate(audio, ACTOR_RATE_HZ, LEX_RATE_HZ), attrs
            )
            logger.info(f"[lex] {time.monotonic() - t0:.2f}s caller_said: {transcript!r}")
            # No reply means nothing was transcribed and the agent never ran, so a
            # pending barge-in is still unreported.
            if not reply:
                continue
            await self._report_turn()
            logger.info(f"[lex] rory_said: {reply}")
            await self._speak(reply)

    async def _report_turn(self) -> None:
        """Export the agent's model rounds for the turn it just took."""
        seq, turn = await asyncio.to_thread(_read_turn, self.session_id)
        if seq != self._turn_seq + 1:
            raise RuntimeError(f"expected agent turn {self._turn_seq + 1}, the record holds {seq}")
        self._turn_seq = seq
        if self._heard is not None:
            self._history[-1] = {"role": "assistant", "content": [{"text": self._heard}]}
            self._heard = None
        trace.report_turn(self._history, turn)

    async def _speak(self, text: str) -> None:
        """Speak a reply, cancellable by barge-in."""
        self._speaking = asyncio.create_task(self._say(text))
        try:
            await self._speaking
        except asyncio.CancelledError:
            # Barge-in cancels the speech task and the worker carries on to the
            # caller's next turn. A cancel aimed at the worker itself (the call
            # ending mid-reply) must propagate, or the worker never exits.
            if asyncio.current_task().cancelling():
                raise
            self._heard = self._heard_part(text) + INTERRUPTED_MARKER
            logger.info(f'[voice] barge-in: caller heard "{self._heard[:80]}"')
        finally:
            self._speaking = None

    def _heard_part(self, text: str) -> str:
        """The part of ``text`` the caller heard before the cut.

        A word counts as heard once it has started playing. Polly's ``end`` is a
        character offset into the text, so the kept text is exact.
        """
        heard_ms = self._sent_bytes // BYTES_PER_MS
        heard = [mark for mark in self._marks if mark["time"] < heard_ms]
        return text[: heard[-1]["end"]] if heard else ""

    async def _say(self, text: str) -> None:
        """One Polly reply, sent to the actor in slices paced against playback."""
        self._sent_bytes = 0
        self._marks, pcm16 = await asyncio.to_thread(_synthesize, text)
        t_play = time.monotonic()
        for i in range(0, len(pcm16), PLAYBACK_CHUNK_BYTES):
            ahead = self._sent_bytes / BYTES_PER_S - (time.monotonic() - t_play)
            if ahead > PLAYBACK_LEAD_S:
                await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
            piece = pcm16[i : i + PLAYBACK_CHUNK_BYTES]
            await self.actor_ws.send_bytes(piece)
            self._sent_bytes += len(piece)
        logger.info(f"[tts] spoke {len(pcm16) / BYTES_PER_S:.1f}s of audio")
