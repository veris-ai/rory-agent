"""Rory on Amazon Connect's voice stack — Lex, AgentCore Runtime, Polly, cascaded.

Amazon Connect is a telephony front end that hands each turn to three services:
Lex turns the caller's speech into text, a Bedrock AgentCore Runtime runs the
model and its tools, and Polly speaks the reply. The bench does not dial a
phone, so this process stands where Connect would and runs the same three legs
over the Veris ``voice_ws`` socket. It owns turn-taking and barge-in, as the
other cascades do.

Everything below the transport is shared with the other agents: same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
transport is the only thing allowed to differ.

The AgentCore Runtime is the one leg not in this repository. It is a deployed
agent that takes one model round per request:

    request   {"messages": [...], "tools": [{name, description, parameters}]}
    response  {"content": str | null, "tool_calls": [{id, name, arguments}]}

It calls Bedrock with those tools. When the model asks for tools, they come
back in ``tool_calls`` and this process runs them through the shared dispatcher
locally, because verification and payment state live here, not in the runtime.

Sample-rate note: the actor speaks and listens at 24 kHz PCM16. Lex accepts
16 kHz linear PCM and Polly's PCM output is 16 kHz, so audio is resampled at
both boundaries.
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

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context
from rory_tools.vad import SileroVad

from .tools import TOOLS

AGENT_PROMPT = load_agent_prompt()

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

# Appended to a reply the caller talked over, so the model reads its own history
# as cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# A turn that still wants tools after this many model rounds is looping.
MAX_TOOL_ROUNDS = 5

# Set once per process by resolve_vendor, after the credential check.
_lex: Any
_polly: Any
_agentcore: Any
_bot_id: str
_bot_alias_id: str
_runtime_arn: str


def resolve_vendor() -> None:
    """Build the AWS clients once per process."""
    global _lex, _polly, _agentcore, _bot_id, _bot_alias_id, _runtime_arn
    region = os.environ["AWS_REGION"]
    _lex = boto3.client("lexv2-runtime", region_name=region)
    _polly = boto3.client("polly", region_name=region)
    _agentcore = boto3.client("bedrock-agentcore", region_name=region)
    _bot_id = os.environ["LEX_BOT_ID"]
    _bot_alias_id = os.environ["LEX_BOT_ALIAS_ID"]
    _runtime_arn = os.environ["AGENTCORE_RUNTIME_ARN"]
    logger.info(
        f"[startup] stt=lex polly={POLLY_VOICE_ID}/{POLLY_ENGINE} "
        f"llm=agentcore region={region}"
    )


def _to_rate(pcm16: bytes, src_hz: int, dst_hz: int) -> bytes:
    converted, _ = audioop.ratecv(pcm16, 2, 1, src_hz, dst_hz, None)
    return converted


def _recognize(session_id: str, pcm16_16k: bytes) -> str:
    """One caller turn through Lex, as a single buffered utterance."""
    resp = _lex.recognize_utterance(
        botId=_bot_id,
        botAliasId=_bot_alias_id,
        localeId=LEX_LOCALE_ID,
        sessionId=session_id,
        requestContentType=f"audio/l16; rate={LEX_RATE_HZ}; channels=1",
        # Text back, so Lex does not synthesise a reply we would discard.
        responseContentType="text/plain; charset=utf-8",
        inputStream=pcm16_16k,
    )
    resp["audioStream"].close()
    # Lex sends the transcript gzipped and base64-encoded in a header.
    return gzip.decompress(base64.b64decode(resp["inputTranscript"])).decode()


def _agent_turn(session_id: str, messages: list[dict]) -> dict:
    """One model round in AgentCore Runtime."""
    resp = _agentcore.invoke_agent_runtime(
        agentRuntimeArn=_runtime_arn,
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps({"messages": messages, "tools": TOOLS}).encode(),
    )
    return json.loads(resp["response"].read())


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
    """The pipeline for one call: actor audio in, Lex turns, AgentCore rounds, Polly out.

    Two tasks run against this state: the actor pump, which does VAD and cuts
    the caller's audio into turns, and the turn worker, which transcribes and
    answers them one at a time. The worker is the only writer of ``messages``,
    so a caller who talks over a reply cannot race the history.

    A fresh ``CallSession`` per call, so the account ``verify_caller`` matched
    dies with the call and identity never becomes a model argument.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self.actor_ws = actor_ws
        self.session = CallSession()
        # Lex and AgentCore sessions are one per call, as the Connect contact is.
        self.session_id = str(uuid.uuid4())
        # The date context is a second system message rather than part of the
        # shared prompt because it is resolved per call, off the frozen clock.
        self.messages: list[dict] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "system", "content": today_context()},
        ]

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
        # Rory speaks first, straight through Polly rather than through the model,
        # so the opening words are verbatim. It is not interruptible: a barge-in
        # during the greeting would leave the caller with silence.
        logger.info("[turn] speaking greeting (agent greets first)")
        await self._say(GREETING)
        self.messages.append({"role": "assistant", "content": GREETING, "tool_calls": []})

        while True:
            audio = await self._turns.get()
            text = await asyncio.to_thread(
                _recognize, self.session_id, _to_rate(audio, ACTOR_RATE_HZ, LEX_RATE_HZ)
            )
            logger.info(f"[stt] caller_said: {text}")
            if text.strip():
                await self._take_turn(text)

    async def _take_turn(self, text: str) -> None:
        """One caller turn: AgentCore rounds with tools, then the spoken reply."""
        self.messages.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            reply = await asyncio.to_thread(_agent_turn, self.session_id, self.messages)
            self.messages.append(
                {"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}
            )
            logger.info(
                f"[llm] agentcore replied in {time.monotonic() - t0:.2f}s "
                f"({len(reply['tool_calls'])} tool calls)"
            )
            if not reply["tool_calls"]:
                break
            # One at a time, in the order emitted: calls share verification and
            # payment state on the session.
            for call in reply["tool_calls"]:
                self.messages.append(await self._run_tool(call))
        else:
            raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")

        if reply["content"]:
            logger.info(f"[llm] rory_said: {reply['content']}")
            await self._speak(reply["content"])

    async def _run_tool(self, call: dict) -> dict:
        # The vendor clients are blocking. Called inline, a slow lookup would
        # stall the loop carrying the caller's audio.
        result = await asyncio.to_thread(dispatch, self.session, call["name"], call["arguments"])
        logger.info(f"[tool] {call['name']} -> {json.dumps(result, default=str)}")
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["name"],
            "content": json.dumps(result, default=str),
        }

    async def _speak(self, text: str) -> None:
        """Speak the last assistant message, cancellable by barge-in."""
        self._speaking = asyncio.create_task(self._say(text))
        try:
            await self._speaking
        except asyncio.CancelledError:
            # Barge-in cancels the speech task and the worker carries on to the
            # caller's next turn. A cancel aimed at the worker itself (the call
            # ending mid-reply) must propagate, or the worker never exits.
            if asyncio.current_task().cancelling():
                raise
            self._trim_interrupted_reply(text)
        finally:
            self._speaking = None

    def _trim_interrupted_reply(self, text: str) -> None:
        """Rewrite the last assistant message to the part of ``text`` that the caller heard."""
        if self._sent_bytes == 0:
            self.messages.pop()
            logger.info("[voice] barge-in: dropped reply, nothing sent before the cut")
            return
        heard_ms = self._sent_bytes // BYTES_PER_MS
        # A word counts as heard once it has started playing before the cut. Polly's
        # `end` is a character offset into the text, so the kept text is exact.
        heard = [mark for mark in self._marks if mark["time"] < heard_ms]
        kept = text[: heard[-1]["end"]] if heard else ""
        self.messages[-1] = {
            "role": "assistant",
            "content": f"{kept}{INTERRUPTED_MARKER}",
            "tool_calls": [],
        }
        logger.info(f'[voice] barge-in: trimmed reply to {heard_ms} ms ("{kept[:60]}")')

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
