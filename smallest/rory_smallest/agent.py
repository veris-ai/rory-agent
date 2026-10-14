"""Rory on Smallest AI's voice stack — Acme Energy's voice support agent, cascaded.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
Smallest ships the three legs of a voice agent as separate products rather
than one session, so this process *is* the pipeline: Pulse transcribes the
caller over a WebSocket, a chat model (``gpt-4.1-mini`` by default, or
Smallest's own Electron) reasons over the transcript and calls Rory's 16
tools, and Lightning speaks the reply back over a second WebSocket, one
connection per utterance. Turn-taking, barge-in and conversation state live
here — no vendor holds the session.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. Pulse's
live endpoint and Lightning both accept 24 kHz directly, so — unlike the
Mistral and Hugging Face cascades — no audio is resampled anywhere in this
file.
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
from openai import AsyncOpenAI
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context
from rory_tools.vad import SileroVad

from .tools import TOOLS

# model → (base URL, credential env). The chat model is ``gpt-4.1-mini`` by
# default, the model the other cascades run, so the vendor legs are what
# differs; ``SMALLEST_LLM_MODEL=electron`` swaps in Smallest's own model,
# making the same image a second candidate.
LLMS = {
    "gpt-4.1-mini": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "electron": ("https://api.smallest.ai/waves/v1", "SMALLEST_API_KEY"),
}
LLM_MODEL = os.environ.get("SMALLEST_LLM_MODEL", "gpt-4.1-mini")
LLM_BASE_URL, LLM_KEY_ENV = LLMS[LLM_MODEL]
# The Pro pool: broadcast-quality voices, dedicated inference capacity, no
# cloning. Pro is picky about voice/model pairing — confirmed live, a
# standard-pool voice like "sophia" on this model is rejected outright by the
# API ("Voice 'sophia' is not available on the lightning_v3.1_pro model")
# rather than silently producing wrong audio, so VOICE must name a voice
# actually in the Pro pool.
TTS_MODEL = os.environ.get("SMALLEST_TTS_MODEL", "lightning_v3.1_pro")
VOICE = os.environ.get("SMALLEST_VOICE", "kelsey")
LANGUAGE = os.environ.get("SMALLEST_LANGUAGE", "en")

# The Veris actor speaks and listens at 24 kHz PCM16. Pulse's live endpoint
# accepts 8000/16000/22050/24000/44100/48000 Hz and Lightning can be asked for
# any of those too, so the actor's own rate is used end to end.
ACTOR_RATE_HZ = 24000
PCM16_BYTES_PER_MS = ACTOR_RATE_HZ * 2 // 1000

STT_URL = "wss://api.smallest.ai/waves/v1/pulse/get_text?" + urlencode(
    {
        "language": LANGUAGE,
        "encoding": "linear16",
        "sample_rate": ACTOR_RATE_HZ,
        "word_timestamps": "false",
        "full_transcript": "false",
        "sentence_timestamps": "false",
        "redact_pii": "false",
        "redact_pci": "false",
        "numerals": "auto",
        "diarize": "false",
    }
)

# The unified streaming route — not the older per-model path
# (`/waves/v1/{model}/get_speech/stream`), which Smallest's own docs flag as
# deprecated. The model is a body field here rather than part of the URL, so
# switching SMALLEST_TTS_MODEL needs no URL change.
TTS_URL = "wss://api.smallest.ai/waves/v1/tts/live"

# End of the caller's turn: this much silence, matching the 0.8 s the Pipecat
# and Gemini candidates land at. Measured in *audio* time — silent bytes seen —
# not the wall clock, for the same reason the Mistral cascade does: frames do
# not arrive at a steady rate, and a wall-clock timer would endpoint mid-burst.
END_OF_TURN_S = 0.8

# Whether the caller is speaking comes from Silero VAD (rory_tools.vad), not
# frame energy: the bench mixes cafe and television audio under the caller,
# and an energy gate hears that as a caller who never stops talking.

# How long a word of the reply takes Lightning to say, measured on `kelsey`
# (338 ms/word over four replies, 288–381 each). Needed because a barge-in
# cancels the TTS connection mid-reply, and Lightning's `chunk` frames carry no
# text alignment of their own, so the cut turns audio sent into words.
TTS_MS_PER_WORD = 340

# Lightning streams a reply faster than it plays, so sent as it lands the
# whole reply sits in the actor's buffer within a second or two and a
# barge-in has nothing left to cut. Replies go out in slices this long, paced
# against playback, as in the xAI and Hugging Face cascades.
PLAYBACK_CHUNK_BYTES = 500 * PCM16_BYTES_PER_MS

# How far ahead of real-time playback the pacer is allowed to run. One slice
# in the actor's buffer keeps the line gapless; more just widens the slice of
# already-delivered audio barge-in can't claw back.
PLAYBACK_LEAD_S = 1.0

# Appended to a reply the caller talked over, in place of the words that were
# never sent, so the model reads its own history as cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# A turn that still wants tools after this many round trips is looping.
MAX_TOOL_ROUNDS = 5

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()

_llm: AsyncOpenAI


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['SMALLEST_API_KEY']}"}


async def resolve_vendor() -> None:
    """Build the shared chat client once per process, after the credential check."""
    global _llm
    _llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=os.environ[LLM_KEY_ENV])
    logger.info(f"[startup] stt=pulse llm={LLM_MODEL} tts={TTS_MODEL} voice={VOICE}")


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Pulse session with Rory's tools."""
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")

    t_start = time.monotonic()
    try:
        await _Call(actor_ws).run()
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


class _Call:
    """The pipeline for one call: STT stream in, LLM turn, TTS connection out.

    Three tasks run concurrently against this state — the actor→STT audio pump
    (which also does VAD and endpointing), the STT event loop, and a turn
    worker. Turns are processed one at a time off a queue, so the message list
    is only ever touched by the worker even when the caller talks over a reply.

    A fresh ``CallSession`` per call, exactly as the Pipecat agent builds one
    per ``PipelineTask``: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self.actor_ws = actor_ws
        self.session = CallSession()
        # The date context is a second system message rather than part of
        # ``agent_desc.txt`` because it is resolved per call, off the frozen
        # clock the attempt restored with its snapshot.
        self.messages: list = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "system", "content": today_context()},
        ]

        self._stt = None
        self._voiced = False               # caller is mid-utterance
        self._silence_s = 0.0              # audio seconds since the last voiced frame
        self._vad = SileroVad(ACTOR_RATE_HZ)
        self._segments: list[str] = []     # `is_final` texts of the turn in progress
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None
        self._sent_bytes = 0  # PCM16 of the in-flight reply handed to the actor

    async def run(self) -> None:
        self._stt = await websockets.connect(STT_URL, additional_headers=_auth_headers())
        logger.info(f"[voice] Pulse connected rate={ACTOR_RATE_HZ}Hz")

        tasks = [
            asyncio.create_task(self._pump_actor_to_stt(), name="actor->stt"),
            asyncio.create_task(self._pump_stt_events(), name="stt->turns"),
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
            await self._stt.close()

    async def _pump_actor_to_stt(self) -> None:
        """Binary PCM16 frames from the actor (24 kHz) → Pulse (24 kHz), unconverted.

        Doubles as the turn detector: Silero scores every frame for speech,
        which drives both barge-in and the turn endpoint.

        The actor streams continuously, silence included, which is also what
        keeps the session alive.
        """
        n_frames = 0
        n_bytes = 0
        try:
            while True:
                frame = await self.actor_ws.receive_bytes()
                n_frames += 1
                n_bytes += len(frame)
                if n_frames == 1:
                    logger.info(f"[a->stt] first frame bytes={len(frame)}")
                elif n_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(f"[a->stt] forwarded {n_frames} frames ({n_bytes} bytes @24kHz)")

                if self._vad.push(frame):
                    self._on_voice()
                elif self._voiced:
                    self._silence_s += len(frame) / 2 / ACTOR_RATE_HZ
                    if self._silence_s >= END_OF_TURN_S:
                        await self._endpoint()

                await self._stt.send(frame)
        except WebSocketDisconnect as exc:
            logger.info(
                f"[a->stt] actor disconnected after {n_frames} frames "
                f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
            )
        except KeyError as exc:
            # Starlette raises KeyError('bytes') when it got a text frame instead of
            # a binary one — a protocol mismatch on the actor side, not a bug here.
            logger.error(
                f"[a->stt] non-binary frame after {n_frames} binary frames — "
                f"actor protocol mismatch? ({exc})"
            )
            raise

    def _on_voice(self) -> None:
        """A speech frame arrived: open the turn, and cut off any reply in progress."""
        self._silence_s = 0.0
        if self._voiced:
            return
        self._voiced = True
        logger.info("[vad] caller started speaking")
        if self._speaking is not None and not self._speaking.done():
            logger.info("[vad] barge-in — cutting the reply short")
            self._speaking.cancel()

    async def _endpoint(self) -> None:
        """The caller has stopped: ask Pulse to finalize the turn.

        Nothing waits here for the answer — the pump goes straight back to
        forwarding audio, and ``_pump_stt_events`` dispatches the turn when
        Pulse's ``from_finalize`` final arrives.
        """
        logger.info("[turn] endpoint — finalizing")
        self._voiced = False
        self._silence_s = 0.0
        await self._stt.send(json.dumps({"type": "finalize"}))

    async def _pump_stt_events(self) -> None:
        """Transcription events → turn text.

        Verified live (2026-10-14, Lightning's own TTS looped into Pulse): an
        ``is_final`` transcript covers only the audio since the previous final.
        Pulse finalizes on its own at short pauses, so a caller who reads an
        account number, pauses half a second and gives a postcode arrives as
        two finals inside one turn — the turn is their concatenation. The final
        answering our ``finalize`` carries ``from_finalize: true`` (with
        whatever text was still pending, often none) and is what closes the
        turn. Interim (``is_final: false``) events are not acted on; they
        exist for live captions, which this transport does not render.
        """
        async for raw in self._stt:
            data = json.loads(raw)
            if not data.get("is_final"):
                continue
            text = data["transcript"].strip()
            if text:
                self._segments.append(text)
            if data.get("from_finalize"):
                turn = " ".join(self._segments)
                self._segments.clear()
                logger.info(f"[stt] caller_said: {turn}")
                if turn:
                    await self._turns.put(turn)

        raise RuntimeError("Pulse STT closed the stream")

    async def _turn_worker(self) -> None:
        """Greet, then process caller turns strictly one at a time.

        The greeting runs here rather than before the pumps start so that actor
        audio is being consumed from the first frame — speaking it inline would
        let three seconds of caller audio pile up in the socket.
        """
        # Rory speaks first. Spoken straight through TTS rather than asked of
        # the LLM, as Pipecat does, so the opening words are verbatim and the
        # first turn costs no completion. Recorded as the first assistant turn
        # so the model knows it has already greeted and doesn't repeat it.
        #
        # The greeting is not interruptible, as in the xAI cascade. The bench's
        # background noise starts with the call, and on the first smoke run
        # Silero fired 0.15–0.3 s in and cancelled the greeting before any
        # audio went out: the caller heard silence and hung up after 30 s. It
        # is spoken by ``_speak_now`` directly, so ``_speaking`` stays unset and
        # ``_on_voice`` has nothing to cancel. A caller who does talk over it
        # has their turn taken right after.
        logger.info("[turn] speaking greeting (agent greets first)")
        await self._speak_now(GREETING)
        self.messages.append({"role": "assistant", "content": GREETING})

        while True:
            text = await self._turns.get()
            await self._take_turn(text)

    async def _take_turn(self, text: str) -> None:
        """One caller turn: LLM (with tools) → spoken reply."""
        self.messages.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            resp = await _llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            assistant_message: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in msg.tool_calls
                ]
            self.messages.append(assistant_message)
            logger.info(
                f"[llm] {LLM_MODEL} replied in {time.monotonic() - t0:.2f}s "
                f"({len(msg.tool_calls or [])} tool calls)"
            )
            if not msg.tool_calls:
                break
            # One at a time, in the order emitted: calls share verification and
            # payment state on the session.
            for call in msg.tool_calls:
                self.messages.append(await self._run_tool(call))
        else:
            raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")

        if msg.content:
            logger.info(f"[llm] rory_said: {msg.content}")
            await self._speak(str(msg.content))

    async def _run_tool(self, call) -> dict:
        """Dispatch one tool call and build the tool message for the next round."""
        name = call.function.name
        args = json.loads(call.function.arguments) if call.function.arguments else {}
        # Every vendor client is blocking, and this coroutine shares the loop
        # carrying the caller's audio. Called inline, a ten-second vendor
        # timeout is ten seconds of dead air — the same reason the Pipecat
        # handler hops threads.
        result = await asyncio.to_thread(dispatch, self.session, name, args)
        logger.info(f"[tool] {name} -> {json.dumps(result, default=str)}")
        return {
            "role": "tool",
            "name": name,
            "tool_call_id": call.id,
            "content": json.dumps(result, default=str),
        }

    async def _speak(self, text: str) -> None:
        """Speak the last assistant message, cancellable by barge-in.

        ``text`` is that message's content, already on ``messages`` so a
        barge-in can cut it down to what the caller heard.
        """
        self._sent_bytes = 0
        self._speaking = asyncio.create_task(self._speak_now(text))
        try:
            await self._speaking
        except asyncio.CancelledError:
            # Barge-in cancels only the TTS task, and the worker carries on to
            # the caller's next turn. A cancel aimed at the worker itself (the
            # call ending mid-reply) lands here too, and swallowing that would
            # leave the worker parked on the queue forever.
            if asyncio.current_task().cancelling():
                raise
            self._trim_interrupted_reply(text)
        finally:
            self._speaking = None

    def _trim_interrupted_reply(self, text: str) -> None:
        """Rewrite the last assistant message to the part of ``text`` that went out.

        Without this the history says Rory told the caller things they never
        heard, and the next turn reasons from that. Lightning's `chunk`
        messages carry no sentence boundaries, so the cut is proportional over
        the reply's words: the PCM sent to the actor, in ms, against the
        reply's length at TTS_MS_PER_WORD. Bytes handed to the actor count as
        heard; the pacer runs at most PLAYBACK_LEAD_S ahead, so that
        overstates it by no more.
        """
        if self._sent_bytes == 0:
            self.messages.pop()
            logger.info("[voice] barge-in: dropped reply, nothing sent before the cut")
            return
        heard_ms = self._sent_bytes // PCM16_BYTES_PER_MS
        words = text.split()
        total_ms = len(words) * TTS_MS_PER_WORD
        kept = " ".join(words[: heard_ms // TTS_MS_PER_WORD])
        self.messages[-1] = {"role": "assistant", "content": f"{kept}{INTERRUPTED_MARKER}"}
        logger.info(
            f'[voice] barge-in: trimmed reply to {heard_ms} ms / {total_ms} ms ("{kept[:60]}")'
        )

    async def _speak_now(self, text: str) -> None:
        """One Lightning utterance: a fresh connection, closed when it completes.

        Unlike Pulse, Lightning's WebSocket takes one utterance per connection
        rather than a session held for the whole call — there is no documented
        way to abandon an utterance mid-stream and reuse the socket cleanly, so
        a barge-in cancelling this coroutine simply closes the connection
        (the ``async with`` below) rather than leaving stale audio for the next
        reply to collide with.

        The first slice goes out as soon as it lands, so the actor hears the
        reply onset immediately; after that slices go out no more than
        PLAYBACK_LEAD_S ahead of real-time playback, and cancelling this task
        strands the rest unsent.
        """
        t0 = time.monotonic()
        # close_timeout=0: Lightning does not answer a close frame while it is
        # still streaming, so a barge-in's close waited out the default 10 s
        # handshake timeout — ten seconds of dead air before the caller's turn
        # (8 of 27 barge-ins on the second VAmoS smoke run). The utterance is
        # being abandoned; there is nothing to wait for.
        async with websockets.connect(
            TTS_URL, additional_headers=_auth_headers(), close_timeout=0
        ) as ws:
            await ws.send(json.dumps({
                "text": text,
                "voice_id": VOICE,
                "model": TTS_MODEL,
                "language": LANGUAGE,
                "sample_rate": ACTOR_RATE_HZ,
            }))
            n_chunks = 0
            n_bytes = 0
            t_play = None  # playback clock, started at the first byte sent
            async for raw in ws:
                msg = json.loads(raw)
                status = msg.get("status")
                if status == "chunk":
                    pcm16 = base64.b64decode(msg["data"]["audio"])
                    n_chunks += 1
                    n_bytes += len(pcm16)
                    if n_chunks == 1:
                        logger.info(
                            f"[tts] first chunk after {time.monotonic() - t0:.2f}s ({len(pcm16)} bytes)"
                        )
                        t_play = time.monotonic()
                    for i in range(0, len(pcm16), PLAYBACK_CHUNK_BYTES):
                        sent_s = self._sent_bytes / 2 / ACTOR_RATE_HZ
                        ahead = sent_s - (time.monotonic() - t_play)
                        if ahead > PLAYBACK_LEAD_S:
                            await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
                        piece = pcm16[i : i + PLAYBACK_CHUNK_BYTES]
                        await self.actor_ws.send_bytes(piece)
                        self._sent_bytes += len(piece)
                elif status == "complete":
                    break
                elif status == "error":
                    raise RuntimeError(f"Lightning TTS error: {msg.get('error', msg)}")
                else:
                    logger.warning(f"[tts] unknown message status: {msg}")
            else:
                raise RuntimeError("Lightning TTS closed the stream before `complete`")
            logger.info(
                f"[tts] spoke {n_chunks} chunks ({n_bytes} bytes, "
                f"{n_bytes / 2 / ACTOR_RATE_HZ:.1f}s audio) in {time.monotonic() - t0:.2f}s"
            )
