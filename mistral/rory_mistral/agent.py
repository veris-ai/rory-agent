"""Rory on Mistral's voice stack — Acme Energy's voice support agent, cascaded.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
Mistral ships the three legs of a voice agent as separate services rather than
one session, so this process *is* the pipeline: Voxtral Realtime transcribes the
caller over a WebSocket, a Mistral chat completion reasons over the transcript
and calls Rory's 16 tools, and Voxtral TTS speaks the reply back. Turn-taking,
barge-in and conversation state live here — no vendor holds the session.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. Voxtral
Realtime wants 16 kHz, so caller audio is downsampled on the way in. Voxtral
TTS emits 24 kHz float32 already, so replies only need float32 → int16.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import time

import numpy as np
from loguru import logger
from mistralai.client import Mistral
from mistralai.client.models import AudioFormat
from mistralai.client.utils import BackoffStrategy, RetryConfig
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context
from rory_tools.vad import SileroVad

from .tools import TOOLS

STT_MODEL = os.environ.get("MISTRAL_STT_MODEL", "voxtral-mini-transcribe-realtime-2602")
TTS_MODEL = os.environ.get("MISTRAL_TTS_MODEL", "voxtral-mini-tts-2603")
LLM_MODEL = os.environ.get("MISTRAL_LLM_MODEL", "mistral-large-latest")

# Voxtral's US-English presets are the `en_paul_*` family; the other presets
# are en_gb or fr_fr. Resolved to a UUID at startup — `voice_id` takes the id,
# not the slug.
VOICE_SLUG = os.environ.get("MISTRAL_VOICE", "en_paul_neutral")

# The Veris actor speaks and listens at 24 kHz. Voxtral Realtime wants 16 kHz
# input; Voxtral TTS emits 24 kHz, so only the input leg needs resampling.
ACTOR_RATE_HZ = 24000
STT_RATE_HZ = 16000
PCM16_BYTES_PER_MS = ACTOR_RATE_HZ * 2 // 1000

# How long a word of the reply takes Voxtral to say — a nominal 150 words per
# minute. Needed because a barge-in cancels the TTS stream mid-way, and the
# stream carries no text alignment and no duration until `speech.audio.done`,
# so the reply's full length is never known at the cut. Compare `[llm]
# rory_said` word counts with `[tts] spoke … s audio` in the container log to
# check it against the deployed voice.
TTS_MS_PER_WORD = 400

# Appended to a reply the caller talked over, in place of the words that were
# never sent, so the model reads its own history as cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# End of the caller's turn: this much silence, matching the 0.8 s the Pipecat
# and Gemini candidates land at.
#
# Endpointing runs on the caller's audio, not on transcription timing. Voxtral
# emits deltas in bursts — it holds words back until it has enough right
# context, and mid-sentence gaps of two seconds happen — so delta silence says
# nothing about whether the caller stopped talking. Measuring the audio keeps
# the endpoint honest and leaves `flush_audio()` for what it is good at:
# forcing out the tail the deltas are still sitting on.
#
# It is measured in *audio* time — silent bytes seen — not on the wall clock.
# Frames do not arrive at a steady 50 fps: anything that stalls the pump lets
# the socket backlog and then delivers a burst, and a wall-clock timer reads
# that burst as a long silence and endpoints in the middle of a sentence.
END_OF_TURN_S = 0.8

# Whether the caller is speaking comes from Silero VAD (rory_tools.vad), not
# frame energy: the bench mixes cafe and television audio under the caller,
# and an energy gate hears that as a caller who never stops talking.

# A turn that still wants tools after this many round trips is looping.
MAX_TOOL_ROUNDS = 5

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()

_client: Mistral
_voice_id: str


async def resolve_vendor() -> None:
    """Build the shared client and resolve the voice slug once per process.

    Called from the web lifespan, after the credential check, so an unknown
    voice fails the boot rather than the first call.
    """
    global _client, _voice_id
    # A turn spends 2–3 completions walking the tool loop, which is enough to
    # trip Mistral's rate limit. The SDK knows how to retry a 429 but ships that
    # switched off, so a single throttled request would otherwise take the call
    # down mid-sentence.
    #
    # The ceiling is 25 s rather than something closer to a caller's patience
    # because of how this limit behaves: throttling arrives as a burst that
    # persists for several seconds, not as a single rejected request. A ceiling
    # short enough to feel natural gives up while the window is still open and
    # kills the call — an outcome strictly worse than a slow reply, since the
    # caller then gets nothing at all.
    _client = Mistral(
        api_key=os.environ["MISTRAL_API_KEY"],
        retry_config=RetryConfig(
            "backoff",
            BackoffStrategy(
                initial_interval=200,
                max_interval=4_000,
                exponent=1.5,
                max_elapsed_time=25_000,
            ),
            retry_connection_errors=True,
        ),
    )
    voices = await _client.audio.voices.list_async(type_="preset", limit=100)
    matches = [v for v in voices.items if v.slug == VOICE_SLUG]
    if not matches:
        raise RuntimeError(
            f"voice {VOICE_SLUG!r} is not a Voxtral preset; available: "
            + ", ".join(sorted(v.slug for v in voices.items if v.slug))
        )
    _voice_id = matches[0].id
    logger.info(
        f"[startup] stt={STT_MODEL} llm={LLM_MODEL} tts={TTS_MODEL} "
        f"voice={VOICE_SLUG} ({_voice_id})"
    )


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Voxtral Realtime session with Rory's tools."""
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")

    t_start = time.monotonic()
    try:
        await _Call(actor_ws).run()
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


class _Call:
    """The pipeline for one call: STT stream in, LLM turn, TTS stream out.

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
        self._partial: list[str] = []      # deltas since the last flush, for the log
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None
        self._sent_bytes = 0  # PCM16 of the in-flight reply handed to the actor

    async def run(self) -> None:
        self._stt = await _client.audio.realtime.connect(
            model=STT_MODEL,
            audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=STT_RATE_HZ),
        )
        logger.info(
            f"[voice] Voxtral Realtime connected request_id={self._stt.request_id} "
            f"in={ACTOR_RATE_HZ}Hz stt={STT_RATE_HZ}Hz"
        )

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
        """Binary PCM16 frames from the actor (24 kHz) → Voxtral Realtime (16 kHz).

        Doubles as the turn detector: Silero scores every frame for speech,
        which drives both barge-in and the turn endpoint.

        The actor streams continuously, silence included, which is also what
        keeps the session alive: Voxtral drops the connection after ~30 s with
        no audio at all.
        """
        resample_state = None  # carried across ratecv calls so the resample is continuous
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

                pcm16k, resample_state = audioop.ratecv(
                    frame, 2, 1, ACTOR_RATE_HZ, STT_RATE_HZ, resample_state
                )
                await self._stt.send_audio(pcm16k)
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
        """The caller has stopped: flush so Voxtral finalizes the turn."""
        logger.info(
            f"[turn] endpoint after {self._silence_s:.2f}s of silence — flushing "
            f"(partial: {''.join(self._partial).strip() or '(none yet)'})"
        )
        self._voiced = False
        self._silence_s = 0.0
        await self._stt.flush_audio()

    async def _pump_stt_events(self) -> None:
        """Transcription events → turn text.

        Deltas are incremental word fragments, collected only so the container log
        shows the transcript building in real time. The turn text itself comes
        from ``transcription.done``, which the flush triggers.
        """
        async for ev in self._stt.events():
            etype = getattr(ev, "type", None)

            if etype == "transcription.text.delta":
                self._partial.append(ev.text)

            elif etype == "transcription.done":
                # Authoritative text for the turn: the flush pushes out the last
                # word or two, which the deltas hold back waiting for context.
                text = ev.text.strip()
                self._partial.clear()
                logger.info(f"[stt] caller_said: {text}")
                if text:
                    await self._turns.put(text)

            elif etype == "error":
                logger.error(f"[stt] Voxtral Realtime error: {ev.error}")
                raise RuntimeError(f"Voxtral Realtime error: {ev.error}")

            else:
                logger.info(f"[stt] {etype}")

    async def _turn_worker(self) -> None:
        """Greet, then process caller turns strictly one at a time.

        The greeting runs here rather than before the pumps start so that actor
        audio is being consumed from the first frame — speaking it inline would
        let three seconds of caller audio pile up in the socket.
        """
        # Rory speaks first. Spoken straight through TTS rather than asked of
        # the LLM, as Pipecat does, so the opening words are verbatim and the
        # first turn costs no completion. Seeded as the first assistant turn so
        # the model knows it has already greeted and doesn't repeat it, and
        # seeded before it is spoken, like every reply, so a caller talking
        # over it trims it too.
        logger.info("[turn] speaking greeting (agent greets first)")
        self.messages.append({"role": "assistant", "content": GREETING})
        await self._speak(GREETING)

        while True:
            text = await self._turns.get()
            await self._take_turn(text)

    async def _take_turn(self, text: str) -> None:
        """One caller turn: LLM (with tools) → spoken reply."""
        self.messages.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            resp = await _client.chat.complete_async(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            self.messages.append(msg)
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
        raw = call.function.arguments
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
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
        heard, and the next turn reasons from that. Voxtral gives no sentence
        boundaries, so the cut is proportional over the reply's words: the
        PCM sent to the actor, in ms, against the reply's length at
        TTS_MS_PER_WORD. Bytes handed to the actor count as heard; the stream
        is unpaced, so that is an upper bound on what was played.
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
        """Stream Voxtral TTS straight to the actor as it arrives.

        Voxtral's ``pcm`` is raw float32 LE at 24 kHz — already the actor's rate,
        so the only conversion is float32 → int16. Chunks go out as they land
        rather than being buffered, so the actor hears the reply onset
        immediately; each is ~0.4 s of audio, which is also the granularity at
        which barge-in can cut the reply.
        """
        t0 = time.monotonic()
        stream = await _client.audio.speech.complete_async(
            model=TTS_MODEL,
            input=text,
            voice_id=_voice_id,
            response_format="pcm",
            stream=True,
        )
        n_chunks = 0
        n_bytes = 0
        async for ev in stream:
            if getattr(ev.data, "type", None) != "speech.audio.delta":
                continue
            f32 = np.frombuffer(base64.b64decode(ev.data.audio_data), dtype="<f4")
            pcm16 = (np.clip(f32, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            n_chunks += 1
            n_bytes += len(pcm16)
            if n_chunks == 1:
                logger.info(f"[tts] first chunk after {time.monotonic() - t0:.2f}s ({len(pcm16)} bytes)")
            await self.actor_ws.send_bytes(pcm16)
            self._sent_bytes += len(pcm16)
        logger.info(
            f"[tts] spoke {n_chunks} chunks ({n_bytes} bytes, "
            f"{n_bytes / 2 / ACTOR_RATE_HZ:.1f}s audio) in {time.monotonic() - t0:.2f}s"
        )
