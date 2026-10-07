"""Rory on xAI's voice stack — Acme Energy's voice support agent, cascaded.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
xAI's streaming STT and TTS are separate services, so this process is the
pipeline: Grok STT transcribes the caller over a WebSocket and decides when
their turn is over (Smart Turn), a chat completion reasons over the transcript
and calls Rory's 16 tools, and Grok TTS speaks the reply back over a second
WebSocket. Conversation state lives here — no vendor holds the session.

The chat model is ``gpt-4.1-mini`` by default, the model the other cascades
run, so the vendor legs are what differs; ``GROK_CASCADE_LLM=grok-4.3`` swaps
in Grok with reasoning off, making the same image a second candidate.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. xAI
documents 16 kHz as its STT model's native rate, so caller audio is downsampled
on the way in. TTS is asked for 24 kHz PCM, so replies need no conversion.
"""

from __future__ import annotations

import asyncio
import audioop
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

# model → (base URL, credential env, extra request fields). grok-4.3 is xAI's
# last model with a non-reasoning mode; `none` is that mode.
LLMS = {
    "gpt-4.1-mini": ("https://api.openai.com/v1", "OPENAI_API_KEY", {}),
    "grok-4.3": ("https://api.x.ai/v1", "XAI_API_KEY", {"reasoning_effort": "none"}),
}
LLM_MODEL = os.environ.get("GROK_CASCADE_LLM", "gpt-4.1-mini")
LLM_BASE_URL, LLM_KEY_ENV, LLM_EXTRA = LLMS[LLM_MODEL]

STT_MODEL = os.environ.get("GROK_STT_MODEL", "grok-voice-transcribe-2.0")
# Same voice as the grok-voice speech-to-speech candidate, so the two xAI
# candidates sound alike.
TTS_VOICE = os.environ.get("GROK_VOICE", "eve")

ACTOR_RATE_HZ = 24000
STT_RATE_HZ = 16000

# Smart Turn: at each silence boundary xAI scores whether the caller has
# finished, and only a score above SMART_TURN ends the turn (`speech_final`);
# below it the utterance stays open. 0.5 is xAI's "balanced" setting. Without a
# timeout the model holds the turn open indefinitely, which background speech —
# the bench mixes a television under some callers — could stretch forever, so
# a turn is forced closed after SMART_TURN_TIMEOUT_MS of silence regardless
# (xAI's documented example value).
SMART_TURN = 0.5
SMART_TURN_TIMEOUT_MS = 3000

STT_URL = "wss://api.x.ai/v1/stt?" + urlencode({
    "model": STT_MODEL,
    "encoding": "pcm",
    "sample_rate": STT_RATE_HZ,
    "language": "en",
    # Spoken numbers come back as digits — account numbers, amounts, dates.
    "format": "true",
    "smart_turn": SMART_TURN,
    "smart_turn_timeout": SMART_TURN_TIMEOUT_MS,
})

TTS_URL = "wss://api.x.ai/v1/tts?" + urlencode({
    "voice": TTS_VOICE,
    "language": "en",
    "codec": "pcm",
    "sample_rate": ACTOR_RATE_HZ,
})
PCM16_BYTES_PER_MS = ACTOR_RATE_HZ * 2 // 1000

# How long a word of the reply takes the voice to say, measured on `eve`
# (318–345 ms/word across replies). A barge-in cuts the reply at the audio
# sent, and this turns that into words. xAI's `with_timestamps` would give
# per-character alignment instead, but it arrives a sentence late — each
# sentence's characters come after its audio — so at the cut the sentence the
# caller interrupted has none yet.
TTS_MS_PER_WORD = 330

# xAI streams a reply about five times faster than it plays, so sent as it
# lands the whole reply sits in the actor's buffer within two seconds and a
# barge-in has nothing left to cut. Replies go out in slices this long, paced
# against playback, as in the Hugging Face cascade.
PLAYBACK_CHUNK_BYTES = 500 * PCM16_BYTES_PER_MS

# How far ahead of real-time playback the pacer is allowed to run. One slice
# in the actor's buffer keeps the line gapless; more just widens the slice of
# already-delivered audio barge-in can't claw back.
PLAYBACK_LEAD_S = 1.0

# Appended to a reply the caller talked over, in place of the words that were
# never sent, so the model reads its own history as cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# Whether the caller is talking over Rory comes from Silero VAD
# (rory_tools.vad), not from STT: barge-in has to cut the reply within a frame
# or two, and the bench mixes cafe and television audio under the caller,
# which a frame-energy gate hears as a caller who never stops talking.
# End of turn is xAI's (Smart Turn above); Silero only drives barge-in.

# A turn that still wants tools after this many round trips is looping.
MAX_TOOL_ROUNDS = 5

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()

_llm: AsyncOpenAI


async def resolve_vendor() -> None:
    """Build the shared chat client once per process, after the credential check."""
    global _llm
    _llm = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=os.environ[LLM_KEY_ENV])
    logger.info(
        f"[startup] stt={STT_MODEL} smart_turn={SMART_TURN}/{SMART_TURN_TIMEOUT_MS}ms "
        f"llm={LLM_MODEL} ({LLM_BASE_URL}) tts_voice={TTS_VOICE}"
    )


def _xai_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"}


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one xAI STT stream and one TTS stream with Rory's tools."""
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
    (which also runs the barge-in VAD), the STT event loop, and a turn worker.
    Turns are processed one at a time off a queue, so the message list is only
    ever touched by the worker even when the caller talks over a reply, and the
    TTS socket is only ever read by it.

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
        self._tts = None
        self._voiced = False               # Silero's last verdict
        self._vad = SileroVad(ACTOR_RATE_HZ)
        self._turns: asyncio.Queue[str] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None
        self._sent_bytes = 0               # PCM16 of the in-flight reply handed to the actor

    async def run(self) -> None:
        async with (
            websockets.connect(STT_URL, additional_headers=_xai_headers()) as stt,
            websockets.connect(TTS_URL, additional_headers=_xai_headers()) as tts,
        ):
            self._stt, self._tts = stt, tts
            # xAI: wait for this before sending audio.
            created = json.loads(await stt.recv())
            if created["type"] != "transcript.created":
                raise RuntimeError(f"xAI STT opened with {created}")
            logger.info(
                f"[voice] xAI STT and TTS connected stt_id={created['id']} "
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

    async def _pump_actor_to_stt(self) -> None:
        """Binary PCM16 frames from the actor (24 kHz) → xAI STT (16 kHz).

        Frames go up as they arrive, at the actor's real-time pace. The actor
        streams continuously, silence included, so the STT session never sits
        without audio.
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

                voiced = self._vad.push(frame)
                if voiced and not self._voiced:
                    self._on_voice()
                self._voiced = voiced

                pcm16k, resample_state = audioop.ratecv(
                    frame, 2, 1, ACTOR_RATE_HZ, STT_RATE_HZ, resample_state
                )
                await self._stt.send(pcm16k)
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
        """The caller started speaking: cut off any reply in progress."""
        logger.info("[vad] caller started speaking")
        if self._speaking is not None and not self._speaking.done():
            logger.info("[vad] barge-in — cutting the reply short")
            self._speaking.cancel()

    async def _pump_stt_events(self) -> None:
        """Transcription events → turn text.

        Only an utterance final (``speech_final``) is a turn, and its text is
        the whole utterance restated — chunk finals before it are logged, never
        accumulated, or the turn would carry its text twice. A chunk final with
        a low ``end_of_turn_confidence`` is a pause Smart Turn decided the
        caller was not done at.
        """
        async for raw in self._stt:
            ev = json.loads(raw)
            etype = ev["type"]

            if etype == "transcript.partial":
                confidence = ev.get("end_of_turn_confidence")
                if ev["speech_final"]:
                    text = ev["text"].strip()
                    logger.info(f"[stt] caller_said (end_of_turn={confidence}): {text}")
                    if text:
                        await self._turns.put(text)
                elif ev["is_final"] and ev["text"]:
                    logger.info(f"[stt] chunk final (end_of_turn={confidence}): {ev['text']}")

            elif etype == "error":
                raise RuntimeError(f"xAI STT error: {ev['message']}")

            else:
                logger.info(f"[stt] {etype}")

        raise RuntimeError("xAI STT closed the stream")

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
            completion = await _llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                **LLM_EXTRA,
            )
            # Keep only the portable OpenAI-schema fields: Grok returns
            # reasoning fields on the message that are not for replaying.
            msg = completion.choices[0].message.model_dump(
                include={"role", "content", "tool_calls"}, exclude_none=True
            )
            self.messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            logger.info(
                f"[llm] {LLM_MODEL} replied in {time.monotonic() - t0:.2f}s "
                f"({len(tool_calls)} tool calls)"
            )
            if not tool_calls:
                break
            # One at a time, in the order emitted: calls share verification and
            # payment state on the session.
            for call in tool_calls:
                self.messages.append(await self._run_tool(call))
        else:
            raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")

        if msg.get("content"):
            logger.info(f"[llm] rory_said: {msg['content']}")
            await self._speak(msg["content"])

    async def _run_tool(self, call: dict) -> dict:
        """Dispatch one tool call and build the tool message for the next round."""
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])
        # Every vendor client is blocking, and this coroutine shares the loop
        # carrying the caller's audio. Called inline, a ten-second vendor
        # timeout is ten seconds of dead air — the same reason the Pipecat
        # handler hops threads.
        result = await asyncio.to_thread(dispatch, self.session, name, args)
        logger.info(f"[tool] {name} -> {json.dumps(result, default=str)}")
        return {
            "role": "tool",
            "name": name,
            "tool_call_id": call["id"],
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
            await self._clear_tts()
        finally:
            self._speaking = None

    def _trim_interrupted_reply(self, text: str) -> None:
        """Rewrite the last assistant message to the part of ``text`` that went out.

        Without this the history says Rory told the caller things they never
        heard, and the next turn reasons from that. The cut is proportional
        over the reply's words: the PCM sent to the actor, in ms, at
        TTS_MS_PER_WORD. Bytes handed to the actor count as heard; the pacer
        runs at most PLAYBACK_LEAD_S ahead, so that overstates it by no more.
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

    async def _clear_tts(self) -> None:
        """Abandon the utterance on the TTS socket so the next reply starts clean.

        ``text.clear`` stops generation and discards xAI's buffer; audio already
        in flight still arrives, up to the ``audio.clear`` that acknowledges it,
        and is dropped here rather than played into the next reply. xAI
        acknowledges a clear on an idle socket too, so a barge-in that lands
        before the reply was sent needs no special case.
        """
        await self._tts.send(json.dumps({"type": "text.clear"}))
        async for raw in self._tts:
            if json.loads(raw)["type"] == "audio.clear":
                break
        else:
            raise RuntimeError("xAI TTS closed the stream")

    async def _speak_now(self, text: str) -> None:
        """Stream Grok TTS to the actor, paced against playback.

        The whole reply goes up as one utterance; audio comes back as 24 kHz
        PCM16, the actor's format. The first slice goes out as soon as it
        lands, so the actor hears the reply onset immediately; after that
        slices go out no more than PLAYBACK_LEAD_S ahead of real-time
        playback, and cancelling this task strands the rest unsent.
        """
        t0 = time.monotonic()
        await self._tts.send(json.dumps({"type": "text.delta", "delta": text}))
        await self._tts.send(json.dumps({"type": "text.done"}))

        # xAI cuts its PCM stream at arbitrary byte boundaries, so a chunk can
        # end mid-sample; the stray byte leads the next chunk.
        carry = b""
        n_chunks = 0
        n_bytes = 0
        t_play = None  # playback clock, started at the first byte sent
        async for raw in self._tts:
            ev = json.loads(raw)
            etype = ev["type"]
            if etype == "audio.delta":
                pcm = carry + base64.b64decode(ev["delta"])
                whole = len(pcm) - len(pcm) % 2
                pcm, carry = pcm[:whole], pcm[whole:]
                n_chunks += 1
                n_bytes += len(pcm)
                if n_chunks == 1:
                    logger.info(f"[tts] first chunk after {time.monotonic() - t0:.2f}s ({len(pcm)} bytes)")
                    t_play = time.monotonic()
                for i in range(0, len(pcm), PLAYBACK_CHUNK_BYTES):
                    sent_s = self._sent_bytes / 2 / ACTOR_RATE_HZ
                    ahead = sent_s - (time.monotonic() - t_play)
                    if ahead > PLAYBACK_LEAD_S:
                        await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
                    piece = pcm[i : i + PLAYBACK_CHUNK_BYTES]
                    await self.actor_ws.send_bytes(piece)
                    self._sent_bytes += len(piece)
            elif etype == "audio.done":
                break
            elif etype == "error":
                raise RuntimeError(f"xAI TTS error: {ev['message']}")
        else:
            raise RuntimeError("xAI TTS closed the stream")
        logger.info(
            f"[tts] spoke {n_chunks} chunks ({n_bytes} bytes, "
            f"{n_bytes / 2 / ACTOR_RATE_HZ:.1f}s audio) in {time.monotonic() - t0:.2f}s"
        )
