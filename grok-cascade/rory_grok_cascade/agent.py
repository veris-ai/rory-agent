"""Rory on xAI's voice stack — Acme Energy's voice support agent, cascaded.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
xAI's streaming STT and TTS are separate services, so this process is the
pipeline: Grok STT transcribes the caller over a WebSocket and decides when
their turn is over (Smart Turn), a streamed chat completion reasons over the
transcript and calls Rory's 16 tools, and its tokens go straight into Grok TTS
over a second WebSocket, which speaks the reply back as it is written. Conversation state lives here — no vendor holds the session.

The chat model is ``gpt-4.1-mini`` by default, the model the other cascades
run, so the vendor legs are what differs; ``GROK_CASCADE_LLM=grok-4.3`` swaps
in Grok with reasoning off, making the same image a second candidate.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and both
xAI legs are told so — STT takes the caller's audio as it arrives and TTS
returns 24 kHz PCM — so no audio is converted here.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator
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
# xAI's recommendation for this agent.
TTS_VOICE = os.environ.get("GROK_VOICE", "carina")

ACTOR_RATE_HZ = 24000

# Smart Turn: at each silence boundary xAI scores whether the caller has
# finished, and only a score above SMART_TURN ends the turn (`speech_final`);
# below it the utterance stays open. 0.7 is xAI's setting for callers reading
# out numbers, which Rory's callers do to be verified. Without a
# timeout the model holds the turn open indefinitely, which background speech —
# the bench mixes a television under some callers — could stretch forever, so
# a turn is forced closed after SMART_TURN_TIMEOUT_MS of silence regardless
# (xAI's documented example value).
SMART_TURN = 0.7
SMART_TURN_TIMEOUT_MS = 3000

STT_URL = "wss://api.x.ai/v1/stt?" + urlencode({
    "model": STT_MODEL,
    "encoding": "pcm",
    "sample_rate": ACTOR_RATE_HZ,
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

# How long a word of the reply takes the voice to say, measured on `carina`
# (306 ms/word over four replies, 238–341 each). A barge-in cuts the reply at the audio
# sent, and this turns that into words. xAI's `with_timestamps` would give
# per-character alignment instead, but it arrives a sentence late — each
# sentence's characters come after its audio — so at the cut the sentence the
# caller interrupted has none yet.
TTS_MS_PER_WORD = 305

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
        self._replying = False             # a caller turn is being answered
        self._cut = False                  # ...and the caller talked over it
        self._player: asyncio.Task | None = None  # TTS audio → actor for the reply in flight
        self._said = ""                    # text of the in-flight reply sent to TTS
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
                f"rate={ACTOR_RATE_HZ}Hz"
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
        """Binary PCM16 frames from the actor → xAI STT, unconverted.

        Frames go up as they arrive, at the actor's real-time pace. The actor
        streams continuously, silence included, so the STT session never sits
        without audio.
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

                voiced = self._vad.push(frame)
                if voiced and not self._voiced:
                    self._on_voice()
                self._voiced = voiced

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
        """The caller started speaking: cut off the reply in progress.

        Only the audio stops. The LLM round behind it runs to the end, so a
        tool call the model has decided on still runs and is recorded — the
        turn just says nothing more.
        """
        logger.info("[vad] caller started speaking")
        if not self._replying or self._cut:
            return
        logger.info("[vad] barge-in — cutting the reply short")
        self._cut = True
        if self._player is not None:
            self._player.cancel()

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
        # first turn costs no completion. Recorded as the first assistant turn
        # so the model knows it has already greeted and doesn't repeat it.
        #
        # The greeting is not interruptible. The bench's background noise
        # starts with the call, and a television under the caller cut the
        # greeting at 0.4 s, before any audio went out: the caller heard
        # silence and hung up after 30 s. A caller who does talk over it has
        # their turn taken right after.
        logger.info("[turn] speaking greeting (agent greets first)")
        await self._say(_once(GREETING))
        self.messages.append({"role": "assistant", "content": GREETING})

        while True:
            text = await self._turns.get()
            self._replying, self._cut = True, False
            try:
                await self._take_turn(text)
            finally:
                self._replying = False

    async def _take_turn(self, text: str) -> None:
        """One caller turn: LLM rounds (with tools), each spoken as it streams.

        A round's text goes to TTS token by token as the model writes it, so a
        round that says "let me check that" before calling a tool is heard
        before the tool runs. A barge-in silences the rest of the turn, but
        the round it lands in still completes and its tool calls still run;
        then the turn ends, and the caller's next utterance takes it from
        there with the results already on the history.
        """
        self.messages.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            tool_calls: dict[int, dict] = {}
            stream = await _llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
                **LLM_EXTRA,
            )
            async with stream:
                said = await self._say(_content(stream, tool_calls, t0))
            msg: dict = {"role": "assistant"}
            content = self._heard(said) if self._cut else said
            if content:
                msg["content"] = content
                logger.info(f"[llm] rory_said: {content}")
            if tool_calls:
                msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
            if len(msg) > 1:
                self.messages.append(msg)
            logger.info(
                f"[llm] {LLM_MODEL} round done in {time.monotonic() - t0:.2f}s "
                f"({len(tool_calls)} tool calls)"
            )
            # One at a time, in the order emitted: calls share verification and
            # payment state on the session.
            for call in msg.get("tool_calls", []):
                self.messages.append(await self._run_tool(call))
            if self._cut or not tool_calls:
                return
        raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")

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

    def _heard(self, said: str) -> str | None:
        """The part of ``said`` whose audio went out before the barge-in, marked.

        Without this the history says Rory told the caller things they never
        heard, and the next turn reasons from that. The cut is proportional
        over the words sent to TTS: the PCM sent to the actor, in ms, at
        TTS_MS_PER_WORD. Bytes handed to the actor count as heard; the pacer
        runs at most PLAYBACK_LEAD_S ahead, so that overstates it by no more.
        """
        if self._sent_bytes == 0:
            logger.info("[voice] barge-in: dropped reply, nothing sent before the cut")
            return None
        heard_ms = self._sent_bytes // PCM16_BYTES_PER_MS
        words = said.split()
        kept = " ".join(words[: heard_ms // TTS_MS_PER_WORD])
        logger.info(
            f"[voice] barge-in: trimmed reply to {heard_ms} ms / "
            f'{len(words) * TTS_MS_PER_WORD} ms ("{kept[:60]}")'
        )
        return f"{kept}{INTERRUPTED_MARKER}"

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

    async def _say(self, deltas: AsyncIterator[str]) -> str:
        """One TTS utterance: text deltas up as they arrive, audio back to the actor.

        Returns the text sent to TTS. Text goes to xAI unchunked — xAI buffers
        it and decides when to synthesize — and the audio is played by a
        second task while text is still going up, so speech starts on the
        reply's first words. A round with no text (only tool calls) opens no
        utterance.

        ``deltas`` is always read to the end, so a barge-in never cuts a
        completion short of its tool calls: once the caller is talking, the
        remaining text is read and not sent, and the utterance is cleared.
        """
        said = ""
        self._sent_bytes = 0
        t0 = time.monotonic()
        try:
            async for delta in deltas:
                if self._cut:
                    continue
                if self._player is None:
                    self._player = asyncio.create_task(self._play(t0), name="tts->actor")
                said += delta
                await self._tts.send(json.dumps({"type": "text.delta", "delta": delta}))
            if self._player is None:
                return said
            if not self._cut:
                await self._tts.send(json.dumps({"type": "text.done"}))
            try:
                await self._player
            except asyncio.CancelledError:
                # The barge-in cancelled the player. A cancel aimed at this
                # task (the call ending mid-reply) lands here too, and must
                # propagate.
                if asyncio.current_task().cancelling():
                    raise
            if self._cut:
                await self._clear_tts()
            return said
        finally:
            if self._player is not None and not self._player.done():
                self._player.cancel()
                await asyncio.gather(self._player, return_exceptions=True)
            self._player = None

    async def _play(self, t0: float) -> None:
        """Grok TTS audio → the actor, paced against playback.

        Audio comes back as 24 kHz PCM16, the actor's format. The first slice
        goes out as soon as it lands, so the actor hears the reply onset
        immediately; after that slices go out no more than PLAYBACK_LEAD_S
        ahead of real-time playback, and cancelling this task strands the rest
        unsent.
        """
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
                    logger.info(f"[tts] first audio {time.monotonic() - t0:.2f}s after the first text")
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
            f"[tts] spoke {n_chunks} chunks ({n_bytes / 2 / ACTOR_RATE_HZ:.1f}s audio) "
            f"in {time.monotonic() - t0:.2f}s"
        )


async def _once(text: str) -> AsyncIterator[str]:
    yield text


async def _content(stream, tool_calls: dict[int, dict], t0: float) -> AsyncIterator[str]:
    """The text of a streamed completion, collecting its tool calls on the side.

    Tool calls stream as fragments keyed by index — the id and name arrive
    once, the arguments in pieces — and are assembled into ``tool_calls``.
    """
    first = True
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if first and (delta.content or delta.tool_calls):
            logger.info(f"[llm] first token after {time.monotonic() - t0:.2f}s")
            first = False
        for part in delta.tool_calls or []:
            call = tool_calls.setdefault(part.index, {
                "id": None, "type": "function", "function": {"name": "", "arguments": ""},
            })
            if part.id:
                call["id"] = part.id
            if part.function.name:
                call["function"]["name"] += part.function.name
            if part.function.arguments:
                call["function"]["arguments"] += part.function.arguments
        if delta.content:
            yield delta.content
