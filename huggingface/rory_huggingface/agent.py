"""Rory on Hugging Face's hosted inference — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
The pipeline is the huggingface/speech-to-speech cascade — VAD, Whisper STT, an
open LLM, Kokoro TTS — with every model leg hosted on Hugging Face and nothing
running locally. Two hosting configurations share this file: dedicated
Inference Endpoints (``HF_STT_URL``/``HF_LLM_URL``/``HF_TTS_URL`` point each leg
at your own deployments) and serverless Inference Providers, the fallback when
no URLs are set — the gpt-oss-120b and fal-ai defaults below belong to it, and
one HF token bills all three legs. HF's hosted legs are plain request/response
HTTP (no streaming STT, no streaming TTS), so the pipeline buffers each caller
utterance, transcribes it on endpoint, and paces the synthesized reply back
out. Turn-taking, barge-in, and conversation state live here — no vendor holds
the session.

The three legs go through the router with plain httpx rather than
``huggingface_hub``'s AsyncInferenceClient: the client's fal-ai text-to-speech
path downloads the result with a *blocking* requests call inside the event
loop, which would stall the 20 ms audio pumps. The router URLs it would build
are constructed the same way here (verified against huggingface_hub 1.26).

Everything below the transport is shared with the other Rory candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher. The candidates exist to be compared, so the transport is the only
thing allowed to differ.

Logging is intentionally chatty so it's obvious from the container log alone
whether audio is flowing, how the turn state is stepping, and where things
stall.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import io
import json
import os
import time
import wave
from collections import deque
from contextlib import asynccontextmanager

import httpx
from loguru import logger
from openai import AsyncOpenAI
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context
from rory_tools.vad import SileroVad

from .tools import TOOLS

ROUTER = "https://router.huggingface.co"

# The three legs of the cascade. These ids are the *serverless* defaults —
# Whisper large-v3 and Kokoro are the huggingface/speech-to-speech repo's own
# STT and TTS choices; gpt-oss-120b is an open LLM with dependable function
# calling. The endpoints configuration overrides them from the image env. On
# the router the LLM id may carry a `:provider` suffix (e.g. `:groq`) to pin
# which partner serves it — without one the router picks.
STT_MODEL = os.environ.get("HF_STT_MODEL", "openai/whisper-large-v3")
LLM_MODEL = os.environ.get("HF_LLM_MODEL", "openai/gpt-oss-120b:groq")
TTS_MODEL = os.environ.get("HF_TTS_MODEL", "hexgrad/Kokoro-82M")

# A Kokoro voice id: `af_*`/`am_*` are American female/male. Passed through to
# the provider verbatim; an unknown voice is the provider's error to raise.
TTS_VOICE = os.environ.get("HF_TTS_VOICE", "af_heart")

# Dedicated Inference Endpoints — HF's other hosting product. When set, the
# leg posts to the deployment instead of the serverless router; the same HF
# token authorizes both. HF_STT_URL is an ASR endpoint (default engine, same
# raw-bytes shape as hf-inference), used as-is. HF_LLM_URL is the endpoint's
# base URL as copied from the console; a vLLM engine serves the OpenAI API
# under /v1, and HF_LLM_MODEL must then be the bare served model id — the
# `:provider` routing suffix is a router concept.
STT_URL = os.environ.get("HF_STT_URL", f"{ROUTER}/hf-inference/models/{STT_MODEL}")
LLM_BASE_URL = (
    f"{os.environ['HF_LLM_URL'].rstrip('/')}/v1"
    if "HF_LLM_URL" in os.environ
    else f"{ROUTER}/v1"
)

# HF_TTS_URL is a custom-handler TTS endpoint speaking Riley's contract —
# POST {"inputs", "parameters": {"voice"}} → {"audio_b64": WAV} — rather than
# fal-ai's two-step URL shape. HF_TTS_VOICE must then name a voice the deployed
# model actually has (Kokoro's are `af_*`/`am_*`).
TTS_URL_OVERRIDE = os.environ.get("HF_TTS_URL")

# Veris's voice_ws actor speaks raw PCM16 mono at 24 kHz in both directions.
# Utterances are downsampled to 16 kHz before upload — Whisper resamples to
# 16 kHz anyway, so the extra bytes buy nothing. Kokoro synthesizes at 24 kHz,
# the actor's rate already; any other rate is resampled on the way back.
ACTOR_RATE_HZ = 24000
STT_RATE_HZ = 16000
PCM16_BYTES_PER_MS = ACTOR_RATE_HZ * 2 // 1000

# Appended to a reply the caller talked over, in place of the words that were
# never sent, so the model reads its own history as cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# End of the caller's turn: this much silence, matching the ~0.8 s effective
# end-of-turn silence the Pipecat and Gemini Live candidates land at. Measured
# in *audio* time — silent bytes seen — not on the wall clock: frames arrive in
# bursts when anything stalls the pump, and a wall-clock timer reads a burst as
# a long silence and endpoints mid-sentence.
END_OF_TURN_S = 0.8

# Whether the caller is speaking comes from Silero VAD (rory_tools.vad), not
# frame energy: the bench mixes cafe and television audio under the caller,
# and an energy gate hears that as a caller who never stops talking.

# Audio kept from just before the first speech frame. The detector trips a
# window or two after the true onset — a soft "h" scores low — and unlike the
# streaming-STT candidates nothing else hears that audio, so without a preroll
# the transcript loses word onsets.
PREROLL_S = 0.24

# The reply is sent in slices this long, paced against playback (see
# _speak_now) so barge-in can cut the tail of a reply that Kokoro returned as
# one complete file.
PLAYBACK_CHUNK_S = 0.5

# How far ahead of real-time playback the pacer is allowed to run. One chunk
# in the actor's buffer keeps the line gapless; more just widens the slice of
# already-delivered audio barge-in can't claw back.
PLAYBACK_LEAD_S = 1.0

# A turn that still wants tools after this many round trips is looping.
MAX_TOOL_ROUNDS = 5

# Bounded retry for the three HTTP legs. 429s arrive in bursts that outlast
# any single retry, and hf-inference answers 503 while a cold model loads —
# both are worth waiting out mid-call, since giving up kills the call.
RETRY_STATUSES = {429, 503}
RETRY_MAX_ELAPSED_S = 25.0
RETRY_INITIAL_S = 0.5

# How often to emit periodic frame-count log lines from the audio pump. At
# 50 fps (20 ms/frame) this is roughly one heartbeat per second.
LOG_EVERY_N_FRAMES = 50

# Used verbatim. The greeting is spoken straight through TTS before the first
# LLM turn, so the prompt's "you have already greeted the caller" is true as
# written by the time the model sees anything.
AGENT_PROMPT = load_agent_prompt()

_http: httpx.AsyncClient       # HF router + hub API, carries the HF token
_download: httpx.AsyncClient   # bare client for signed audio URLs — no token leaves HF
_llm: AsyncOpenAI              # the LLM leg, through the SDK so OTel instrumentation sees it
_tts_url: str


async def _resolve_tts_route(http: httpx.AsyncClient) -> str:
    """Resolve the TTS model id to its provider route on the router.

    Text-to-speech is not served on HF's own hf-inference infra, so the hub's
    provider mapping says which partner serves the model and under what id —
    for Kokoro that is fal-ai's `fal-ai/kokoro/american-english`. Resolved at
    startup so an unservable model fails the boot with the live mapping
    instead of 404ing mid-call.
    """
    resp = await http.get(
        f"https://huggingface.co/api/models/{TTS_MODEL}",
        params={"expand": "inferenceProviderMapping"},
    )
    resp.raise_for_status()
    mapping = resp.json().get("inferenceProviderMapping", {})
    live = {
        provider: entry["providerId"]
        for provider, entry in mapping.items()
        if entry["task"] == "text-to-speech" and entry["status"] == "live"
    }
    # The request/response shape below (`{"text": ...}` in, `audio.url` out)
    # is fal-ai's; other providers speak other shapes, so only fal-ai counts.
    if "fal-ai" not in live:
        raise RuntimeError(
            f"{TTS_MODEL!r} has no live fal-ai text-to-speech route; "
            f"live providers: {sorted(live) or 'none'}"
        )
    return f"{ROUTER}/fal-ai/{live['fal-ai']}"


@asynccontextmanager
async def connect_legs():
    """Build the shared HTTP clients and resolve the TTS route once per process.

    Entered from the FastAPI lifespan, after the credential check, so the
    process refuses to serve rather than 404 on its first call.
    """
    global _http, _download, _llm, _tts_url
    token = os.environ["HF_TOKEN"]
    # The read timeout is sized for provider cold starts, not for a healthy
    # request: fal spins Kokoro's worker down when idle, and the first
    # synthesis after that took 20–60 s measured (1–2 s warm). A timeout is
    # not a 429/503 — the retry loop can't save it — so the ceiling has to
    # clear the cold start or the first call of a run dies at the greeting.
    _http = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    _download = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    # The model leg goes through the openai SDK rather than a hand-built POST:
    # the router serves the OpenAI chat API either way, and OpenTelemetry
    # instrumentation of the openai SDK only sees calls made through it.
    # The SDK's own retry (429/5xx, exponential from 0.5 s) replaces the
    # _post_with_retry loop for this leg.
    _llm = AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=token,
        timeout=httpx.Timeout(120.0, connect=10.0),
        max_retries=4,
    )
    _tts_url = TTS_URL_OVERRIDE or await _resolve_tts_route(_http)
    logger.info(
        f"[startup] stt={STT_MODEL} llm={LLM_MODEL} tts={TTS_MODEL} voice={TTS_VOICE} "
        f"stt_url={STT_URL} llm_url={LLM_BASE_URL} tts_url={_tts_url}"
    )
    # The cold start above lands on whoever synthesizes first. Spend it here,
    # before /health answers, instead of on the greeting: the actor hangs up
    # after 30 s without audio, and a cold Kokoro worker measured 21–30 s on
    # the bench. Synthesizing the greeting itself makes the first real call
    # the warm case.
    t0 = time.monotonic()
    await _synthesize(GREETING)
    logger.info(f"[startup] tts warm in {time.monotonic() - t0:.2f}s")
    try:
        yield
    finally:
        await _http.aclose()
        await _download.aclose()
        await _llm.close()


async def _post_with_retry(url: str, *, label: str, **kwargs) -> httpx.Response:
    """POST, waiting out 429 bursts and hf-inference cold starts (503)."""
    delay = RETRY_INITIAL_S
    t0 = time.monotonic()
    while True:
        resp = await _http.post(url, **kwargs)
        if resp.status_code not in RETRY_STATUSES:
            if resp.is_error:
                # raise_for_status omits the body, which is where providers
                # put the actual reason.
                logger.error(f"[{label}] HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            return resp
        elapsed = time.monotonic() - t0
        if elapsed + delay > RETRY_MAX_ELAPSED_S:
            logger.error(
                f"[{label}] still HTTP {resp.status_code} after {elapsed:.1f}s — "
                f"giving up: {resp.text[:500]}"
            )
            resp.raise_for_status()
        logger.warning(
            f"[{label}] HTTP {resp.status_code}, retrying in {delay:.1f}s "
            f"({elapsed:.1f}s elapsed): {resp.text[:200]}"
        )
        await asyncio.sleep(delay)
        delay *= 2


async def _synthesize(text: str) -> bytes:
    """Text → PCM16 at the actor rate, via whichever TTS route is configured.

    A custom TTS endpoint returns the WAV in one JSON hop (base64); fal-ai's
    shape is two round trips — the POST returns JSON carrying a signed URL,
    the GET fetches the finished WAV.
    """
    if TTS_URL_OVERRIDE:
        resp = await _post_with_retry(
            _tts_url, label="tts",
            json={"inputs": text, "parameters": {"voice": TTS_VOICE}},
        )
        wav_bytes = base64.b64decode(resp.json()["audio_b64"])
    else:
        resp = await _post_with_retry(
            _tts_url, label="tts", json={"text": text, "voice": TTS_VOICE},
        )
        audio_url = resp.json()["audio"]["url"]
        wav = await _download.get(audio_url)
        wav.raise_for_status()
        wav_bytes = wav.content
    return _wav_to_actor_pcm(wav_bytes)


def _wav_bytes(pcm: bytes, rate_hz: int) -> bytes:
    """Wrap raw PCM16 mono in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate_hz)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_to_actor_pcm(wav: bytes) -> bytes:
    """Decode a WAV file to the actor's raw format: PCM16 mono at 24 kHz."""
    with wave.open(io.BytesIO(wav), "rb") as w:
        pcm = w.readframes(w.getnframes())
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if channels == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != ACTOR_RATE_HZ:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, ACTOR_RATE_HZ, None)
    return pcm


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one VAD → Whisper → LLM → Kokoro pipeline.

    The WebSocket is already accepted by the FastAPI route. A fresh
    ``CallSession`` per connection, exactly as the other candidates build one
    per call: it holds the account ``verify_caller`` matched and dies with the
    call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")

    t_start = time.monotonic()
    try:
        await _Call(actor_ws).run()
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


class _Call:
    """The pipeline for one call: utterance in, LLM turn, paced reply out.

    Two tasks run concurrently against this state — the actor-audio pump
    (which does VAD, utterance buffering, and endpointing) and a turn worker.
    Turns are processed one at a time off a queue, so the message list is only
    ever touched by the worker even when the caller talks over a reply.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self.actor_ws = actor_ws
        self.session = CallSession()
        # One system message rather than two: a chat template behind
        # HF_LLM_URL may reject a second system turn, and the router forwards
        # whatever it is given. The date context is resolved per call, off the
        # frozen clock the attempt restored with its snapshot.
        self.messages: list = [
            {"role": "system", "content": f"{AGENT_PROMPT}\n\n{today_context()}"}
        ]

        self._voiced = False               # caller is mid-utterance
        self._silence_s = 0.0              # audio seconds since the last voiced frame
        self._vad = SileroVad(ACTOR_RATE_HZ)
        self._utterance: list[bytes] = []  # voiced frames of the open utterance
        self._preroll: deque[bytes] = deque(
            maxlen=max(1, int(PREROLL_S * 1000 / 20))
        )
        self._turns: asyncio.Queue[bytes] = asyncio.Queue()
        self._speaking: asyncio.Task | None = None
        self._sent_bytes = 0    # PCM16 of the in-flight reply handed to the actor
        self._total_bytes = 0   # PCM16 of the whole in-flight reply

    async def run(self) -> None:
        pump = asyncio.create_task(self._pump_actor(), name="actor->turns")
        worker = asyncio.create_task(self._turn_worker(), name="turns")
        try:
            done, _ = await asyncio.wait({pump, worker}, return_when=asyncio.FIRST_COMPLETED)
            # Surface the first task's failure so the socket closes 1011 and
            # the container log carries the traceback, instead of a silent hang-up.
            for task in done:
                task.result()
        finally:
            for task in (pump, worker):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pump, worker, return_exceptions=True)

    async def _pump_actor(self) -> None:
        """Binary PCM16 frames from the actor → VAD → buffered utterances.

        Whisper is request/response, so unlike the streaming-STT candidates
        nothing consumes audio continuously: voiced frames collect in
        `_utterance`, and the 800 ms endpoint closes it and hands it to the
        turn worker.
        """
        n_frames = 0
        n_bytes = 0
        try:
            while True:
                frame = await self.actor_ws.receive_bytes()
                n_frames += 1
                n_bytes += len(frame)
                if n_frames == 1:
                    logger.info(f"[pump] first frame received bytes={len(frame)}")
                elif n_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(f"[pump] {n_frames} frames ({n_bytes} bytes)")

                if self._vad.push(frame):
                    self._on_voice()
                    self._utterance.append(frame)
                elif self._voiced:
                    # Trailing frames belong to the utterance — the endpoint
                    # threshold is silence *within* it, and Whisper is happy
                    # to see the pause.
                    self._utterance.append(frame)
                    self._silence_s += len(frame) / 2 / ACTOR_RATE_HZ
                    if self._silence_s >= END_OF_TURN_S:
                        self._endpoint()
                else:
                    self._preroll.append(frame)
        except WebSocketDisconnect as exc:
            logger.info(
                f"[pump] actor disconnected after {n_frames} frames "
                f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
            )
        except KeyError as exc:
            # Starlette raises KeyError('bytes') when it received a text frame
            # instead of a binary one — i.e. protocol mismatch on the actor side.
            logger.error(
                f"[pump] received non-binary frame after {n_frames} binary frames — "
                f"actor protocol mismatch? ({exc})"
            )
            raise

    def _on_voice(self) -> None:
        """A speech frame arrived: open the utterance, cut off any reply in progress."""
        self._silence_s = 0.0
        if self._voiced:
            return
        self._voiced = True
        self._utterance = list(self._preroll)
        self._preroll.clear()
        logger.info("[vad] caller started speaking")
        if self._speaking is not None and not self._speaking.done():
            logger.info("[vad] barge-in — cutting the reply short")
            self._speaking.cancel()

    def _endpoint(self) -> None:
        """The caller has stopped: close the utterance and queue it for the worker."""
        pcm = b"".join(self._utterance)
        logger.info(
            f"[turn] endpoint after {self._silence_s:.2f}s of silence "
            f"({len(pcm) / 2 / ACTOR_RATE_HZ:.1f}s audio)"
        )
        self._voiced = False
        self._silence_s = 0.0
        self._utterance = []
        self._turns.put_nowait(pcm)

    async def _turn_worker(self) -> None:
        """Greet, then process caller utterances strictly one at a time.

        The greeting runs here rather than before the pumps start so that
        actor audio is being consumed from the first frame — speaking it
        inline would let seconds of caller audio pile up in the socket.
        """
        # Call etiquette: Rory speaks first. Spoken directly rather than asked
        # of the LLM, so the opening words match the other candidates exactly
        # and the first turn costs no completion. Appended before it is spoken,
        # like every reply, so a caller talking over it trims it too.
        logger.info("[turn] speaking greeting (agent greets first)")
        self.messages.append({"role": "assistant", "content": GREETING})
        await self._speak(GREETING)

        while True:
            pcm = await self._turns.get()
            text = await self._transcribe(pcm)
            if not text:
                continue
            await self._take_turn(text)

    async def _transcribe(self, pcm: bytes) -> str:
        """One utterance → Whisper on hf-inference → turn text."""
        pcm16k, _ = audioop.ratecv(pcm, 2, 1, ACTOR_RATE_HZ, STT_RATE_HZ, None)
        wav = _wav_bytes(pcm16k, STT_RATE_HZ)
        t0 = time.monotonic()
        # Pinned to English. Left to detect the language, Whisper read accented
        # callers into Arabic and Portuguese script on the bench, and a name in
        # the wrong alphabet never verifies. Parameters travel only in the
        # JSON shape, so the audio goes up base64 rather than as raw bytes.
        try:
            resp = await _post_with_retry(
                STT_URL,
                label="stt",
                json={
                    "inputs": base64.b64encode(wav).decode(),
                    "parameters": {"generate_kwargs": {"language": "en", "task": "transcribe"}},
                },
            )
        except httpx.TimeoutException:
            # One utterance lost is a dropped turn the caller will repeat;
            # letting it escape would end the call, and hf-inference has been
            # seen to hang for minutes at a time.
            logger.warning(
                f"[stt] no transcript after {time.monotonic() - t0:.1f}s — dropping the turn"
            )
            return ""
        text = resp.json()["text"].strip()
        logger.info(
            f"[stt] {len(wav) / 2 / STT_RATE_HZ:.1f}s audio → "
            f"{time.monotonic() - t0:.2f}s latency, caller_said: {text[:200]}"
        )
        return text

    async def _take_turn(self, text: str) -> None:
        """One caller turn: LLM (with tools) → spoken reply."""
        self.messages.append({"role": "user", "content": text})

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            extra_body = None
            if "HF_LLM_URL" in os.environ:
                # Reasoning-capable models behind vLLM think when they judge a
                # turn hard — measured 28 s on the verification turn, against
                # sub-second everywhere else. A caller cannot wait on that.
                # Templates without a `thinking` flag simply ignore the kwarg.
                extra_body = {"chat_template_kwargs": {"thinking": False}}
            completion = await _llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body=extra_body,
            )
            # Keep only the portable OpenAI-schema fields. gpt-oss on Groq
            # returns extra reasoning fields in the message, and replaying
            # those must not break when HF_LLM_MODEL pins another provider.
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
            # Sequential on purpose: calls share verification and payment
            # state, so the emitted order is the order they run in.
            for call in tool_calls:
                self.messages.append(await self._run_tool(call))
        else:
            raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")

        if msg.get("content"):
            logger.info(f"[llm] rory_said: {msg['content'][:200]}")
            await self._speak(msg["content"])

    async def _run_tool(self, call: dict) -> dict:
        """Dispatch one tool call and build the tool message for the next round."""
        name = call["function"]["name"]
        raw = call["function"]["arguments"]
        logger.info(f"[tool] {name} args={str(raw)[:200]}")
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        # Every vendor client is blocking, and this coroutine shares the loop
        # with the audio pump. Called inline, a ten-second vendor timeout is
        # ten seconds of dead air — the same reason the Pipecat handler hops
        # threads.
        output = await asyncio.to_thread(dispatch, self.session, name, args)
        logger.info(f"[tool] {name} -> {json.dumps(output, default=str)[:200]}")
        return {
            "role": "tool",
            "name": name,
            "tool_call_id": call["id"],
            "content": json.dumps(output, default=str),
        }

    async def _speak(self, text: str) -> None:
        """Speak the last assistant message, cancellable by barge-in.

        ``text`` is that message's content, already on ``messages`` so a
        barge-in can cut it down to what the caller heard.
        """
        self._sent_bytes = 0
        self._total_bytes = 0
        self._speaking = asyncio.create_task(self._speak_now(text))
        try:
            await self._speaking
        except asyncio.CancelledError:
            # A barge-in cancels only the inner pacer. If the worker itself is
            # being cancelled (the call is ending), swallowing this would leave
            # it parked on the queue forever and the handler never exiting.
            if asyncio.current_task().cancelling():
                raise
            self._trim_interrupted_reply(text)
        finally:
            self._speaking = None

    def _trim_interrupted_reply(self, text: str) -> None:
        """Rewrite the last assistant message to the part of ``text`` that went out.

        Without this the history says Rory told the caller things they never
        heard, and the next turn reasons from that. Kokoro returns the reply
        as one file with no per-sentence boundaries, so the cut is
        proportional: bytes sent over total bytes, applied to the reply's
        words. Bytes handed to the actor count as heard — the pacer runs at
        most PLAYBACK_LEAD_S ahead of playback, so the overshoot is bounded
        by one lead.
        """
        if self._sent_bytes == 0:
            self.messages.pop()
            logger.info("[voice] barge-in: dropped reply, nothing sent before the cut")
            return
        heard_ms = self._sent_bytes // PCM16_BYTES_PER_MS
        total_ms = self._total_bytes // PCM16_BYTES_PER_MS
        words = text.split()
        kept = " ".join(words[: len(words) * heard_ms // total_ms])
        self.messages[-1] = {"role": "assistant", "content": f"{kept}{INTERRUPTED_MARKER}"}
        logger.info(
            f'[voice] barge-in: trimmed reply to {heard_ms} ms / {total_ms} ms ("{kept[:60]}")'
        )

    async def _speak_now(self, text: str) -> None:
        """Synthesize the reply, then pace the audio out against playback.

        Either route hands back the whole reply at once, so pacing is what
        preserves barge-in — blast
        it and the actor's buffer already holds the full reply by the time
        the caller interrupts. Chunks go out no more than PLAYBACK_LEAD_S
        ahead of real-time playback; cancelling this task strands the rest
        unsent.
        """
        t0 = time.monotonic()
        pcm = await _synthesize(text)
        self._total_bytes = len(pcm)
        total_s = len(pcm) / 2 / ACTOR_RATE_HZ
        logger.info(
            f"[tts] {total_s:.1f}s audio for {len(text)} chars in {time.monotonic() - t0:.2f}s"
        )

        chunk_bytes = int(PLAYBACK_CHUNK_S * ACTOR_RATE_HZ) * 2
        t_play = time.monotonic()  # playback clock starts at first byte sent
        sent_s = 0.0
        for i in range(0, len(pcm), chunk_bytes):
            ahead = sent_s - (time.monotonic() - t_play)
            if ahead > PLAYBACK_LEAD_S:
                await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
            chunk = pcm[i : i + chunk_bytes]
            await self.actor_ws.send_bytes(chunk)
            self._sent_bytes += len(chunk)
            sent_s += len(chunk) / 2 / ACTOR_RATE_HZ
        logger.info(f"[tts] spoke {total_s:.1f}s in {time.monotonic() - t0:.2f}s")
