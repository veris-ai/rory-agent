"""rory-pipecat — Rory customer-service agent on Pipecat.

A cascaded Pipecat pipeline — Deepgram STT, an OpenAI gpt-4.1-mini chat LLM,
and ElevenLabs TTS. Rory handles Acme Energy billing questions and payment
assistance against two real vendor APIs — Stripe for billing and Apache
Fineract for payment arrangements — which the bench points at its twins:

    transport.input() -> stt -> user_aggregator -> llm -> tts
    -> transport.output() -> assistant_aggregator

The transport is Veris's ``voice_ws`` channel: ``run_voice_ws_bot(websocket)``
wraps a FastAPI ``WebSocket`` (accepted by ``rory_pipecat/web.py:/voice``) in a
``FastAPIWebsocketTransport`` with ``RawPCM16Serializer``. The actor's
PCM16/24 kHz frames are resampled to the pipeline's 16 kHz input rate on the
way in (Silero VAD runs only at 8 or 16 kHz); Rory's 24 kHz TTS audio goes
back out unchanged.
"""

from __future__ import annotations

import os

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from rory_tools import CallSession
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import HANDLERS, RORY_TOOLS
from .serializers import VERIS_SAMPLE_RATE_HZ, RawPCM16Serializer


AGENT_PROMPT = load_agent_prompt()

# The pipeline's audio rates, set once in PipelineParams; every service and the
# transport picks its rate up from there at setup. Input is 16 kHz because
# Silero VAD accepts only 8 or 16 kHz (the serializer resamples the actor's
# 24 kHz down to it). Output is the wire's 24 kHz, so TTS audio goes back out
# without resampling.
PIPELINE_IN_SAMPLE_RATE_HZ = 16000
PIPELINE_OUT_SAMPLE_RATE_HZ = VERIS_SAMPLE_RATE_HZ


# Silero VAD — the speech detector — constructed ONCE at import.
# uvicorn imports this module before it binds :8008, so the platform's TCP
# readiness probe only passes after it has loaded. Building it inside
# _build_pipeline_worker instead cold-loads the ONNX session on the first
# /voice connection (~4 s warm, but 25 s+ under concurrent cluster load),
# which races — and loses to — the actor's 10 s voice_ws connect timeout,
# dropping the call before it starts. pipecat has no global model cache, so we
# share the instance; the bench runs one call per container, and per-connection
# turn state (including Smart Turn, which loads in milliseconds) is built fresh
# in _build_pipeline_worker.
_SHARED_VAD = SileroVADAnalyzer()


def _build_llm() -> OpenAILLMService:
    """gpt-4.1-mini chat-completions LLM with Rory's account tools wired up."""
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
        ),
        # Calls share verification and payment state; preserve emitted order.
        run_in_parallel=False,
    )
    _register_tools(llm)
    return llm


def _register_tools(llm: LLMService) -> None:
    """Register Rory's tools so an interruption never hides what they did.

    Each handler runs its blocking vendor call on a thread, and a thread cannot
    be cancelled: with Pipecat's default (``cancel_on_interruption=True``) a
    caller who barges in mid-call cancelled the handler's await, the LLM was
    told the call was CANCELLED, and the payment or arrangement went through
    anyway. With ``cancel_on_interruption=False`` the call always completes and
    its real result reaches the LLM: in place, exactly as a synchronous result,
    when nothing interrupted; as a message that prompts a reply when the
    caller spoke in the meantime. Pipecat adds a note on such late results to
    the system prompt.
    """
    for name, handler in HANDLERS.items():
        llm.register_function(name, handler, cancel_on_interruption=False)


def _build_stt() -> DeepgramSTTService:
    """Deepgram STT on PCM16 at the pipeline's input rate."""
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        encoding="linear16",
        settings=DeepgramSTTService.Settings(
            model=os.environ.get("DEEPGRAM_MODEL", "nova-3-general"),
        ),
    )


def _build_tts() -> ElevenLabsTTSService:
    """ElevenLabs TTS, emitting PCM16 at the pipeline's output rate."""
    return ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice=os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model=os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2"),
        ),
    )


def _build_pipeline_worker(transport: BaseTransport) -> PipelineWorker:
    """Assemble the Pipecat pipeline for a voice_ws transport."""
    stt = _build_stt()
    llm = _build_llm()
    tts = _build_tts()

    # System prompt drives Rory. The opening greeting is seeded as the first
    # assistant turn (and also spoken on connect, see _run_with_transport) so it
    # is visible to the trace-based grader and the model doesn't repeat it.
    # Today's date, because a bill is only "last month's" relative to now and
    # an instalment date has to be real — left to guess, the model picks a year
    # from its training data and Fineract refuses the transaction outright.
    context = LLMContext(
        [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "system", "content": today_context()},
            {"role": "assistant", "content": GREETING},
        ],
        RORY_TOOLS,
    )

    # Pipecat's turn detection — a turn starts on voice activity or a
    # transcript, and Smart Turn v3 judges when the caller has finished — gated
    # on the LLM's own verdict. Every reply opens with ● (complete), ◐ or ○
    # (cut off, or needs time). Only ● is spoken; ◐ and ○ keep the turn open
    # and re-prompt the caller after 5 s or 10 s of silence. A tool call counts
    # as ●. Pipecat appends the marker instructions to the system prompt.
    #
    # A sound the STT can't transcribe (a cough, background noise) can still
    # start a turn and cut Rory off; when that empty turn interrupted a reply,
    # the aggregator's empty-turn recovery (``empty_user_turn``, on by default)
    # runs the LLM to ask the caller to repeat, so the call cannot stall.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_SHARED_VAD,
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    return PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PIPELINE_IN_SAMPLE_RATE_HZ,
            audio_out_sample_rate=PIPELINE_OUT_SAMPLE_RATE_HZ,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # Who the caller has proved themselves to be, for this call only. Tool
        # handlers read it as ``params.app_resources``; the model never sees it,
        # because it is not a tool argument.
        app_resources=CallSession(),
    )


async def _run_with_transport(transport: BaseTransport, label: str) -> None:
    """Spin up the pipeline for ``transport``, wire connect/disconnect
    handlers that greet then cancel, and block until the runner exits."""
    logger.info(f"[{label}] starting pipeline")
    worker = _build_pipeline_worker(transport)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client) -> None:
        logger.info(f"[{label}] client connected — speaking greeting")
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client) -> None:
        logger.info(f"[{label}] client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    logger.info(f"[{label}] pipeline finished")


async def run_voice_ws_bot(websocket) -> None:
    """Run a Pipecat pipeline for a single Veris ``voice_ws`` connection.

    The WebSocket is already accepted by the FastAPI route. Frames on the wire
    are raw PCM16/24 kHz/mono in both directions — see ``RawPCM16Serializer``.
    """
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=RawPCM16Serializer(),
        ),
    )
    await _run_with_transport(transport, label="voice_ws")
