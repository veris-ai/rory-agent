"""Rory on the Gemini Live API — Acme Energy's voice support agent, speech-to-speech.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it opens its own Gemini Live session (server-to-server, via
the ``google-genai`` SDK), registers Rory's 16 tools, and bridges audio and
function calls in both directions. The agent speaks first.

Unlike the Pipecat cascade there is no STT/LLM/TTS pipeline to assemble:
Gemini Live *is* the whole voice loop. What this module adds is the ``voice_ws``
bridging, the per-call ``CallSession``, and the hop onto a worker thread for the
blocking vendor calls.

Everything below the transport is shared with the Pipecat agent — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
two exist to be compared, so the transport is the only thing allowed to differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. Gemini
Live *requires* 16 kHz PCM16 input and *emits* 24 kHz PCM16 output, so only the
inbound leg is resampled; Gemini's output goes back to the actor untouched.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import os
import time

from google import genai
from google.genai import errors, types
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

GEMINI_LIVE_MODEL = os.environ.get(
    "GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
)
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Puck")

# The Veris actor speaks and listens at 24 kHz. Gemini Live is fixed at 16 kHz
# input / 24 kHz output, so only the input leg needs resampling.
ACTOR_RATE_HZ = 24000
GEMINI_INPUT_RATE_HZ = 16000

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _build_config() -> types.LiveConnectConfig:
    """The Live session config, pinned so a run is reproducible.

    The date context is a second system instruction rather than part of
    ``agent_desc.txt`` because it is resolved per call, off the frozen clock the
    attempt restored with its snapshot.
    """
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=f"{AGENT_PROMPT}\n\n{today_context()}",
        tools=TOOLS,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
            )
        ),
        # Turn-taking parity with the Pipecat candidate, which lands at ~0.8 s
        # of effective end-of-turn silence (0.2 s Silero stop_secs + 0.6 s
        # speech timeout). Gemini keeps its own VAD model — only the threshold
        # is matched — and its unset server default is undocumented, so pinning
        # this is what makes the two candidates comparable on interruption and
        # latency rather than on an accident of defaults.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                silence_duration_ms=800,
                prefix_padding_ms=300,
            )
        ),
        # Both legs transcribed so agent.log shows what was said, and so the
        # graded trace carries the same speech the Pipecat candidate's STT
        # would have produced.
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Gemini Live session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the Pipecat agent builds
    one per ``PipelineTask``: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    t_start = time.monotonic()
    try:
        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL, config=_build_config()
        ) as live:
            logger.info(
                f"[voice] Gemini Live connected model={GEMINI_LIVE_MODEL} "
                f"voice={GEMINI_VOICE} in={GEMINI_INPUT_RATE_HZ}Hz out={ACTOR_RATE_HZ}Hz"
            )
            # Native audio generates the greeting from a user trigger. Unlike
            # Pipecat's separate TTS, this can paraphrase the requested wording.
            await live.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=f"<call connected — greet the caller with exactly: {GREETING}>")],
                ),
                turn_complete=True,
            )
            logger.info("[voice] sent greeting trigger (agent greets first)")

            up = asyncio.create_task(_pump_actor_to_gemini(actor_ws, live), name="actor->gemini")
            down = asyncio.create_task(
                _pump_gemini_to_actor(live, actor_ws, session_state), name="gemini->actor"
            )
            try:
                done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in (up, down):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(up, down, return_exceptions=True)
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_gemini(actor_ws: WebSocket, live) -> None:
    """Binary PCM16 frames from the actor (24 kHz) → Gemini Live input (16 kHz)."""
    n_frames = 0
    n_bytes = 0
    resample_state = None  # carried across ratecv calls so the resample is continuous
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            pcm16, resample_state = audioop.ratecv(
                frame, 2, 1, ACTOR_RATE_HZ, GEMINI_INPUT_RATE_HZ, resample_state
            )
            if n_frames == 1:
                logger.info(f"[a->g] first frame bytes={len(frame)} (->{len(pcm16)} @16kHz)")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->g] forwarded {n_frames} frames ({n_bytes} bytes @24kHz)")
            await live.send_realtime_input(
                audio=types.Blob(data=pcm16, mime_type=f"audio/pcm;rate={GEMINI_INPUT_RATE_HZ}")
            )
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->g] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->g] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_gemini_to_actor(live, actor_ws: WebSocket, session_state: CallSession) -> None:
    """Gemini Live events → 24 kHz audio back to the actor; dispatch tool calls.

    Gemini emits 24 kHz PCM16, which matches the actor's rate, so audio streams
    straight through. After a tool call the result goes back with
    ``send_tool_response`` and Gemini resumes the turn on its own.

    ``live.receive()`` yields one *complete model turn* and then returns (it
    breaks internally on ``turn_complete``), so it is wrapped in an outer loop
    and re-entered per turn — otherwise the pump would exit after the opening
    greeting and drop the call.
    """
    n_audio_frames = 0
    n_audio_bytes = 0
    n_turns = 0

    try:
        while True:
            agent_transcript = ""
            async for response in live.receive():

                audio = response.data
                if audio:
                    n_audio_frames += 1
                    n_audio_bytes += len(audio)
                    if n_audio_frames == 1:
                        logger.info(f"[g->a] first audio chunk bytes={len(audio)}")
                    elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                        logger.info(
                            f"[g->a] streamed {n_audio_frames} chunks ({n_audio_bytes} bytes)"
                        )
                    await actor_ws.send_bytes(audio)

                sc = response.server_content
                if sc is not None:
                    if sc.output_transcription and sc.output_transcription.text:
                        agent_transcript += sc.output_transcription.text
                    if sc.input_transcription and sc.input_transcription.text:
                        logger.info(f"[g->a] caller_said: {sc.input_transcription.text}")
                    if sc.interrupted:
                        logger.info("[g->a] turn interrupted (barge-in)")
                    if sc.turn_complete:
                        n_turns += 1
                        if agent_transcript.strip():
                            logger.info(
                                f"[g->a] rory_said (turn #{n_turns}): "
                                f"{agent_transcript.strip()}"
                            )
                        agent_transcript = ""

                if response.tool_call and response.tool_call.function_calls:
                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        args = dict(fc.args or {})
                        # Every vendor client is blocking, and this coroutine is
                        # carrying the caller's audio. Called inline, a ten-second
                        # vendor timeout is ten seconds of dead air — the same
                        # reason the Pipecat handler hops threads.
                        result = await asyncio.to_thread(dispatch, session_state, fc.name, args)
                        logger.info(
                            f"[g->a] tool {fc.name} -> {json.dumps(result, default=str)}"
                        )
                        function_responses.append(
                            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                        )
                    await live.send_tool_response(function_responses=function_responses)

    except errors.APIError as exc:
        if exc.code not in (1000, 1001):
            raise
        logger.info(f"[g->a] session closed; {n_turns} turns, {n_audio_bytes} audio bytes")
    except WebSocketDisconnect:
        logger.info("[g->a] actor WS closed; ending Gemini receive loop")
