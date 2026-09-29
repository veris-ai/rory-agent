"""Rory on ElevenLabs Conversational AI — Acme Energy's voice support agent on a hosted runtime.

ElevenLabs runs the whole voice loop on its side — ASR, the LLM, TTS and
turn-taking — behind a stored *agent* that carries the prompt, the opening line
and the tool declarations. What this module adds is the provisioning of that
agent, and per ``/voice`` connection the bridge: one ``AsyncConversation``, an
``AsyncAudioInterface`` shuttling PCM16 between the actor and the platform, and
Rory's 16 tools registered as client tools so every call the hosted LLM makes
comes back down the same socket and resolves here through ``rory_tools.dispatch``.

Everything below the transport is shared with the other agents — same prompt,
same greeting, same frozen-clock date context, same tools, same dispatcher. The
candidates exist to be compared, so the transport is the only thing allowed to
differ.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16. The
ElevenLabs audio interface carries whatever format the agent was provisioned
with — the SDK's docstring says 16 kHz, but that is the platform default, not a
constraint — so the agent is created with ``pcm_24000`` on both legs and audio
passes through untouched in both directions.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Awaitable, Callable

from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import AsyncAudioInterface, AsyncConversation
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS, build_client_tools

# Creation-time only: a pinned AGENT_ID keeps whatever it was created with.
ELEVENLABS_LLM = os.environ.get("ELEVENLABS_LLM", "gpt-4.1-mini")
# English ConvAI agents reject the v2_5 / v3 variants.
ELEVENLABS_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

# The Veris actor speaks and listens at 24 kHz; the agent is provisioned to
# match on both legs, so neither direction is resampled.
ACTOR_RATE_HZ = 24000
AUDIO_FORMAT = f"pcm_{ACTOR_RATE_HZ}"

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _build_conversation_config() -> dict:
    """The stored agent's config, pinned so a run is reproducible.

    A dict rather than the SDK's nested models: pydantic accepts one that
    matches the schema, and this is the shape the platform documents. The
    date context is appended to the prompt because it is resolved off the
    frozen clock the attempt restored with its snapshot — provisioning happens
    once per boot and the bench runs one call per pod, so it is that call's
    today.
    """
    return {
        "agent": {
            "first_message": GREETING,
            "language": "en",
            "prompt": {
                "prompt": f"{AGENT_PROMPT}\n\n{today_context()}",
                "llm": ELEVENLABS_LLM,
                "temperature": 0.3,
                "tools": TOOLS,
                # The platform otherwise prepends generic agent boilerplate
                # that conflicts with the Rory persona.
                "ignore_default_personality": True,
            },
        },
        "tts": {
            "model_id": ELEVENLABS_TTS_MODEL,
            "voice_id": ELEVENLABS_VOICE_ID,
            "agent_output_audio_format": AUDIO_FORMAT,
        },
        "asr": {
            "quality": "high",
            "user_input_audio_format": AUDIO_FORMAT,
        },
        # ElevenLabs' end-of-turn is its own learned turn model and exposes no
        # silence threshold to pin to the ~0.8 s the other candidates share;
        # turn_timeout is an inactivity re-prompt, not the endpoint. Eagerness
        # is pinned to the default so the setting is explicit rather than
        # inherited.
        "turn": {
            "turn_eagerness": "normal",
        },
    }


def ensure_agent(client: ElevenLabs) -> str:
    """The stored agent every conversation on this pod runs against.

    ``AGENT_ID`` pins an existing one; otherwise a fresh agent is created from
    the shared prompt, greeting and tools. Runs at boot, before the readiness
    probe passes, so a rejected config fails the pod rather than each call.
    """
    pinned = os.environ.get("AGENT_ID")
    if pinned:
        logger.info(f"[agent] using pinned AGENT_ID={pinned}")
        return pinned

    response = client.conversational_ai.agents.create(
        name="rory-elevenlabs (Rory @ Acme Energy)",
        conversation_config=_build_conversation_config(),
        tags=["rory", "voice"],
    )
    logger.warning(
        f"[agent] created ElevenLabs agent {response.agent_id} "
        f"llm={ELEVENLABS_LLM} tts={ELEVENLABS_TTS_MODEL} voice={ELEVENLABS_VOICE_ID} "
        f"audio={AUDIO_FORMAT} — set AGENT_ID={response.agent_id} to reuse it"
    )
    return response.agent_id


class ActorAudioInterface(AsyncAudioInterface):
    """The actor WebSocket as the conversation's microphone and speaker.

    ``start`` hands over the SDK's ``input_callback``, which the actor read
    loop then drives through ``push_actor_audio``; ``output`` is the SDK
    delivering the agent's audio, forwarded as-is. Both legs are
    ``pcm_24000`` by agent config, so nothing is resampled.
    """

    def __init__(self, actor_ws: WebSocket) -> None:
        self._actor_ws = actor_ws
        self._input_callback: Callable[[bytes], Awaitable[None]] | None = None
        self._stopped = False
        self.n_in = 0
        self.n_out = 0
        self.bytes_in = 0
        self.bytes_out = 0

    async def start(self, input_callback: Callable[[bytes], Awaitable[None]]) -> None:
        self._input_callback = input_callback
        logger.info("[audio] interface started — platform socket up, forwarding actor frames")

    async def stop(self) -> None:
        self._stopped = True
        logger.info(
            f"[audio] interface stopped in={self.n_in} frames/{self.bytes_in} bytes "
            f"out={self.n_out} chunks/{self.bytes_out} bytes"
        )

    async def output(self, audio: bytes) -> None:
        self.n_out += 1
        self.bytes_out += len(audio)
        if self.n_out == 1:
            logger.info(f"[el->a] first audio chunk bytes={len(audio)}")
        elif self.n_out % LOG_EVERY_N_FRAMES == 0:
            logger.info(f"[el->a] streamed {self.n_out} chunks ({self.bytes_out} bytes)")
        await self._actor_ws.send_bytes(audio)

    async def interrupt(self) -> None:
        # Barge-in. Chunks go straight to the actor, so there is nothing
        # buffered here to drop; the platform has already stopped sending.
        logger.info("[el->a] interrupt (barge-in)")

    async def push_actor_audio(self, frame: bytes) -> None:
        """One PCM16 frame from the actor → the platform."""
        # Frames that arrive before the platform socket is up have nowhere to
        # go, and the SDK contract forbids the callback after stop(). Nothing
        # said is lost either way: the agent speaks first, and a stopped
        # session is already over.
        if self._input_callback is None or self._stopped:
            return
        self.n_in += 1
        self.bytes_in += len(frame)
        if self.n_in == 1:
            logger.info(f"[a->el] first actor frame bytes={len(frame)}")
        elif self.n_in % LOG_EVERY_N_FRAMES == 0:
            logger.info(f"[a->el] forwarded {self.n_in} frames ({self.bytes_in} bytes)")
        await self._input_callback(frame)


async def _log_rory_said(text: str) -> None:
    logger.info(f"[el->a] rory_said: {text}")


async def _log_rory_said_corrected(original: str, corrected: str) -> None:
    # After a barge-in the platform reports what was actually voiced.
    logger.info(f"[el->a] rory_said (cut short): {corrected}")


async def _log_caller_said(text: str) -> None:
    logger.info(f"[a->el] caller_said: {text}")


async def _log_latency(ms: int) -> None:
    logger.info(f"[el] latency={ms} ms")


async def run_voice_ws_bot(actor_ws: WebSocket, client: ElevenLabs, agent_id: str) -> None:
    """One actor connection ↔ one ElevenLabs conversation with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other transports
    build one per call: it holds the account ``verify_caller`` matched and
    dies with the call, so identity never becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()
    audio = ActorAudioInterface(actor_ws)
    conversation = AsyncConversation(
        client=client,
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=audio,
        client_tools=build_client_tools(session_state, asyncio.get_running_loop()),
        callback_agent_response=_log_rory_said,
        callback_agent_response_correction=_log_rory_said_corrected,
        callback_user_transcript=_log_caller_said,
        callback_latency_measurement=_log_latency,
    )

    t_start = time.monotonic()
    try:
        await conversation.start_session()
        logger.info(f"[voice] ElevenLabs session started agent_id={agent_id} audio={AUDIO_FORMAT}")
        up = asyncio.create_task(_pump_actor_to_elevenlabs(actor_ws, audio), name="actor->elevenlabs")
        # Completes when the platform side ends — a normal close, or the SDK
        # giving up on its socket — so a call the platform hangs up on does
        # not sit waiting for actor frames that will never come.
        down = asyncio.create_task(conversation.wait_for_session_end(), name="elevenlabs->actor")
        try:
            done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            if not down.done():
                await conversation.end_session()
            if not up.done():
                up.cancel()
            await asyncio.gather(up, down, return_exceptions=True)
    finally:
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_elevenlabs(actor_ws: WebSocket, audio: ActorAudioInterface) -> None:
    """Binary PCM16 frames from the actor → the conversation's input."""
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            await audio.push_actor_audio(frame)
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->el] actor disconnected after {audio.n_in} frames "
            f"({audio.bytes_in} bytes): code={exc.code}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->el] non-binary frame after {audio.n_in} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise
