"""Rory on Smallest AI's Atoms platform — Acme Energy's voice support agent.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor
and bridges it to one Atoms realtime session per call, over the unified
websocket endpoint:

    wss://api.smallest.ai/atoms/v1/agent/connect

Like Hydra, Grok Voice and Gemini Live, Atoms *is* the whole voice loop — one
speech-to-speech session, no STT/LLM/TTS pipeline to assemble here, and
turn-taking and barge-in are owned server-side. What this module adds is the
``voice_ws`` bridging, the per-call ``CallSession``, and the hop onto a
worker thread for the blocking vendor calls, the same shape every other
speech-to-speech candidate in this repo uses.

Two things make Atoms unlike every other candidate here, confirmed against
Smallest's public docs (the WebSocket SDK page, the Realtime Agent WebSocket
API reference, and the client-tools page — none of it undocumented or
guessed):

1. There is no per-connection session configuration at all. The prompt,
   greeting, voice, and turn-taking settings (silence/release timing, "smart
   turn detection" — Atoms' equivalent of the benchmark's MaAI) all live on
   the Atoms AGENT, set once through the console or the agent's versioned
   config, not sent here. ``AGENT_PROMPT``/``GREETING`` below are still
   imported from the shared module so this transport asserts the same parity
   contract as every other candidate (``rory_tools.parity.assert_shares_prompt_and_greeting``),
   but they are read, not transmitted — a human has to push them into the
   agent itself. See the README for the exact values to set and why this
   repo cannot script that push.
2. Tool calls only reach this pod if Rory's 16 tools are registered as
   Atoms **client tools** (not server-side "API tools", which Atoms would
   call directly over HTTP, bypassing ``rory_tools.dispatch`` — and this
   repo's harness — entirely). ``rory_atoms.tools.provisioning_payload()``
   is the exact definition to push for each one.

Sample-rate note: the Veris actor speaks and listens at 24 kHz PCM16, and
Atoms' connection accepts an explicit ``input_audio_format``/
``output_audio_format`` query parameter rather than negotiating one, so both
legs are opened at ``pcm_24000`` and no resampling runs in either direction —
unlike Hydra, whose input leg is pinned at 16 kHz regardless of what is asked
for.
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
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt

from .tools import TOOLS

ATOMS_BASE_URL = "wss://api.smallest.ai/atoms/v1/agent/connect"
ATOMS_AGENT_ID = os.environ.get("ATOMS_AGENT_ID", "")

# The actor's rate both ways, requested explicitly on the connection query
# string rather than negotiated — Atoms documents this as one of a fixed set
# of pcm_{8000,16000,22050,24000,44100,48000} values, and 24000 is in it, so
# both legs run at the actor's native rate with no resampling.
ACTOR_RATE_HZ = 24000
AUDIO_FORMAT = f"pcm_{ACTOR_RATE_HZ}"

# webcall is Atoms' audio mode (the other documented mode, "chat", is text
# over input_text.send/transcript — not this bridge).
CONNECT_MODE = "webcall"

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()


def _atoms_url() -> str:
    """Built per call so a missing credential or agent id fails at connect, not at import."""
    return ATOMS_BASE_URL + "?" + urlencode({
        "token": os.environ["SMALLEST_API_KEY"],
        "agent_id": ATOMS_AGENT_ID,
        "mode": CONNECT_MODE,
        "input_audio_format": AUDIO_FORMAT,
        "output_audio_format": AUDIO_FORMAT,
    })


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection <-> one Atoms session with Rory's tools.

    A fresh ``CallSession`` per connection, exactly as the other
    speech-to-speech candidates build one per call: it holds the account
    ``verify_caller`` matched and dies with the call, so identity never
    becomes a model argument.
    """
    peer = f"{actor_ws.client.host}:{actor_ws.client.port}" if actor_ws.client else "?"
    logger.info(f"[voice] actor connected peer={peer}")
    session_state = CallSession()

    t_start = time.monotonic()
    try:
        async with websockets.connect(_atoms_url(), max_size=None) as atoms_ws:
            created = json.loads(await atoms_ws.recv())
            if created.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, got {created.get('type')!r}")
            sample_rate = created.get("sample_rate")
            if sample_rate != ACTOR_RATE_HZ:
                raise RuntimeError(
                    f"Atoms announced sample_rate={sample_rate}, expected {ACTOR_RATE_HZ} "
                    f"(requested via input_audio_format/output_audio_format={AUDIO_FORMAT})"
                )
            logger.info(
                f"[voice] Atoms connected agent_id={ATOMS_AGENT_ID} "
                f"session_id={created.get('session_id')} call_id={created.get('call_id')} "
                f"sample_rate={sample_rate}"
            )

            up = asyncio.create_task(_pump_actor_to_atoms(actor_ws, atoms_ws), name="actor->atoms")
            down = asyncio.create_task(
                _pump_atoms_to_actor(atoms_ws, actor_ws, session_state),
                name="atoms->actor",
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


async def _pump_actor_to_atoms(actor_ws: WebSocket, atoms_ws) -> None:
    """Binary PCM16 frames from the actor -> Atoms' input buffer, both at 24 kHz."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->at] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->at] forwarded {n_frames} frames ({n_bytes} bytes)")
            await atoms_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->at] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead
        # of a binary one — a protocol mismatch on the actor side, not a bug
        # here (the same defensive case Hydra's pump handles).
        logger.error(
            f"[a->at] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_atoms_to_actor(
    atoms_ws, actor_ws: WebSocket, session_state: CallSession
) -> None:
    """Atoms events -> audio bytes back to the actor; dispatch tool calls.

    No response needs to be explicitly resumed after a tool call the way
    Hydra's ``response.create`` does: Atoms' documented client message list
    has no such message at all, and the client-tools doc describes nothing
    the app sends beyond ``function_call.result`` — the agent is expected to
    continue on its own once it has the result.
    """
    n_audio_frames = 0
    n_audio_bytes = 0

    try:
        async for raw in atoms_ws:
            evt = json.loads(raw)
            etype = evt.get("type", "")

            if etype == "output_audio.delta":
                audio = base64.b64decode(evt.get("audio", ""))
                n_audio_frames += 1
                n_audio_bytes += len(audio)
                if n_audio_frames == 1:
                    logger.info(f"[at->a] first audio delta bytes={len(audio)}")
                elif n_audio_frames % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        f"[at->a] streamed {n_audio_frames} audio deltas "
                        f"({n_audio_bytes} bytes) so far this session"
                    )
                if audio:
                    await actor_ws.send_bytes(audio)

            elif etype == "agent_start_talking":
                logger.info("[at->a] agent_start_talking")

            elif etype == "agent_stop_talking":
                logger.info("[at->a] agent_stop_talking")

            elif etype == "interruption":
                # The caller barged in. Atoms owns stopping its own
                # generation server-side; there is nothing documented for
                # this bridge to send back — it just stops receiving further
                # output_audio.delta for the interrupted turn.
                logger.info("[at->a] interruption (barge-in)")

            elif etype in ("transcript", "transcript.delta"):
                continue  # logged at debug volume only; not needed to bridge the call

            elif etype == "function_call":
                name = evt["name"]
                call_id = evt["call_id"]
                raw_args = evt.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Not confirmed in Atoms' docs the way Hydra's truncated-
                    # arguments case is, but the same shape of failure (a
                    # barge-in cutting generation off mid-argument) is
                    # possible on any realtime tool-calling wire, so the same
                    # defensive handling applies: a call that never finished
                    # streaming is not dispatched.
                    logger.warning(
                        f"[at->a] tool {name} ({call_id}) arguments truncated — "
                        f"not dispatched: {raw_args!r}"
                    )
                    continue
                # Every vendor client is blocking, and this coroutine is
                # carrying the caller's audio. Called inline, a vendor
                # timeout is dead air on the mic leg — the same reason every
                # other speech-to-speech candidate hops threads.
                result = await asyncio.to_thread(dispatch, session_state, name, args)
                logger.info(
                    f"[at->a] tool {name} ({call_id}) -> {json.dumps(result, default=str)}"
                )
                await atoms_ws.send(json.dumps({
                    "type": "function_call.result",
                    "call_id": call_id,
                    "output": json.dumps(result, default=str),
                }))

            elif etype == "session.closed":
                logger.info(f"[at->a] session.closed reason={evt.get('reason')}")

            elif etype == "error":
                logger.error(f"[at->a] Atoms error: {evt.get('code')} {evt.get('message')}")
                raise RuntimeError(f"Atoms error: {evt.get('code')} {evt.get('message')}")
    except websockets.ConnectionClosed as exc:
        logger.info(
            f"[at->a] Atoms WS closed code={exc.code} reason={exc.reason}; "
            f"{n_audio_bytes} audio bytes streamed"
        )
    except WebSocketDisconnect:
        logger.info("[at->a] actor WS closed; ending Atoms receive loop")
