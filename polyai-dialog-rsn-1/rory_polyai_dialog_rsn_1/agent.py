"""Rory on PolyAI's Dialog-RSN-1, bridged from one voice_ws call to its realtime API.

Dialog-RSN-1 is PolyAI's audio-native dialog model. Its realtime WebSocket
speaks the OpenAI Realtime wire format: the model hears the caller's raw
audio, decides when their turn is over, and writes either a reply or one tool
call. The server voices replies itself (a Cartesia voice) and sends that audio
at the pace it plays. Turn-taking, barge-in, transcription and speech are all
PolyAI's. This module is the bridge: one PolyAI socket per call, a fresh
``CallSession``, and the shared dispatcher on a worker thread.

This is not the ``polyai/`` candidate. That one pushes Rory onto an Agent
Studio project and runs whatever model the project is set to; this one
calls Dialog-RSN-1 directly with its own prompt and the shared tools per call.

Where it differs from the other OpenAI-Realtime-shaped bridges:

- Its own prompt, ``agent_desc.txt`` beside this module. Dialog-RSN-1 treats
  any text it writes as the end of its turn, so the shared prompt's "tell the
  caller, then transfer" lines would end the turn before the transfer. The
  copy here puts the words in the tool call's spoken ``utterance`` (see
  tools.py); the rest is the shared prompt verbatim.
- No ``conversation.item.truncate`` on barge-in. The server speaks and paces
  the reply, so it knows what the caller heard and cuts history itself.
- Each tool runs on its own task the moment its call is written, as PolyAI's
  OpenAI migration guide says. PolyAI speaks the call's ``utterance`` while
  the tool runs and sends that audio at the pace it plays, so running the
  tool inline would stall the line, and ``response.done`` arrives only once
  the line has played.
- No turn timing is pinned: the model confirms every turn end, and the
  service ignores the silence settings the other candidates pin.

Both legs are PCM16 at 24 kHz, so neither is resampled.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, today_context

from .tools import TOOLS

# Self-serve keys from studio.poly.ai work only here; an enterprise key from
# studio.us.poly.ai needs wss://api.us.poly.ai/v1/realtime.
REALTIME_URL = os.environ.get("POLYAI_RSN_URL", "wss://api.studio.poly.ai/v1/realtime")
# A Cartesia voice id. The default is the voice the cartesia candidate speaks
# with ("Skylar – Friendly Guide"), so the two share a voice.
RSN_VOICE = os.environ.get("POLYAI_RSN_VOICE", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")
SAMPLE_RATE_HZ = 24000
AGENT_PROMPT = Path(__file__).with_name("agent_desc.txt").read_text()


def _session_update() -> dict:
    """The session config. The date context is resolved per call off the frozen clock."""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                    # Changes nothing on PolyAI's side: the model always
                    # transcribes the caller. The bench's openai-realtime trace
                    # adapter reads it to hold the caller's turn in place at
                    # commit; without it the transcript, which arrives during
                    # the reply, lands after the reply in the judge's trace.
                    "transcription": {"language": "en"},
                    # create_response is the only turn_detection field the
                    # service reads: true has the model confirm each turn end
                    # and the server start the reply.
                    "turn_detection": {"type": "server_vad", "create_response": True},
                },
                "output": {"voice": RSN_VOICE},
            },
            "tools": TOOLS,
        },
    }


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one Dialog-RSN-1 session with Rory's tools."""
    session_state = CallSession()
    headers = {"X-API-KEY": os.environ["POLY_API_KEY"]}
    async with websockets.connect(REALTIME_URL, additional_headers=headers) as rsn_ws:
        event = json.loads(await rsn_ws.recv())
        if event["type"] != "session.created":
            raise RuntimeError(f"Dialog-RSN-1 session failed: {event}")
        await rsn_ws.send(json.dumps(_session_update()))
        # A refused update applies none of its fields, prompt and tools
        # included, so the call must not go on without the confirmation.
        event = json.loads(await rsn_ws.recv())
        if event["type"] != "session.updated":
            raise RuntimeError(f"Dialog-RSN-1 refused the session: {event}")
        logger.info(f"[voice] Dialog-RSN-1 session ready: {len(TOOLS)} tools, voice={RSN_VOICE}")

        # Agent speaks first. The shared prompt says the greeting already
        # happened, so this one response is told to say it.
        await rsn_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "instructions": (
                    "Open the call by greeting the caller now. Say exactly: "
                    f'"{GREETING}" Then stop and wait for the caller to respond.'
                ),
            },
        }))

        calls: asyncio.Queue = asyncio.Queue()
        tasks = {
            asyncio.create_task(_pump_actor_to_rsn(actor_ws, rsn_ws)),
            asyncio.create_task(_pump_rsn_to_actor(rsn_ws, actor_ws, calls)),
            asyncio.create_task(_run_tools(rsn_ws, session_state, calls)),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _pump_actor_to_rsn(actor_ws: WebSocket, rsn_ws) -> None:
    """Binary PCM16 frames from the actor → the Dialog-RSN-1 input buffer."""
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            await rsn_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect:
        logger.info("[voice] actor disconnected")


async def _pump_rsn_to_actor(rsn_ws, actor_ws: WebSocket, calls: asyncio.Queue) -> None:
    """Dialog-RSN-1 events → reply audio to the actor; tool calls and response ends → ``calls``."""
    async for raw in rsn_ws:
        event = json.loads(raw)
        event_type = event["type"]

        if event_type == "response.output_audio.delta":
            await actor_ws.send_bytes(base64.b64decode(event["delta"]))

        elif event_type == "response.output_audio_transcript.done":
            logger.info(f"[voice] rory_said: {event['transcript']}")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            logger.info(f"[voice] caller_said: {event['transcript']}")

        elif event_type == "response.function_call_arguments.done":
            calls.put_nowait(event)

        elif event_type == "response.done":
            response = event["response"]
            if response["status"] not in ("completed", "cancelled"):
                raise RuntimeError(f"Dialog-RSN-1 response {response['status']}: {response}")
            calls.put_nowait(response["status"])

        elif event_type == "error":
            if event["error"]["code"] != "conversation_already_has_active_response":
                raise RuntimeError(f"Dialog-RSN-1 error: {event['error']}")
            calls.put_nowait("refused")


async def _run_tools(rsn_ws, session_state: CallSession, calls: asyncio.Queue) -> None:
    """Run each tool call in the order written; continue once its response is done.

    A response's status is queued behind its calls, so the ``response.create``
    that follows waits for both the results and the spoken line, since a tool
    result never starts a turn on its own. A cancelled response gets none: the
    caller's barge-in starts its own, and the results it missed are in the
    conversation for the turn after.

    If the caller spoke during the round, the server may have opened a
    response of its own, and the ``response.create`` is refused with
    ``conversation_already_has_active_response``. PolyAI's error table makes
    that retryable after ``response.done``, so it is sent again when the open
    response ends, at most once per response end. The server answers it at
    once, or holds it for the caller's turn.
    """
    answered = False
    refused = False
    while True:
        call = await calls.get()
        if call == "refused":
            refused = True
            continue
        if isinstance(call, str):
            if (call == "completed" and answered) or refused:
                await rsn_ws.send(json.dumps({"type": "response.create"}))
            answered = refused = False
            continue
        args = json.loads(call["arguments"])
        # Every vendor client blocks, so each runs on a worker thread; one at
        # a time, since they share account state.
        result = await asyncio.to_thread(dispatch, session_state, call["name"], args)
        logger.info(f"[voice] tool {call['name']} ({call['call_id']}) -> {json.dumps(result, default=str)}")
        await rsn_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, default=str),
            },
        }))
        answered = True
