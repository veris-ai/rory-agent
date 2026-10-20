"""Bridge one voice_ws call to ElevenLabs' standalone Realtime API.

Each call mints its own session-scoped client secret using the documented
authentication flow. Both audio legs are PCM16 at 24 kHz; tools and caller
verification stay in rory-core. No dashboard agent is needed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

import httpx
import websockets
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

# Max is suited to Rory's multi-step tool workflow; Mini is an env override.
REALTIME_MODEL = os.environ.get("ELEVENLABS_REALTIME_MODEL", "eleven_realtime_v1_max")
REALTIME_VOICE = os.environ.get("ELEVENLABS_REALTIME_VOICE", "brian")
REALTIME_URL = "wss://api.elevenlabs.io/v1/convai/realtime"
CLIENT_SECRETS_URL = "https://api.elevenlabs.io/v1/convai/realtime/client_secrets"
SAMPLE_RATE_HZ = 24000
PCM16_BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1000
AGENT_PROMPT = load_agent_prompt()


def _session_config() -> dict:
    """The initial session is authoritative when minting the client secret."""
    return {
        "type": "realtime",
        "model": REALTIME_MODEL,
        "instructions": f"{AGENT_PROMPT}\n\n{today_context()}",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                "transcription": {"language": "en"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "medium",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            # ElevenLabs' output format has no configurable sample-rate field.
            "output": {"format": {"type": "audio/pcm"}, "voice": REALTIME_VOICE},
        },
        "tools": TOOLS,
        "parallel_tool_calls": False,
    }


async def _create_client_secret() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            CLIENT_SECRETS_URL,
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            json={
                "session": _session_config(),
                "expires_after": {"anchor": "created_at", "seconds": 600},
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            exc.add_note(f"ElevenLabs Realtime response: {response.text}")
            raise
        return response.json()["value"]


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    session_state = CallSession()
    client_secret = await _create_client_secret()
    # This is the browser authentication wire used by the documented Agents
    # SDK integration, without depending on an unpublished server SDK flow.
    async with websockets.connect(
        REALTIME_URL,
        subprotocols=["realtime", f"openai-insecure-api-key.{client_secret}"],
    ) as realtime_ws:
        event = json.loads(await realtime_ws.recv())
        if event["type"] != "session.created":
            raise RuntimeError(f"ElevenLabs Realtime session failed: {event}")
        logger.info(f"[voice] ElevenLabs Realtime connected model={REALTIME_MODEL} voice={REALTIME_VOICE}")
        # ElevenLabs needs conversation input before it will generate a reply.
        await realtime_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "The call just connected. Greet the caller with exactly: "
                        f'"{GREETING}" Then stop and wait for the caller to respond.'
                    ),
                }],
            },
        }))
        await realtime_ws.send(json.dumps({"type": "response.create"}))

        up = asyncio.create_task(_pump_actor_to_realtime(actor_ws, realtime_ws))
        down = asyncio.create_task(_pump_realtime_to_actor(realtime_ws, actor_ws, session_state))
        try:
            done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (up, down):
                if not task.done():
                    task.cancel()
            await asyncio.gather(up, down, return_exceptions=True)


async def _pump_actor_to_realtime(actor_ws: WebSocket, realtime_ws) -> None:
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            await realtime_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
    except WebSocketDisconnect:
        logger.info("[voice] actor disconnected")


async def _pump_realtime_to_actor(realtime_ws, actor_ws: WebSocket, session_state: CallSession) -> None:
    live_item_id = None
    live_item_bytes = 0

    async for raw in realtime_ws:
        event = json.loads(raw)
        event_type = event["type"]

        if event_type == "response.created":
            live_item_id = None
            live_item_bytes = 0

        elif event_type == "response.output_audio.delta":
            if event["item_id"] != live_item_id:
                live_item_id = event["item_id"]
                live_item_bytes = 0
            audio = base64.b64decode(event["delta"])
            await actor_ws.send_bytes(audio)
            live_item_bytes += len(audio)

        elif event_type == "response.output_audio_transcript.done":
            logger.info(f"[voice] rory_said: {event['transcript']}")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            logger.info(f"[voice] caller_said: {event['transcript']}")

        elif event_type == "input_audio_buffer.speech_started":
            if live_item_id is not None:
                await realtime_ws.send(json.dumps({
                    "type": "conversation.item.truncate",
                    "item_id": live_item_id,
                    "content_index": 0,
                    # The voice_ws actor plays the audio forwarded to it.
                    "audio_end_ms": live_item_bytes // PCM16_BYTES_PER_MS,
                }))
                live_item_id = None

        elif event_type == "response.done":
            live_item_id = None
            response = event["response"]
            if response["status"] == "cancelled":
                continue
            if response["status"] != "completed":
                raise RuntimeError(f"ElevenLabs Realtime response failed: {response}")
            # Wait for the complete response rather than dispatching argument
            # fragments from a tool call that can still be cancelled by VAD.
            tool_calls = [item for item in response["output"] if item["type"] == "function_call"]
            for call in tool_calls:
                args = json.loads(call["arguments"])
                result = await asyncio.to_thread(dispatch, session_state, call["name"], args)
                logger.info(f"[voice] tool {call['name']} ({call['call_id']}) -> {json.dumps(result, default=str)}")
                await realtime_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result, default=str),
                    },
                }))
            if tool_calls:
                await realtime_ws.send(json.dumps({"type": "response.create"}))

        elif event_type == "error":
            raise RuntimeError(f"ElevenLabs Realtime error: {event['error']}")
