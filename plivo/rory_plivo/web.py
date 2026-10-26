"""Web/transport server for rory-plivo.

One process carries both ends of a real Plivo call:

- ``WS /voice`` — the Veris ``voice_ws`` actor channel (raw PCM16/24 kHz).
  ``rory_plivo.caller`` places the actor's audio as a WebRTC call from this
  pod's Plivo endpoint.
- ``POST /plivo/answer`` — Plivo fetches this when that call reaches the
  endpoint's Application, and gets back a bidirectional ``<Stream>``.
- ``WS /plivo/stream`` — Plivo Audio Streaming for the call; Rory's Pipecat
  pipeline (``rory_plivo.agent``) runs on it.

Plivo must reach the last two from the internet, so boot opens the public URL
(``PUBLIC_BASE_URL`` or a cloudflared quick tunnel, ``rory_tools.tunnel``) and
provisions this pod's own Application + Endpoint before ``/health`` answers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials
from rory_tools.tunnel import public_base_url

from .agent import run_stream_bot
from .caller import run_caller_leg
from .plivo_api import provision, teardown

PORT = 8008
STATE: dict = {}


def answer_xml(stream_url: str) -> str:
    """Stream the whole call, both directions, as 8 kHz μ-law (Pipecat's Plivo serializer format)."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
        f"{stream_url}</Stream></Response>"
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials(
        "STRIPE_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY",
        "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN",
    )
    base_url, tunnel = public_base_url(PORT)
    STATE["stream_url"] = base_url.replace("https://", "wss://", 1) + "/plivo/stream"
    STATE["endpoint"] = await provision(f"{base_url}/plivo/answer")
    try:
        yield
    finally:
        await teardown(STATE["endpoint"])
        if tunnel is not None:
            tunnel.terminate()


app = FastAPI(title="rory-plivo", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plivo/answer")
async def plivo_answer(request: Request) -> Response:
    form = await request.body()
    logger.info(f"[web] /plivo/answer {form.decode()[:300]}")
    return Response(answer_xml(STATE["stream_url"]), media_type="application/xml")


@app.websocket("/plivo/stream")
async def plivo_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("[web] /plivo/stream connected")
    try:
        await run_stream_bot(websocket)
    finally:
        logger.info("[web] /plivo/stream closed")


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel: raw PCM16/24 kHz mono binary both ways."""
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info(f"[web] /voice connection peer={peer}")
    await websocket.accept()
    try:
        await run_caller_leg(websocket, STATE["endpoint"])
    except Exception:
        logger.exception(f"[web] /voice failed peer={peer}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Candidate voice handler failed")
        raise
    else:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
    finally:
        logger.info(f"[web] /voice connection closed peer={peer}")
