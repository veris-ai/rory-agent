"""Web/transport server for rory-elevenlabs.

FastAPI app exposing the Veris ``voice_ws`` transport onto an ElevenLabs
conversation: ``WS /voice`` accepts raw PCM16/24 kHz mono binary frames (the
wire protocol Veris's ``voice_ws`` actor channel speaks) and hands the socket
to ``rory_elevenlabs.agent.run_voice_ws_bot``.

Deliberately the same surface as the other candidates' ``web.py`` — same
route, same wire format, same health probe — so the candidates differ only in
which image is booted.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from elevenlabs import ElevenLabs
from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import ensure_agent, run_voice_ws_bot


@asynccontextmanager
async def _lifespan(app: FastAPI):
    require_credentials("STRIPE_API_KEY", "ELEVENLABS_API_KEY")
    # The stored agent is provisioned here, before :8008 answers /health, so
    # the readiness probe only passes once there is an agent to talk to.
    app.state.eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    app.state.agent_id = ensure_agent(app.state.eleven)
    logger.info(f"[web] elevenlabs ready agent_id={app.state.agent_id}")
    yield


app = FastAPI(title="rory-elevenlabs", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. The FastAPI WS handler is
    already its own task per connection, and ``run_voice_ws_bot`` blocks until
    the WS closes or the ElevenLabs conversation ends.
    """
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info(f"[web] /voice connection peer={peer}")
    await websocket.accept()
    try:
        state = websocket.app.state
        await run_voice_ws_bot(websocket, state.eleven, state.agent_id)
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
