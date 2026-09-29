"""Web/transport server for rory-pipecat.

FastAPI app exposing the Veris ``voice_ws`` transport onto the Pipecat
pipeline: ``WS /voice`` accepts raw PCM16/24 kHz mono binary frames (the wire
protocol Veris's ``voice_ws`` actor channel speaks). The handler hands the
WebSocket to ``rory_pipecat.agent.run_voice_ws_bot`` which wraps it in a
``FastAPIWebsocketTransport`` + ``RawPCM16Serializer`` and runs the pipeline.

No SFU, no WebRTC — audio terminates inside this process.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import run_voice_ws_bot

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials("STRIPE_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")
    yield


app = FastAPI(title="rory-pipecat", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. The FastAPI WS handler
    is already its own task per connection, and ``run_voice_ws_bot`` blocks
    until the WS closes or the runner exits.
    """
    peer = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "?"
    )
    logger.info(f"[web] /voice connection peer={peer}")
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket)
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
