"""Web/transport server for rory-gradium.

FastAPI app exposing the Veris ``voice_ws`` transport onto a gradbot session:
``WS /voice`` accepts raw PCM16/24 kHz mono binary frames and hands the socket
to ``rory_gradium.agent.run_voice_ws_bot``.

Deliberately the same surface as the other candidates' ``web.py`` — same route,
same wire format, same health probe — so the candidates differ only in which
image is booted.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import GRADIUM_LLM_MODEL, GRADIUM_VOICE_ID, run_voice_ws_bot


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials("STRIPE_API_KEY", "GRADIUM_API_KEY", "OPENAI_API_KEY")
    logger.info(f"[web] gradium ready voice={GRADIUM_VOICE_ID} model={GRADIUM_LLM_MODEL}")
    yield


app = FastAPI(title="rory-gradium", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. ``run_voice_ws_bot``
    blocks until the WS closes or the gradbot session ends.
    """
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
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
