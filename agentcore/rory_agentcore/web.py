"""Web/transport server for rory-agentcore.

FastAPI app exposing the Veris ``voice_ws`` transport onto Amazon Connect's
voice stack: ``WS /voice`` accepts raw PCM16/24 kHz mono binary frames and hands
the socket to ``rory_agentcore.agent.run_voice_ws_bot``. Same route, wire format
and health probe as the other candidates.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import resolve_vendor, run_voice_ws_bot


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials(
        "STRIPE_API_KEY",
        "AWS_REGION",
        "LEX_BOT_ID",
        "LEX_BOT_ALIAS_ID",
        "AGENTCORE_RUNTIME_ARN",
    )
    resolve_vendor()
    yield


app = FastAPI(title="rory-agentcore", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel: raw PCM16/24 kHz mono in both directions."""
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
