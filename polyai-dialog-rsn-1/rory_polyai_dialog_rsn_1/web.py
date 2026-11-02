"""Serve the standard voice_ws audio channel and readiness probe on :8008."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import REALTIME_URL, RSN_VOICE, run_voice_ws_bot


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials("STRIPE_API_KEY", "POLY_API_KEY")
    logger.info(f"[web] polyai-dialog-rsn-1 ready url={REALTIME_URL} voice={RSN_VOICE}")
    yield


app = FastAPI(title="rory-polyai-dialog-rsn-1", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket)
    except Exception:
        logger.exception("[web] Dialog-RSN-1 voice handler failed")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Candidate voice handler failed")
        raise
    else:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
