"""Serve the standard voice_ws audio channel and readiness probe on :8008."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import REALTIME_MODEL, REALTIME_VOICE, run_voice_ws_bot


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials("STRIPE_API_KEY", "ELEVENLABS_API_KEY")
    logger.info(f"[web] elevenlabs-realtime ready model={REALTIME_MODEL} voice={REALTIME_VOICE}")
    yield


app = FastAPI(title="rory-elevenlabs-realtime", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket)
    except Exception:
        logger.exception("[web] ElevenLabs Realtime voice handler failed")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Candidate voice handler failed")
        raise
    else:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
