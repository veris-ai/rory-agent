"""The benchmark's health probe and PCM16/24 kHz mono voice_ws endpoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import run_voice_ws_bot, validate_config


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    require_credentials("STRIPE_API_KEY", "DIALT_API_KEY")
    validate_config()
    yield


app = FastAPI(title="rory-dialt", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket)
    except Exception:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Candidate voice handler failed")
        raise
    else:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
