"""Web/transport server for rory-fluxions.

FastAPI app exposing the Veris ``voice_ws`` transport onto a Fluxions
realtime session: ``WS /voice`` accepts raw PCM16/24 kHz mono binary frames
(the wire protocol Veris's ``voice_ws`` actor channel speaks) and hands the
socket to ``rory_fluxions.agent.run_voice_ws_bot``.

Deliberately the same surface as the other candidates' ``web.py`` — same
route, same wire format, same health probe — so the candidates differ only in
the voice stack behind it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import run_voice_ws_bot, vendors


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # FLUXIONS_API_KEY is required only by a host whose /config says so; see
    # agent.vendors, which checks it there.
    require_credentials("STRIPE_API_KEY")
    async with vendors():
        logger.info("[web] fluxions ready")
        yield


app = FastAPI(title="rory-fluxions", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. The FastAPI WS handler is
    already its own task per connection, and ``run_voice_ws_bot`` blocks until
    the WS closes or the pipeline fails.
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
