"""Web/transport server for rory-polyai.

FastAPI app exposing the Veris ``voice_ws`` transport onto a PolyAI call:
``WS /voice`` accepts raw PCM16/24 kHz mono binary frames (the wire protocol
Veris's ``voice_ws`` actor channel speaks) and hands the socket to
``rory_polyai.agent.run_voice_ws_bot``. ``POST /tool`` is the webhook the
Agent Studio functions call for every tool the model invokes.

Deliberately the same surface as the other candidates' ``web.py`` — same
route, same wire format, same health probe — so the candidates differ only in
which image is booted. The webhook is the one addition, and it is reached
through the tunnel opened at boot. Boot also provisions the Agent Studio
project, which is why ``/health`` takes most of a minute to first answer.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials
from rory_tools.tunnel import public_base_url

from .agent import GATEWAY_URL, run_voice_ws_bot
from .studio import provision
from .tools import run_tool_call

PORT = 8008


@asynccontextmanager
async def _lifespan(app: FastAPI):
    require_credentials(
        "STRIPE_API_KEY", "POLY_ADK_KEY", "POLYAI_AUTH_TOKEN", "POLYAI_ACCOUNT_ID", "POLYAI_PROJECT_ID"
    )
    # Both happen before :8008 answers /health: the readiness probe only
    # passes once the functions have somewhere to POST and the live
    # deployment carries this boot's URL. Push + merge + promote takes about
    # 45 s, and a caller already on the line would hang up waiting for a
    # greeting the old deployment cannot give.
    base_url, tunnel = public_base_url(PORT)
    tool_webhook_url = f"{base_url}/tool"
    await asyncio.to_thread(provision, tool_webhook_url)
    logger.info(f"[web] polyai ready tool_webhook_url={tool_webhook_url} gateway={GATEWAY_URL}")
    try:
        yield
    finally:
        if tunnel is not None:
            tunnel.terminate()


app = FastAPI(title="rory-polyai", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tool")
async def tool_webhook(request: Request) -> JSONResponse:
    """The rendered functions' webhook: one ``{name, args, conversation_id}`` envelope in, the result out.

    Always 200. The function body in PolyAI's runtime raises on a non-2xx and
    the model is left with nothing to say; a refusal from the shared gate is
    already ``{"error": ...}`` content the model relays.
    """
    envelope = await request.json()
    result = await run_tool_call(envelope["conversation_id"], envelope["name"], envelope["args"])
    return JSONResponse(result)


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. The FastAPI WS handler is
    already its own task per connection, and ``run_voice_ws_bot`` blocks until
    the WS closes or the gateway ends the call.
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
