"""Web/transport server for rory-vapi.

FastAPI app exposing the Veris ``voice_ws`` transport onto a Vapi call:
``WS /voice`` accepts raw PCM16/24 kHz mono binary frames (the wire protocol
Veris's ``voice_ws`` actor channel speaks) and hands the socket to
``rory_vapi.agent.run_voice_ws_bot``. ``POST /tool`` is the server-tool
webhook Vapi calls for every tool the model invokes.

Deliberately the same surface as the other candidates' ``web.py`` — same
route, same wire format, same health probe — so the candidates differ only in
which image is booted. The webhook is the one addition, and it is reached
through the tunnel opened at boot.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from .agent import LIVE_CALLS, run_voice_ws_bot
from .tools import run_tool_calls
from rory_tools.tunnel import public_base_url

PORT = 8008


@asynccontextmanager
async def _lifespan(app: FastAPI):
    require_credentials("STRIPE_API_KEY", "VAPI_API_KEY")
    # The tunnel is opened here, before :8008 answers /health, so the
    # readiness probe only passes once Vapi has somewhere to send tool calls.
    base_url, tunnel = public_base_url(PORT)
    app.state.tool_webhook_url = f"{base_url}/tool"
    logger.info(f"[web] vapi ready tool_webhook_url={app.state.tool_webhook_url}")
    try:
        yield
    finally:
        if tunnel is not None:
            tunnel.terminate()


app = FastAPI(title="rory-vapi", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tool")
async def tool_webhook(request: Request) -> JSONResponse:
    """Vapi's server-tool webhook: one ``tool-calls`` envelope in, ``results`` out.

    The envelope names its call, and only a call this process created has a
    session to resolve against. Anything else is a misrouted webhook and is
    refused rather than answered from a fresh, unverified session.
    """
    message = (await request.json())["message"]
    call_id = message["call"]["id"]
    live = LIVE_CALLS.get(call_id)
    if live is None:
        logger.error(f"[tool] webhook for unknown call {call_id}")
        return JSONResponse({"error": f"unknown call {call_id}"}, status_code=404)
    results = await run_tool_calls(live.session, live.lock, message["toolCallList"])
    return JSONResponse({"results": results})


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. The FastAPI WS handler is
    already its own task per connection, and ``run_voice_ws_bot`` blocks until
    the WS closes or the Vapi call ends.
    """
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info(f"[web] /voice connection peer={peer}")
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket, websocket.app.state.tool_webhook_url)
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
