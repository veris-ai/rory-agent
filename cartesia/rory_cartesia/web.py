"""Web/transport server for rory-cartesia.

FastAPI app exposing the Veris ``voice_ws`` transport onto a Cartesia managed
agent: ``WS /voice`` accepts raw PCM16/24 kHz mono binary frames and hands the
socket to ``rory_cartesia.agent.run_voice_ws_bot``. ``POST /tool/<tool name>``
is the webhook Cartesia calls for every tool the model invokes.

Deliberately the same surface as the other candidates' ``web.py`` — same route,
same wire format, same health probe. The webhook is the one addition, reached
through the attempt's public endpoint (``PUBLIC_BASE_URL``).

The pod's 16 webhook tools are created in the lifespan, once, and deleted at
shutdown. A pod that cannot create them never comes up, so the bench sees a
failed health check instead of a candidate that answers calls without tools.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from loguru import logger
from starlette.websockets import WebSocketState

from rory_tools.session import require_credentials

from . import agent
from .agent import CARTESIA_MODEL, CARTESIA_VOICE_ID, cartesia_api, create_tools, delete_tools, run_tool_call, run_voice_ws_bot


def public_base_url() -> str:
    """The attempt's public endpoint, which Cartesia's webhooks are addressed to."""
    base = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base.startswith("https://"):
        raise RuntimeError(
            f"PUBLIC_BASE_URL must be an absolute https:// URL, got {base!r}"
        )
    return base


@asynccontextmanager
async def _lifespan(app: FastAPI):
    require_credentials("STRIPE_API_KEY", "CARTESIA_API_KEY")
    base = public_base_url()
    secret = secrets.token_urlsafe(32)
    async with cartesia_api() as api:
        tool_ids = await create_tools(api, f"{base}/tool", secret)
        app.state.api = api
        app.state.tool_ids = tool_ids
        app.state.secret = secret
        logger.info(
            f"[web] cartesia ready model={CARTESIA_MODEL} voice={CARTESIA_VOICE_ID} "
            f"tools={len(tool_ids)} public_base_url={base}"
        )
        try:
            yield
        finally:
            # The server drains its connections before it gets here, and every
            # call deletes its agent on the way out, so nothing references the
            # tools any more.
            await delete_tools(api, tool_ids)


app = FastAPI(title="rory-cartesia", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tool/{tool_name}")
async def tool_webhook(tool_name: str, request: Request) -> Response:
    """Cartesia's webhook: the model's arguments in, the tool result out.

    The bearer secret proves the request came from the tools this pod created,
    and the pod's live call is the session it runs against. Anything else is
    refused rather than answered from a fresh, unverified session.
    """
    if not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {request.app.state.secret}"):
        logger.warning(f"[tool] rejected {tool_name}: bad bearer token")
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    live = agent.LIVE_CALL
    if live is None:
        logger.error(f"[tool] webhook {tool_name} with no live call")
        return JSONResponse({"error": "no live call"}, status_code=404)
    # A tool with no arguments can arrive with an empty body.
    args = json.loads(await request.body() or b"{}")
    result = json.dumps(await run_tool_call(live, tool_name, args), default=str)
    logger.info(f"[tool] {tool_name} -> {result}")
    return Response(result, media_type="application/json")


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Veris ``voice_ws`` actor channel.

    Raw PCM16/24 kHz mono binary in both directions. ``run_voice_ws_bot``
    blocks until the WS closes or the Cartesia session ends.
    """
    peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"
    logger.info(f"[web] /voice connection peer={peer}")
    await websocket.accept()
    try:
        await run_voice_ws_bot(websocket, websocket.app.state.api, websocket.app.state.tool_ids)
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
