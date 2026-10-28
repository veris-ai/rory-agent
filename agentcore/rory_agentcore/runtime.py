"""The AgentCore Runtime agent: Rory's reasoning loop on Bedrock, tools through the gateway.

AgentCore Runtime runs this app in its own microVM per session and calls
``POST /invocations`` once per caller turn, with the session in the
``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id`` header. The history for that
session lives in this process.

One turn: the caller's words go into the history, Bedrock Converse answers, and
any tool calls it makes go out through AgentCore Gateway (MCP) to the tools Lambda,
one at a time, until the model replies in words. The reply goes back to the Lex
fulfillment Lambda, which hands it to Lex, which hands it to the pod.

The model sees the shared prompt, the shared greeting and the shared tool
declaration. The session id is added to each tool call after the model has
chosen it, so the model never supplies identity.
"""

from __future__ import annotations

import os
import time
from typing import Any

import boto3
from fastapi import FastAPI, Header
from loguru import logger

from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from . import gateway
from .contract import (
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    CALL_CONNECTED,
    SESSION_ARG,
)
from .tools import CONVERSE_TOOLS

AGENT_PROMPT = load_agent_prompt()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-5-5")

# A turn that still wants tools after this many model rounds is looping.
MAX_TOOL_ROUNDS = 5

_bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

# AgentCore gives each session its own microVM, so this holds one call, but keying
# by session keeps a mixed-up routing visible instead of blending two callers.
_sessions: dict[str, dict[str, Any]] = {}

app = FastAPI(title="rory-agentcore-runtime")


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "Healthy"}


@app.post("/invocations")
def invocations(
    body: dict,
    x_amzn_bedrock_agentcore_runtime_session_id: str = Header(),
) -> dict[str, Any]:
    """The reply, and the turn's model exchange for the pod to report as the agent's trace."""
    session_id = x_amzn_bedrock_agentcore_runtime_session_id
    if session_id not in _sessions:
        _sessions[session_id] = _new_session(body.get(ATTR_REFERENCE_TIME))
    session = _sessions[session_id]
    if ATTR_INTERRUPTED_REPLY in body:
        _replace_last_reply(session["messages"], body[ATTR_INTERRUPTED_REPLY])
    reply, turn = take_turn(session, session_id, body["text"])
    return {"reply": reply, "turn": turn}


def _new_session(reference_time: str | None) -> dict[str, Any]:
    """System prompt, date context, and the greeting the pod already spoke."""
    # The date context reads the frozen clock the bench restored with the twins'
    # snapshot. Unset in production, where it is the wall clock.
    if reference_time:
        os.environ["RORY_REFERENCE_TIME"] = reference_time
    return {
        "system": [{"text": AGENT_PROMPT}, {"text": today_context()}],
        "messages": [
            {"role": "user", "content": [{"text": CALL_CONNECTED}]},
            {"role": "assistant", "content": [{"text": GREETING}]},
        ],
    }


def _replace_last_reply(messages: list[dict], heard: str) -> None:
    """The caller talked over the last reply: keep only what they heard."""
    last = messages[-1]
    assert last["role"] == "assistant", "a barge-in can only trim a spoken reply"
    last["content"] = [{"text": heard}]


def take_turn(session: dict[str, Any], session_id: str, text: str) -> tuple[str, dict[str, Any]]:
    """One caller turn: the reply, and the turn as the model saw it.

    The turn joins the history only once it has a reply, so a failed turn cannot
    leave a half-finished exchange that every later Converse call rejects.

    The returned turn is what the bench needs to rebuild the agent's trace: the
    system prompt, the turn's messages, and one entry per model round with its
    timing and token usage. The pod already holds the history before this turn.
    """
    turn = [{"role": "user", "content": [{"text": text}]}]
    rounds = []
    for _ in range(MAX_TOOL_ROUNDS):
        start_ns = time.time_ns()
        resp = _bedrock.converse(
            modelId=MODEL_ID,
            system=session["system"],
            messages=session["messages"] + turn,
            toolConfig={"tools": CONVERSE_TOOLS},
        )
        message = resp["output"]["message"]
        turn.append(message)
        rounds.append({
            "start_ns": start_ns,
            "end_ns": time.time_ns(),
            "output_index": len(turn) - 1,
            "input_tokens": resp["usage"]["inputTokens"],
            "output_tokens": resp["usage"]["outputTokens"],
        })
        if resp["stopReason"] != "tool_use":
            break
        # One at a time, in the order emitted: calls share verification and
        # payment state.
        results = []
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            use = block["toolUse"]
            result = gateway.call_tool(use["name"], {**use["input"], SESSION_ARG: session_id})
            logger.info(f"[tool] {use['name']} -> {result}")
            results.append({"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}]}})
        turn.append({"role": "user", "content": results})
    else:
        raise RuntimeError(f"tool loop did not settle in {MAX_TOOL_ROUNDS} rounds")
    session["messages"].extend(turn)
    reply = " ".join(block["text"] for block in message["content"] if "text" in block).strip()
    logger.info(f"[llm] rory_said: {reply}")
    return reply, {
        "model": MODEL_ID,
        "system": [block["text"] for block in session["system"]],
        "messages": turn,
        "rounds": rounds,
    }
