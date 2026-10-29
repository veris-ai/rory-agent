"""The AgentCore Runtime agent: Rory's reasoning loop on Bedrock, tools through the gateway.

AgentCore Runtime runs this app in its own microVM per session and calls
``POST /invocations`` with the session in the ``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id``
header. The history for that session lives in this process.

Two kinds of invocation:

- ``{"init": true}``, once per call from the init Lambda while the greeting plays:
  boots this session's microVM, seeds the history, and primes Bedrock's prompt
  cache, so the caller's first turn pays for none of it.
- ``{"text": ...}``, once per caller turn from the pod: the caller's words go into
  the history, Bedrock answers with ConverseStream, and any tool calls go out
  through AgentCore Gateway (MCP) to the tools Lambda, one at a time, until the
  model replies in words. The reply streams back as server-sent events while it
  is generated, so the pod can speak the first sentence before the last is written.

The model sees the shared prompt, the shared greeting and the shared tool
declaration, as every other candidate's model does. A demo can add one line, off
on the bench (see ``TOOL_PREAMBLE``). The session id is added to each
tool call after the model has chosen it, so the model never supplies identity.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import boto3
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from loguru import logger

from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from . import gateway
from .contract import (
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    ATTR_TOOL_PREAMBLE,
    CALL_CONNECTED,
    SESSION_ARG,
)
from .tools import CONVERSE_TOOLS

AGENT_PROMPT = load_agent_prompt()
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-5-5")

# Demo only. Tools run 2 to 5 s against the bench's twins, and the model usually calls
# one without a word first, so the caller hears silence. This line asks it to say what
# it is doing. The benchmark gives every candidate the same prompt, so it stays off
# there; a demo pod turns it on with RORY_TOOL_PREAMBLE=1.
TOOL_PREAMBLE = (
    "This is a live phone call: the caller hears silence while a tool runs. Before "
    "every tool call, including one that follows another tool's result, first say one "
    'short sentence telling the caller what you are doing, such as "Let me pull up '
    'your account."'
)

# Everything up to here is the same on every request of a call, and most of it on
# every call: tools, prompt, date. Bedrock caches it.
CACHE_POINT = {"cachePoint": {"type": "default"}}

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
):
    session_id = x_amzn_bedrock_agentcore_runtime_session_id
    if body.get("init"):
        _sessions[session_id] = _new_session(body.get(ATTR_REFERENCE_TIME), body[ATTR_TOOL_PREAMBLE])
        _prime_cache(_sessions[session_id])
        return {"ready": True}
    session = _sessions[session_id]
    if ATTR_INTERRUPTED_REPLY in body:
        _replace_last_reply(session["messages"], body[ATTR_INTERRUPTED_REPLY])
    events = (f"data: {json.dumps(event, default=str)}\n\n" for event in take_turn(session, session_id, body["text"]))
    return StreamingResponse(events, media_type="text/event-stream")


def _new_session(reference_time: str | None, tool_preamble: bool) -> dict[str, Any]:
    """System prompt, date context (and the demo line, if asked), and the greeting the pod speaks."""
    # The date context reads the frozen clock the bench restored with the twins'
    # snapshot. Unset in production, where it is the wall clock.
    if reference_time:
        os.environ["RORY_REFERENCE_TIME"] = reference_time
    system = [{"text": AGENT_PROMPT}, {"text": today_context()}]
    if tool_preamble:
        system.append({"text": TOOL_PREAMBLE})
    return {
        "system": system,
        "messages": [
            {"role": "user", "content": [{"text": CALL_CONNECTED}]},
            {"role": "assistant", "content": [{"text": GREETING}]},
        ],
    }


def _prime_cache(session: dict[str, Any]) -> None:
    """One token's worth of request, so the cached prefix is warm before the caller speaks."""
    _bedrock.converse(
        modelId=MODEL_ID,
        system=[*session["system"], CACHE_POINT],
        messages=session["messages"][:1],
        toolConfig={"tools": CONVERSE_TOOLS},
        inferenceConfig={"maxTokens": 1},
    )


def _replace_last_reply(messages: list[dict], heard: str) -> None:
    """The caller talked over the last reply: keep only what they heard."""
    last = messages[-1]
    assert last["role"] == "assistant", "a barge-in can only trim a spoken reply"
    last["content"] = [{"text": heard}]


def _with_cache_point(messages: list[dict]) -> list[dict]:
    """The request's messages with a cache point after the last one: the history so far
    is the next request's prefix."""
    last = messages[-1]
    return [*messages[:-1], {**last, "content": [*last["content"], CACHE_POINT]}]


def take_turn(session: dict[str, Any], session_id: str, text: str) -> Iterator[dict[str, Any]]:
    """One caller turn, as events: ``round`` when a model round starts, ``text`` as
    words arrive, ``tool`` when the model calls one, and a final ``turn``.

    The final event carries the turn as the model saw it, which the pod reports as
    the agent's trace: the system prompt, the turn's messages, and one entry per
    model round with its timing and token usage. The pod already holds the history
    before this turn.

    The turn joins the history only once it has a reply, so a failed turn cannot
    leave a half-finished exchange that every later Converse call rejects.
    """
    turn = [{"role": "user", "content": [{"text": text}]}]
    rounds = []
    for index in range(MAX_TOOL_ROUNDS):
        yield {"type": "round", "index": index}
        start_ns = time.time_ns()
        resp = _bedrock.converse_stream(
            modelId=MODEL_ID,
            system=[*session["system"], CACHE_POINT],
            messages=_with_cache_point(session["messages"] + turn),
            toolConfig={"tools": CONVERSE_TOOLS},
        )
        blocks: dict[int, dict[str, Any]] = {}
        for event in resp["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]["start"]
                if "toolUse" in start:
                    use = start["toolUse"]
                    blocks[event["contentBlockStart"]["contentBlockIndex"]] = {
                        "toolUse": {"toolUseId": use["toolUseId"], "name": use["name"], "input": ""}
                    }
                    yield {"type": "tool", "name": use["name"]}
            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                block_index = event["contentBlockDelta"]["contentBlockIndex"]
                if "text" in delta:
                    blocks.setdefault(block_index, {"text": ""})["text"] += delta["text"]
                    yield {"type": "text", "text": delta["text"]}
                elif "toolUse" in delta:
                    blocks[block_index]["toolUse"]["input"] += delta["toolUse"]["input"]
            elif "messageStop" in event:
                stop_reason = event["messageStop"]["stopReason"]
            elif "metadata" in event:
                usage = event["metadata"]["usage"]
        content = [blocks[i] for i in sorted(blocks)]
        for block in content:
            if "toolUse" in block:
                block["toolUse"]["input"] = json.loads(block["toolUse"]["input"] or "{}")
        message = {"role": "assistant", "content": content}
        turn.append(message)
        rounds.append({
            "start_ns": start_ns,
            "end_ns": time.time_ns(),
            "output_index": len(turn) - 1,
            "input_tokens": usage["inputTokens"] + usage.get("cacheReadInputTokens", 0)
            + usage.get("cacheWriteInputTokens", 0),
            "output_tokens": usage["outputTokens"],
        })
        if stop_reason != "tool_use":
            break
        # One at a time, in the order emitted: calls share verification and
        # payment state.
        results = []
        for block in content:
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
    yield {
        "type": "turn",
        "turn": {
            "model": MODEL_ID,
            "system": [block["text"] for block in session["system"]],
            "messages": turn,
            "rounds": rounds,
        },
    }
