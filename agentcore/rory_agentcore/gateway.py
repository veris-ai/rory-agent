"""A minimal MCP client for AgentCore Gateway: ``tools/call`` over SigV4-signed HTTP.

The runtime already knows the tool list (``tools.CONVERSE_TOOLS`` is the shared
declaration), so the gateway is used for one thing only: invoking a tool. The
gateway authorizes with AWS IAM, so each request is signed with the runtime's own
role.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from .contract import GATEWAY_TARGET

_session = boto3.Session(region_name=os.environ["AWS_REGION"])


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one Rory tool through the gateway and return its result object."""
    url = os.environ["GATEWAY_URL"]
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": f"{GATEWAY_TARGET}___{name}", "arguments": arguments},
    })
    request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    SigV4Auth(_session.get_credentials(), "bedrock-agentcore", _session.region_name).add_auth(request)
    with urllib.request.urlopen(
        urllib.request.Request(url, data=body.encode(), headers=dict(request.headers), method="POST"),
        timeout=60,
    ) as resp:
        raw = resp.read().decode()
        content_type = resp.headers.get("Content-Type", "")
    message = _parse(raw, content_type)
    if "error" in message:
        raise RuntimeError(f"gateway refused {name}: {message['error']}")
    result = message["result"]
    if result.get("isError"):
        raise RuntimeError(f"gateway tool {name} failed: {result['content']}")
    return json.loads(result["content"][0]["text"])


def _parse(raw: str, content_type: str) -> dict[str, Any]:
    """A JSON-RPC response, sent either as plain JSON or as one server-sent event."""
    if content_type.startswith("text/event-stream"):
        data = [line[len("data:"):].strip() for line in raw.splitlines() if line.startswith("data:")]
        return json.loads(data[-1])
    return json.loads(raw)
