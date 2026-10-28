"""The init Lambda: sets up a call's agent while the greeting plays.

This is the "agent initialization" step of the Connect pattern, run once per call,
when the call connects. It writes the session record the tools Lambda reads (the
twins this call runs against and a clean verification state), then invokes the
AgentCore Runtime on the call's session, which boots the session's microVM and
primes Bedrock's prompt cache. The caller's first turn pays for none of it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3

from .contract import (
    ATTR_FINERACT_API_BASE,
    ATTR_REFERENCE_TIME,
    ATTR_STRIPE_API_BASE,
    ATTR_STRIPE_API_KEY,
)

# Session records outlive the call by a day, then DynamoDB drops them.
RECORD_TTL_S = 24 * 3600

_table = boto3.resource("dynamodb").Table(os.environ["SESSIONS_TABLE"])
_runtime = boto3.client("bedrock-agentcore")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    session_id = event["session_id"]
    item = {
        "session_id": session_id,
        "stripe_api_base": event[ATTR_STRIPE_API_BASE],
        "stripe_api_key": event[ATTR_STRIPE_API_KEY],
        "fineract_api_base": event[ATTR_FINERACT_API_BASE],
        "failed_attempts": 0,
        "expires_at": int(time.time()) + RECORD_TTL_S,
    }
    payload: dict[str, Any] = {"init": True}
    if ATTR_REFERENCE_TIME in event:
        item["reference_time"] = event[ATTR_REFERENCE_TIME]
        payload[ATTR_REFERENCE_TIME] = event[ATTR_REFERENCE_TIME]
    _table.put_item(Item=item, ConditionExpression="attribute_not_exists(session_id)")
    resp = _runtime.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENTCORE_RUNTIME_ARN"],
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(payload).encode(),
    )
    return json.loads(resp["response"].read())
