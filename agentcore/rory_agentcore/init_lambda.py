"""The fulfillment Lambda: Lex hands it each transcribed turn, it hands the turn to AgentCore.

This is the "agent initialization" step of the Connect pattern. On a call's first
turn it writes the session record the tools Lambda reads: the twins this call runs
against and a clean verification state. Every turn it invokes the AgentCore Runtime
on the call's session and returns the agent's reply for Lex to send back.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3

from .contract import (
    ATTR_FINERACT_API_BASE,
    ATTR_INTERRUPTED_REPLY,
    ATTR_REFERENCE_TIME,
    ATTR_STRIPE_API_BASE,
    ATTR_STRIPE_API_KEY,
)

# Session records outlive the call by a day, then DynamoDB drops them.
RECORD_TTL_S = 24 * 3600

_table = boto3.resource("dynamodb").Table(os.environ["SESSIONS_TABLE"])
_runtime = boto3.client("bedrock-agentcore")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    session_id = event["sessionId"]
    attrs = dict(event["sessionState"]["sessionAttributes"])
    _init_record(session_id, attrs)

    # Noise the pod took for speech reaches here with nothing transcribed. There is no
    # turn to answer, and Bedrock refuses a blank message; Rory stays silent.
    if not event["inputTranscript"].strip():
        return {"sessionState": {"dialogAction": {"type": "ElicitIntent"}, "sessionAttributes": attrs}}

    payload = {"text": event["inputTranscript"]}
    for key in (ATTR_REFERENCE_TIME, ATTR_INTERRUPTED_REPLY):
        if key in attrs:
            payload[key] = attrs[key]
    resp = _runtime.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENTCORE_RUNTIME_ARN"],
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(payload).encode(),
    )
    answer = json.loads(resp["response"].read())
    # The pod reads the turn back from the record to report the agent's trace. The
    # sequence number tells it which turn it is reading.
    _table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="SET last_turn = :turn, turn_seq = turn_seq + :one",
        ExpressionAttributeValues={":turn": json.dumps(answer["turn"], default=str), ":one": 1},
    )

    # The trim applies once; the next turn must not repeat it.
    attrs.pop(ATTR_INTERRUPTED_REPLY, None)
    return {
        "sessionState": {"dialogAction": {"type": "ElicitIntent"}, "sessionAttributes": attrs},
        "messages": [{"contentType": "PlainText", "content": answer["reply"]}],
    }


def _init_record(session_id: str, attrs: dict[str, str]) -> None:
    """Write the call's twins and a clean verification state, once per call."""
    item = {
        "session_id": session_id,
        "stripe_api_base": attrs[ATTR_STRIPE_API_BASE],
        "stripe_api_key": attrs[ATTR_STRIPE_API_KEY],
        "fineract_api_base": attrs[ATTR_FINERACT_API_BASE],
        "failed_attempts": 0,
        "turn_seq": 0,
        "expires_at": int(time.time()) + RECORD_TTL_S,
    }
    if ATTR_REFERENCE_TIME in attrs:
        item["reference_time"] = attrs[ATTR_REFERENCE_TIME]
    try:
        _table.put_item(Item=item, ConditionExpression="attribute_not_exists(session_id)")
    except _table.meta.client.exceptions.ConditionalCheckFailedException:
        pass  # a later turn of a call already initialized
