"""The tools Lambda: AgentCore Gateway's target, running Rory's tools through the shared dispatcher.

A Lambda is stateless, so who the caller has proved themselves to be lives in a
DynamoDB session record instead of on a ``CallSession`` that lives as long as the
call. Each invocation rebuilds the ``CallSession`` from the record, runs exactly one
tool through ``rory_tools.dispatch``, and writes the result of verification back.
The record also carries the twins this call runs against, which the bench injects
per attempt; the fulfillment Lambda wrote them on the call's first turn.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from rory_tools import CallSession, dispatch
from rory_tools.session import clients

from .contract import SESSION_ARG

_table = boto3.resource("dynamodb").Table(os.environ["SESSIONS_TABLE"])


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    name = context.client_context.custom["bedrockAgentCoreToolName"].split("___", 1)[1]
    args = dict(event)
    session_id = args.pop(SESSION_ARG)
    record = _table.get_item(Key={"session_id": session_id}, ConsistentRead=True)["Item"]

    # A warm Lambda serves calls against different twins, so the vendor clients
    # are rebuilt from this call's record every time.
    os.environ["STRIPE_API_BASE"] = record["stripe_api_base"]
    os.environ["STRIPE_API_KEY"] = record["stripe_api_key"]
    os.environ["FINERACT_API_BASE"] = record["fineract_api_base"]
    if "reference_time" in record:
        os.environ["RORY_REFERENCE_TIME"] = record["reference_time"]
    else:
        os.environ.pop("RORY_REFERENCE_TIME", None)
    clients._stripe = None
    clients._fineract = None

    session = CallSession()
    session.account = json.loads(record["account"]) if "account" in record else None
    session.fineract_client_id = int(record["fineract_client_id"]) if "fineract_client_id" in record else None
    session.failed_attempts = int(record["failed_attempts"])

    result = dispatch(session, name, args)

    values: dict[str, Any] = {":attempts": session.failed_attempts}
    update = "SET failed_attempts = :attempts"
    if session.account is not None:
        update += ", account = :account"
        values[":account"] = json.dumps(session.account, default=str)
    if session.fineract_client_id is not None:
        update += ", fineract_client_id = :client"
        values[":client"] = session.fineract_client_id
    _table.update_item(Key={"session_id": session_id}, UpdateExpression=update, ExpressionAttributeValues=values)
    return result
