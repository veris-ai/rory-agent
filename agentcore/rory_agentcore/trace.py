"""The agent's trace, reported from the pod.

The bench reads a candidate's model calls from OpenInference LLM spans exported
from inside the candidate's container, and folds them into the conversation as the
model saw it: tool calls, tool results and all. Rory's model runs in AgentCore,
out of the bench's reach, so the runtime hands back each turn's exchange (through
the fulfillment Lambda and the call's DynamoDB record) and the pod exports one LLM
span per model round: the history so far as input, the round's message as output.

The spans go through the OpenTelemetry API to whatever provider the process has.
On the bench that is the platform's exporter; elsewhere the API is a no-op.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace as otel

_tracer = otel.get_tracer("rory-agentcore")


def _openinference(messages: list[dict]) -> list[dict[str, Any]]:
    """Converse messages as OpenInference messages: one per tool result, as in the chat format."""
    out = []
    for message in messages:
        texts = [block["text"] for block in message["content"] if "text" in block]
        uses = [block["toolUse"] for block in message["content"] if "toolUse" in block]
        results = [block["toolResult"] for block in message["content"] if "toolResult" in block]
        if texts or uses:
            entry: dict[str, Any] = {"role": message["role"], "content": "\n".join(texts)}
            if uses:
                entry["tool_calls"] = [
                    {"id": use["toolUseId"], "name": use["name"], "arguments": json.dumps(use["input"])}
                    for use in uses
                ]
            out.append(entry)
        for result in results:
            out.append({
                "role": "tool",
                "content": json.dumps(result["content"][0]["json"]),
                "tool_call_id": result["toolUseId"],
            })
    return out


def _flatten(prefix: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for i, message in enumerate(messages):
        key = f"{prefix}.{i}.message"
        attrs[f"{key}.role"] = message["role"]
        if message.get("content"):
            attrs[f"{key}.content"] = message["content"]
        if "tool_call_id" in message:
            attrs[f"{key}.tool_call_id"] = message["tool_call_id"]
        for j, call in enumerate(message.get("tool_calls", [])):
            call_key = f"{key}.tool_calls.{j}.tool_call"
            attrs[f"{call_key}.id"] = call["id"]
            attrs[f"{call_key}.function.name"] = call["name"]
            attrs[f"{call_key}.function.arguments"] = call["arguments"]
    return attrs


def report_turn(history: list[dict], turn: dict[str, Any]) -> None:
    """Export one LLM span per model round of ``turn``, then add the turn to ``history``.

    ``history`` is the Converse history before this turn, kept by the pod in step
    with the runtime's.
    """
    system = [{"role": "system", "content": "\n\n".join(turn["system"])}]
    messages = turn["messages"]
    for round_ in turn["rounds"]:
        index = round_["output_index"]
        attrs = {
            "openinference.span.kind": "LLM",
            "llm.model_name": turn["model"],
            "llm.token_count.prompt": round_["input_tokens"],
            "llm.token_count.completion": round_["output_tokens"],
            **_flatten("llm.input_messages", system + _openinference(history + messages[:index])),
            **_flatten("llm.output_messages", _openinference([messages[index]])),
        }
        span = _tracer.start_span("bedrock.converse", start_time=round_["start_ns"], attributes=attrs)
        span.end(end_time=round_["end_ns"])
    history.extend(messages)
