"""The agent's trace as the bench will fold it: OpenInference LLM spans, one per model round."""

from __future__ import annotations

import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rory_agentcore import trace
from rory_agentcore.contract import CALL_CONNECTED

from rory_tools.prompt import GREETING

TURN = {
    "model": "us.anthropic.claude-sonnet-5-5",
    "system": ["You are Rory.", "Today is Thursday."],
    "messages": [
        {"role": "user", "content": [{"text": "what do I owe"}]},
        {"role": "assistant", "content": [
            {"text": "Let me check."},
            {"toolUse": {"toolUseId": "t1", "name": "get_balance", "input": {"x": 1}}},
        ]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"balance": 42}}]}}]},
        {"role": "assistant", "content": [{"text": "You owe 42 dollars."}]},
    ],
    "rounds": [
        {"start_ns": 1_000, "end_ns": 2_000, "output_index": 1, "input_tokens": 100, "output_tokens": 10},
        {"start_ns": 3_000, "end_ns": 4_000, "output_index": 3, "input_tokens": 150, "output_tokens": 8},
    ],
}


def test_each_model_round_becomes_one_llm_span(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_tracer", provider.get_tracer("test"))
    history = [
        {"role": "user", "content": [{"text": CALL_CONNECTED}]},
        {"role": "assistant", "content": [{"text": GREETING}]},
    ]

    trace.report_turn(history, TURN)

    first, second = exporter.get_finished_spans()
    a = dict(first.attributes)
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.model_name"] == "us.anthropic.claude-sonnet-5-5"
    assert (first.start_time, first.end_time) == (1_000, 2_000)
    assert a["llm.token_count.prompt"] == 100
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.0.message.content"] == "You are Rory.\n\nToday is Thursday."
    assert a["llm.input_messages.2.message.content"] == GREETING
    assert a["llm.input_messages.3.message.content"] == "what do I owe"
    assert "llm.input_messages.4.message.role" not in a
    assert a["llm.output_messages.0.message.content"] == "Let me check."
    assert a["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "t1"
    assert a["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "get_balance"
    assert json.loads(a["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]) == {"x": 1}

    b = dict(second.attributes)
    assert b["llm.input_messages.5.message.role"] == "tool"
    assert b["llm.input_messages.5.message.tool_call_id"] == "t1"
    assert json.loads(b["llm.input_messages.5.message.content"]) == {"balance": 42}
    assert b["llm.output_messages.0.message.content"] == "You owe 42 dollars."

    assert history[-1] == TURN["messages"][-1]  # the turn joined the pod's copy
