"""Hugging Face's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, is what lets this transport ship its own image and its own
lockfile without needing any other voice SDK resolvable alongside it.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from rory_tools import SCHEMAS
from rory_tools.parity import (
    assert_argument_schemas_match,
    assert_description_matches,
    assert_required_arguments_match,
    assert_shares_prompt_and_greeting,
    assert_tool_names_match,
    assert_uses_shared_dispatcher,
)

from rory_huggingface import agent, tools

SURFACE = tools.tool_surface()


def test_tool_names_match_the_shared_declaration():
    assert_tool_names_match(SURFACE)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_description_reaches_the_model_unedited(schema):
    assert_description_matches(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_required_arguments_match(schema):
    assert_required_arguments_match(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_argument_schemas_match(schema):
    assert_argument_schemas_match(SURFACE, schema)


def test_prompt_and_greeting_are_the_shared_ones():
    assert_shares_prompt_and_greeting(agent)


def test_tool_calls_go_through_the_shared_gate():
    # The cascade dispatches from the agent module rather than the adapter,
    # because its tool calls arrive in the chat completion it drives itself
    # rather than as framework callbacks.
    assert_uses_shared_dispatcher(agent)


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_huggingface.web import app

    saved = os.environ.pop("HF_TOKEN", None)
    try:
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["HF_TOKEN"] = saved


def test_model_leg_goes_through_the_openai_sdk(monkeypatch):
    """OpenTelemetry instrumentation of the openai SDK only sees calls made through it.

    A hand-built POST to /v1/chat/completions is invisible to that trace, so
    the turn loop must go through the SDK client, replay only portable fields,
    and feed tool results back as tool messages.
    """
    import asyncio
    import types

    from openai.types.chat import ChatCompletion

    def completion(message: dict) -> ChatCompletion:
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": agent.LLM_MODEL,
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            }
        )

    calls: list[dict] = []
    replies = [
        completion(
            {
                "role": "assistant",
                "content": None,
                "reasoning": "gpt-oss on Groq adds this; it must not be replayed",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "verify_caller", "arguments": "{}"},
                    }
                ],
            }
        ),
        completion({"role": "assistant", "content": "You're verified."}),
    ]

    class _Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return replies[len(calls) - 1]

    # _llm is only annotated at import; connect_legs binds it at boot.
    monkeypatch.setattr(
        agent,
        "_llm",
        types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions())),
        raising=False,
    )
    monkeypatch.delenv("HF_LLM_URL", raising=False)

    call = agent._Call(actor_ws=None)
    spoken: list[str] = []
    ran: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    async def run_tool(tool_call: dict) -> dict:
        ran.append(tool_call["function"]["name"])
        return {
            "role": "tool",
            "name": tool_call["function"]["name"],
            "tool_call_id": tool_call["id"],
            "content": '{"verified": true}',
        }

    monkeypatch.setattr(call, "_speak", speak)
    monkeypatch.setattr(call, "_run_tool", run_tool)

    asyncio.run(call._take_turn("seven three one zero dash two one one zero four"))

    assert [c["model"] for c in calls] == [agent.LLM_MODEL] * 2
    assert all(c["tools"] is agent.TOOLS and c["tool_choice"] == "auto" for c in calls)
    assert all(c["extra_body"] is None for c in calls), "thinking flag only behind HF_LLM_URL"
    assert ran == ["verify_caller"]
    assert spoken == ["You're verified."]

    roles = [m["role"] for m in call.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    replayed = call.messages[2]
    assert set(replayed) == {"role", "tool_calls"}, "only portable fields go back to the model"
    assert replayed["tool_calls"][0]["function"]["name"] == "verify_caller"
    assert calls[1]["messages"] is call.messages
