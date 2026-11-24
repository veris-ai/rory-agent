"""Grok voice's own parity run.

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
from rory_tools.prompt import GREETING, load_agent_prompt
from rory_tools.parity import (
    assert_argument_schemas_match,
    assert_description_matches,
    assert_required_arguments_match,
    assert_tool_names_match,
    assert_uses_shared_dispatcher,
)

from rory_grok_voice import agent, tools

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


def test_prompt_differs_from_the_shared_one_only_in_the_account_format():
    shared = load_agent_prompt()
    assert agent.AGENT_PROMPT != shared
    assert agent.AGENT_PROMPT == shared.replace(agent.SHARED_ACCOUNT_FORMAT, agent.GROK_ACCOUNT_FORMAT)


def test_greeting_is_the_shared_one():
    assert agent.GREETING is GREETING


def test_tool_calls_go_through_the_shared_gate():
    # Grok dispatches from the agent module rather than the adapter, because
    # its function calls arrive as events on the vendor websocket rather than as
    # framework callbacks — the same shape as the Realtime transport.
    assert_uses_shared_dispatcher(agent)


def test_session_keeps_grok_shape():
    """The two wire deltas from OpenAI Realtime, pinned.

    Voice, instructions and turn detection sit at the session's top level (not
    under ``audio``), and input transcription names ``grok-transcribe`` — without
    it Grok emits no caller transcripts at all.
    """
    session = agent._session_update()["session"]
    assert session["voice"] == agent.GROK_VOICE
    assert session["instructions"].startswith(agent.AGENT_PROMPT)
    assert session["turn_detection"] == {"type": "server_vad"}
    assert session["audio"]["input"]["transcription"] == {"model": "grok-transcribe"}
    assert session["audio"]["input"]["format"]["rate"] == agent.SAMPLE_RATE_HZ
    assert session["audio"]["output"]["format"]["rate"] == agent.SAMPLE_RATE_HZ
    assert session["tools"] is tools.TOOLS


def test_greeting_is_a_force_message_with_the_shared_text():
    item = agent._greeting()["item"]
    assert item["type"] == "force_message"
    assert item["role"] == "assistant"
    assert item["content"] == [{"type": "output_text", "text": agent.GREETING}]


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_grok_voice.web import app

    saved = os.environ.pop("XAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="XAI_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["XAI_API_KEY"] = saved
