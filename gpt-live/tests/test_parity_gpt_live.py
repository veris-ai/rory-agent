"""GPT-Live's own parity run.

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

from rory_gpt_live import agent, tools

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
    # Function calls arrive as nested Responses events on the vendor websocket
    # rather than as framework callbacks, so the agent module dispatches.
    assert_uses_shared_dispatcher(agent)


def test_session_keeps_gpt_live_shape():
    """Tools sit on the Responses backend, not the live model, and both read
    the shared prompt. Pinned so a later edit cannot quietly move the tools or
    let the backend sit a different exam."""
    session = agent._session_start()["session"]
    assert session["model"] == agent.LIVE_MODEL
    assert session["instructions"].startswith(agent.AGENT_PROMPT)
    assert session["audio"] == {"output": {"voice": agent.LIVE_VOICE}}
    assert "tools" not in session
    backend = session["delegation"]["responses"]
    assert session["delegation"]["type"] == "responses"
    assert backend["model"] == agent.LIVE_BACKEND_MODEL
    assert backend["tools"] is tools.TOOLS
    assert backend["parallel_tool_calls"] is False
    assert agent.AGENT_PROMPT in backend["instructions"]


def test_greeting_is_a_trusted_instruction_append():
    greeting = agent._greeting_instruction()
    assert greeting["type"] == "session.instructions.append"
    assert greeting["delegation_id"] is None
    assert agent.GREETING in greeting["content"]


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_gpt_live.web import app

    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
