"""Hydra's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, is what lets this transport ship its own image and its
own lockfile without needing any other voice SDK resolvable alongside it.
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

from rory_hydra import agent, tools

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
    assert_uses_shared_dispatcher(agent)


def test_session_keeps_hydra_shape():
    """The session.configure shape, pinned."""
    session = agent._session_configure()["session"]
    assert session["voice"] == agent.HYDRA_VOICE
    assert session["instructions"].startswith(agent.AGENT_PROMPT)
    assert session["generate_initial_response"] is False
    assert session["tools"] is tools.TOOLS


def test_output_rate_table_covers_the_default_model():
    assert agent.HYDRA_OUTPUT_RATE_HZ[agent.HYDRA_MODEL] == 24000


def test_greeting_trigger_forces_the_exact_wording():
    trigger = agent._greeting_trigger()
    assert trigger["type"] == "response.create"
    assert agent.GREETING in trigger["response"]["instructions"]


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_hydra.web import app

    saved = os.environ.pop("SMALLEST_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="SMALLEST_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["SMALLEST_API_KEY"] = saved
