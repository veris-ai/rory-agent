"""Prism's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, means this transport's tests run with only its own voice SDK
installed.
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

from rory_rime_prism import agent, tools

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
    assert agent.GREETING in agent.GREETING_INSTRUCTION


def test_tool_calls_go_through_the_shared_gate():
    # Prism dispatches from the agent module: its tool calls arrive as session
    # events rather than as framework callbacks.
    assert_uses_shared_dispatcher(agent)


def test_session_settings_are_pinned():
    """The connect options the run depends on.

    The shared prompt plus the frozen-clock date, the shared tools unchanged, and
    no interruption or turn-detection override, so Prism keeps its own defaults.
    """
    options = agent._connect_options()
    assert set(options) == {"voice", "instructions", "tools"}
    assert options["voice"] == agent.PRISM_VOICE == "aurora"
    assert options["instructions"].startswith(agent.AGENT_PROMPT)
    assert options["tools"] is tools.TOOLS


def test_no_tool_overrides_prisms_filler_phrase():
    assert all(tool.prism_filler is None for tool in tools.TOOLS)


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_rime_prism.web import app

    saved = os.environ.pop("RIME_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="RIME_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["RIME_API_KEY"] = saved


def test_every_write_tool_is_a_real_shared_tool():
    assert tools.WRITE_TOOLS <= {schema.name for schema in SCHEMAS}
