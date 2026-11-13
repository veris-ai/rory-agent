"""Atoms' own parity run.

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

from rory_atoms import agent, tools

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


def test_provisioning_payload_is_what_tool_surface_was_read_from():
    """The thing a human actually pastes into the agent's Tools tab is the
    same object the parity check above ran against — not two copies that
    could quietly drift apart."""
    assert tools.provisioning_payload() is tools.TOOLS


def test_connect_url_requests_the_actors_rate_both_ways(monkeypatch):
    monkeypatch.setenv("SMALLEST_API_KEY", "sk_fixture")
    monkeypatch.setattr(agent, "ATOMS_AGENT_ID", "agent_fixture")
    url = agent._atoms_url()
    assert url.startswith(agent.ATOMS_BASE_URL + "?")
    assert "input_audio_format=pcm_24000" in url
    assert "output_audio_format=pcm_24000" in url
    assert "mode=webcall" in url
    assert "agent_id=agent_fixture" in url
    assert "token=sk_fixture" in url


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_atoms.web import app

    saved_key = os.environ.pop("SMALLEST_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="SMALLEST_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved_key is not None:
            os.environ["SMALLEST_API_KEY"] = saved_key


def test_candidate_fails_closed_without_an_agent_id(monkeypatch):
    monkeypatch.delenv("ATOMS_AGENT_ID", raising=False)
    from rory_atoms.web import app

    with pytest.raises(RuntimeError, match="ATOMS_AGENT_ID"):
        with TestClient(app) as client:
            client.get("/health")
