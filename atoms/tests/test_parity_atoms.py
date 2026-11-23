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

# tools.TOOLS is what session.update actually sends (see agent._connect_once) —
# each entry's "parameters" is ToolSchema.parameters() verbatim, so this is a
# straight read of the wire payload, not a reconstruction from a lossier shape.
SURFACE = {
    tool["name"]: {
        "description": tool["description"],
        "required": tool["parameters"]["required"],
        "properties": tool["parameters"]["properties"],
    }
    for tool in tools.TOOLS
}


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


def test_session_update_sends_the_same_tools_object_parity_checked_above():
    """What agent._connect_once actually sends is the same object the parity
    check above ran against — not two copies that could quietly drift apart."""
    assert agent.TOOLS is tools.TOOLS


def test_connect_url_requests_the_actors_rate_both_ways(monkeypatch):
    monkeypatch.setattr(agent, "ATOMS_AGENT_ID", "agent_fixture")
    url = agent._atoms_url()
    assert url.startswith(agent.ATOMS_BASE_URL + "?")
    assert "input_audio_format=pcm_24000" in url
    assert "output_audio_format=pcm_24000" in url
    assert "mode=webcall" in url
    assert "agent_id=agent_fixture" in url


def test_the_key_never_rides_in_the_url(monkeypatch):
    monkeypatch.setenv("ATOMS_API_KEY", "sk_fixture_should_not_appear_in_url")
    monkeypatch.setattr(agent, "ATOMS_AGENT_ID", "agent_fixture")
    assert "sk_fixture_should_not_appear_in_url" not in agent._atoms_url()


def test_the_key_rides_in_the_authorization_header(monkeypatch):
    monkeypatch.setenv("ATOMS_API_KEY", "sk_fixture")
    assert agent._auth_headers() == {"Authorization": "Bearer sk_fixture"}


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_atoms.web import app

    saved_key = os.environ.pop("ATOMS_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="ATOMS_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved_key is not None:
            os.environ["ATOMS_API_KEY"] = saved_key


def test_candidate_fails_closed_without_an_agent_id(monkeypatch):
    monkeypatch.delenv("ATOMS_AGENT_ID", raising=False)
    from rory_atoms.web import app

    with pytest.raises(RuntimeError, match="ATOMS_AGENT_ID"):
        with TestClient(app) as client:
            client.get("/health")
