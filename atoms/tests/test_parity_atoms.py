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


# make_payment.amount_cents carries a `minimum` constraint Atoms' parameter
# schema has no field for (see tools.py's module docstring and
# _UNREPRESENTABLE_CONSTRAINTS) — excluded from the generic check below and
# covered on its own, explicitly, rather than silently passing or failing.
_NO_FULL_FIDELITY = {"make_payment"}


@pytest.mark.parametrize(
    "schema", [s for s in SCHEMAS if s.name not in _NO_FULL_FIDELITY], ids=lambda s: s.name
)
def test_argument_schemas_match(schema):
    assert_argument_schemas_match(SURFACE, schema)


def test_make_payment_argument_schemas_match_except_unrepresentable_constraints():
    schema = next(s for s in SCHEMAS if s.name == "make_payment")
    properties = SURFACE[schema.name]["properties"]
    assert set(properties) == set(schema.properties)
    for name, expected in schema.properties.items():
        dropped = tools._UNREPRESENTABLE_CONSTRAINTS.get((schema.name, name), set())
        expected = {k: v for k, v in expected.items() if k not in dropped}
        assert properties[name] == expected, f"{schema.name}.{name}: drifted beyond the documented gap"


def test_prompt_and_greeting_are_the_shared_ones():
    assert_shares_prompt_and_greeting(agent)


def test_tool_calls_go_through_the_shared_gate():
    assert_uses_shared_dispatcher(agent)


def test_provisioning_payload_is_what_tool_surface_was_read_from():
    """What provision_agent.py actually pushes is the same object the parity
    check above ran against — not two copies that could quietly drift
    apart."""
    assert tools.provisioning_payload() is tools.TOOLS


def test_native_end_call_tool_is_not_one_of_the_shared_sixteen():
    assert tools.NATIVE_END_CALL_TOOL["name"] not in {s.name for s in SCHEMAS}


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
