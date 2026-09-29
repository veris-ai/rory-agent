"""Cartesia's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, is what lets this transport ship its own image and its own
lockfile without needing any other voice SDK resolvable alongside it.

One allowance: Cartesia's webhook body schema has no ``minimum`` and rejects a
tool that carries one, so the adapter drops it. The argument-schema check here
compares against the shared declaration less that keyword, and a separate test
pins that exactly one argument is affected.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rory_tools import SCHEMAS
from rory_tools.parity import (
    assert_description_matches,
    assert_required_arguments_match,
    assert_shares_prompt_and_greeting,
    assert_tool_names_match,
    assert_uses_shared_dispatcher,
)

from rory_cartesia import agent, tools

SURFACE = tools.tool_surface(tools.build_tools("https://rory.example/tool", "secret"))


def test_tool_names_match_the_shared_declaration():
    assert_tool_names_match(SURFACE)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_description_reaches_the_model_unedited(schema):
    assert_description_matches(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_required_arguments_match(schema):
    assert_required_arguments_match(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_argument_schemas_match_less_the_keyword_cartesia_rejects(schema):
    properties = SURFACE[schema.name]["properties"]
    assert set(properties) == set(schema.properties), f"{schema.name}: argument names differ"
    for name, expected in schema.properties.items():
        assert properties[name] == {k: v for k, v in expected.items() if k not in tools.UNSUPPORTED_KEYWORDS}, (
            f"{schema.name}.{name}: argument schema differs from the shared declaration"
        )


def test_the_dropped_keyword_touches_exactly_one_argument():
    assert tools.UNSUPPORTED_KEYWORDS == {"minimum"}
    dropped = [(s.name, n) for s in SCHEMAS for n, p in s.properties.items() if "minimum" in p]
    assert dropped == [("make_payment", "amount_cents")]


def test_webhook_tools_post_to_the_pods_url_with_its_bearer():
    for tool in tools.build_tools("https://rory.example/tool", "secret"):
        assert tool["type"] == "webhook"
        assert tool["api_schema"]["url"] == f"https://rory.example/tool/{tool['name']}"
        assert tool["api_schema"]["method"] == "POST"
        assert tool["api_schema"]["authentication"] == {
            "mode": "bearer",
            "token": {"type": "secret", "secret_value": "secret"},
        }
        assert tool["response_timeout_secs"] == tools.RESPONSE_TIMEOUT_S


def test_prompt_and_greeting_are_the_shared_ones():
    assert_shares_prompt_and_greeting(agent)


def test_tool_calls_go_through_the_shared_gate():
    # Cartesia dispatches from the agent module: the webhook in ``web`` hands
    # each request to ``run_tool_call``, which runs the shared dispatcher.
    assert_uses_shared_dispatcher(agent)


@pytest.mark.parametrize("missing", ["CARTESIA_API_KEY", "PUBLIC_BASE_URL"])
def test_candidate_fails_closed_without_its_key_or_public_url(monkeypatch, missing):
    """A missing key or webhook address must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_cartesia.web import app

    monkeypatch.setenv("STRIPE_API_KEY", "test")
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://rory.example.com/hooks/1")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(app) as client:
            client.get("/health")
