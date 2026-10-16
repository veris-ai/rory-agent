"""Read back the actual Dialt manifest using the shared parity assertions."""

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
from rory_dialt import agent, tools
from rory_dialt.web import app


def test_shared_contract():
    surface = tools.tool_surface()
    assert_tool_names_match(surface)
    assert_shares_prompt_and_greeting(agent)
    assert_uses_shared_dispatcher(agent)
    for schema in SCHEMAS:
        assert_description_matches(surface, schema)
        assert_required_arguments_match(surface, schema)
        assert_argument_schemas_match(surface, schema)


@pytest.mark.parametrize("missing", ["DIALT_API_KEY", "STRIPE_API_KEY"])
def test_missing_credentials_fail_at_boot(monkeypatch, missing):
    monkeypatch.setenv("DIALT_API_KEY", "fixture")
    monkeypatch.setenv("STRIPE_API_KEY", "fixture")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(app):
            pass


def test_health_and_invalid_tier(monkeypatch):
    monkeypatch.setenv("DIALT_API_KEY", "fixture")
    monkeypatch.setenv("STRIPE_API_KEY", "fixture")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
    monkeypatch.setattr(agent, "DIALT_BRAIN", "not-a-tier")
    with pytest.raises(ValueError, match="fast or smart"):
        with TestClient(app):
            pass
