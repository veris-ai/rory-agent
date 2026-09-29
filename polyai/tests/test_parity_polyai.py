"""PolyAI's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, is what lets this transport ship its own image and its own
lockfile without needing any other voice SDK resolvable alongside it.

Agent Studio cannot express two things the shared declaration says, and the
two cases are pinned as strict expected failures rather than hidden: every
parameter is required (no defaults, no Optional), and a parameter is a type
plus a description, so a JSON-Schema keyword like ``minimum`` has no place.
If the platform ever gains either, the xfail turns into a failure and the
adapter gets to say it properly.
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

from rory_polyai import studio, tools

SURFACE = tools.tool_surface()


def _has_optional_parameter(schema) -> bool:
    return list(schema.required) != list(schema.properties)


def _has_schema_keywords(schema) -> bool:
    return any(set(prop) - {"type", "description"} for prop in schema.properties.values())


def _params(lossy, reason):
    return [
        pytest.param(
            schema,
            id=schema.name,
            marks=[pytest.mark.xfail(strict=True, reason=reason)] if lossy(schema) else [],
        )
        for schema in SCHEMAS
    ]


def test_tool_names_match_the_shared_declaration():
    assert_tool_names_match(SURFACE)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.name)
def test_description_reaches_the_model_unedited(schema):
    assert_description_matches(SURFACE, schema)


@pytest.mark.parametrize(
    "schema",
    _params(_has_optional_parameter, "Agent Studio has no optional parameters: every one is required, with no default"),
)
def test_required_arguments_match(schema):
    assert_required_arguments_match(SURFACE, schema)


@pytest.mark.parametrize(
    "schema",
    _params(_has_schema_keywords, "Agent Studio parameters carry a type and a description only; 'minimum' has no place"),
)
def test_argument_schemas_match(schema):
    assert_argument_schemas_match(SURFACE, schema)


def test_prompt_and_greeting_are_the_shared_ones():
    # The prompt and greeting are pushed to the project by the provisioner,
    # not sent per call by the bridge, so the provisioner is what is checked.
    assert_shares_prompt_and_greeting(studio)


def test_tool_calls_go_through_the_shared_gate():
    # PolyAI dispatches from the adapter: the rendered functions' webhook
    # resolves each envelope through ``run_tool_call``.
    assert_uses_shared_dispatcher(tools)


def test_candidate_fails_closed_without_its_credentials():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_polyai.web import app

    saved = os.environ.pop("POLYAI_AUTH_TOKEN", None)
    try:
        with pytest.raises(RuntimeError, match="POLYAI_AUTH_TOKEN"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["POLYAI_AUTH_TOKEN"] = saved
