"""Check the declarations handed to Dialog-RSN-1 against Rory's shared contract."""

import pytest

from rory_tools import SCHEMAS
from rory_tools.parity import (
    assert_argument_schemas_match,
    assert_description_matches,
    assert_required_arguments_match,
    assert_tool_names_match,
    assert_uses_shared_dispatcher,
)
from rory_tools.prompt import GREETING

from rory_polyai_dialog_rsn_1 import agent, tools

SURFACE = tools.tool_surface()


def test_session_passes_the_shared_tool_surface():
    assert agent._session_update()["session"]["tools"] is tools.TOOLS
    assert_tool_names_match(SURFACE)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_description_reaches_the_model_unedited(schema):
    assert_description_matches(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_required_arguments_match(schema):
    assert_required_arguments_match(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_argument_schemas_match(schema):
    assert_argument_schemas_match(SURFACE, schema)


def test_greeting_and_dispatcher_are_shared():
    # The prompt is this transport's own (see agent.py); the greeting is not.
    assert agent.GREETING is GREETING
    assert_uses_shared_dispatcher(agent)
