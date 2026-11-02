"""Check the declarations handed to Dialog-RSN-1 against Rory's shared contract.

The one allowed difference is the spoken ``utterance`` argument on the tools in
``SPOKEN_TOOLS`` (see tools.py). It is checked on its own, then taken out, and
everything left has to match the shared declaration exactly.
"""

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


def _without_spoken_argument(surface):
    shared = {}
    for name, tool in surface.items():
        properties = dict(tool["properties"])
        required = list(tool["required"])
        if name in tools.SPOKEN_TOOLS:
            properties.pop(tools.SPOKEN_ARGUMENT)
            required.remove(tools.SPOKEN_ARGUMENT)
        shared[name] = {**tool, "properties": properties, "required": required}
    return shared


SHARED_SURFACE = _without_spoken_argument(SURFACE)


def test_session_passes_the_shared_tool_surface():
    assert agent._session_update()["session"]["tools"] is tools.TOOLS
    assert_tool_names_match(SURFACE)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_description_reaches_the_model_unedited(schema):
    assert_description_matches(SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_required_arguments_match(schema):
    assert_required_arguments_match(SHARED_SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_argument_schemas_match(schema):
    assert_argument_schemas_match(SHARED_SURFACE, schema)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda schema: schema.name)
def test_only_the_spoken_tools_carry_a_required_poly_spoken_utterance(schema):
    tool = SURFACE[schema.name]
    if schema.name not in tools.SPOKEN_TOOLS:
        assert tools.SPOKEN_ARGUMENT not in tool["properties"]
        return
    assert tool["properties"][tools.SPOKEN_ARGUMENT] == {
        "type": "string",
        "description": tools.SPOKEN_TOOLS[schema.name],
        "poly_spoken": True,
    }
    assert tool["required"][-1] == tools.SPOKEN_ARGUMENT
    assert tools.SPOKEN_ARGUMENT not in schema.properties


def test_spoken_tools_are_shared_tools():
    assert set(tools.SPOKEN_TOOLS) <= {schema.name for schema in SCHEMAS}


def test_greeting_and_dispatcher_are_shared():
    # The prompt is this transport's own (see agent.py); the greeting is not.
    assert agent.GREETING is GREETING
    assert_uses_shared_dispatcher(agent)
