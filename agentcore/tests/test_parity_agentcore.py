"""The AgentCore candidate's own parity run.

The model runs in the AgentCore Runtime and the tools run in the tools Lambda, so
those are the modules checked: the runtime must hand Bedrock the shared tools,
prompt and greeting, and the tools Lambda must run them through the shared gate.
"""

from __future__ import annotations

import pytest
from rory_agentcore import agent, runtime, tools, tools_lambda
from rory_agentcore.contract import SESSION_ARG

from rory_tools import SCHEMAS
from rory_tools.parity import (
    assert_argument_schemas_match,
    assert_description_matches,
    assert_required_arguments_match,
    assert_shares_prompt_and_greeting,
    assert_tool_names_match,
    assert_uses_shared_dispatcher,
)
from rory_tools.prompt import GREETING

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
    assert_shares_prompt_and_greeting(runtime)
    assert agent.GREETING is GREETING  # the pod speaks the same opening line


def test_tool_calls_go_through_the_shared_gate():
    assert_uses_shared_dispatcher(tools_lambda)


def test_model_never_sees_the_session_argument():
    """The gateway needs the session; the model must not be the one supplying it."""
    assert all(SESSION_ARG not in props["properties"] for props in SURFACE.values())
    for gateway_tool, schema in zip(tools.GATEWAY_TOOLS, SCHEMAS, strict=True):
        assert gateway_tool["name"] == schema.name
        assert gateway_tool["description"] == schema.description
        assert gateway_tool["inputSchema"]["properties"] == {
            **{name: tools._for_gateway(prop) for name, prop in schema.properties.items()},
            SESSION_ARG: {"type": "string", "description": "The call this tool runs for."},
        }
        assert gateway_tool["inputSchema"]["required"] == [*schema.required, SESSION_ARG]
