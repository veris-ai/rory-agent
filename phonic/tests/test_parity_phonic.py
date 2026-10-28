"""Phonic's own parity run.

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

from rory_phonic import agent, tools

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
    # Phonic dispatches from the agent module rather than the adapter, because
    # its tool calls arrive as messages on the vendor websocket rather than as
    # framework callbacks — the same shape as the Grok transport.
    assert_uses_shared_dispatcher(agent)


def test_config_keeps_the_shared_turn_taking_and_audio():
    """The Phonic config the run depends on, pinned.

    Audio in and out at 24 kHz to match the actor, 800 ms of end-of-turn silence,
    the shared greeting spoken first, the shared tools handed over unchanged, and
    Phonic's built-in turn-taking tool after them.
    """
    config = agent._config()
    assert config["type"] == "config"
    assert config["input_format"] == agent.AUDIO_FORMAT == "pcm_24000"
    assert config["output_format"] == agent.AUDIO_FORMAT
    assert config["vad_min_silence_duration_ms"] == 800
    assert config["welcome_message"] is agent.GREETING
    assert config["system_prompt"].startswith(agent.AGENT_PROMPT)
    assert config["tools"] == tools.TOOLS + tools.BUILT_IN_TOOLS
    assert config["phonic_model"] == "phonic-1"
    assert config["intelligence_level"] == "high"


def test_shared_tools_are_sync_and_uninterruptible():
    """Caller speech must not cancel a tool call in flight.

    By default Phonic cancels a sync call when the caller talks over it. A call that
    has started may already have acted, so the model asks again and a payment is
    recorded twice. With ``uninterruptible`` Phonic waits for the output.
    """
    for tool in tools.TOOLS:
        assert tool["execution_mode"] == "sync"
        assert tool["uninterruptible"] is True


def test_built_in_turn_taking_tool_is_outside_the_shared_surface():
    """The 16 shared tools are the parity surface; the built-in is Phonic's addition."""
    assert [t["name"] for t in tools.BUILT_IN_TOOLS] == ["choose_not_to_respond"]
    assert tools.BUILT_IN_TOOLS[0]["tool_config"] == {"respond_after_sec": 3}
    assert "choose_not_to_respond" not in tools.tool_surface()
    assert len(tools.tool_surface()) == 16


def test_candidate_fails_closed_without_its_credential():
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_phonic.web import app

    saved = os.environ.pop("PHONIC_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="PHONIC_API_KEY"):
            with TestClient(app) as client:
                client.get("/health")
    finally:
        if saved is not None:
            os.environ["PHONIC_API_KEY"] = saved
