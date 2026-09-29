"""Gradium's own parity run.

The assertions live in ``rory_tools.parity`` and are the same ones every
transport runs. Keeping them here, rather than in a suite that imports all
transports at once, is what lets this transport ship its own image and its own
lockfile without needing any other voice SDK resolvable alongside it.

gradbot takes the JSON-Schema object as a string and passes it to the LLM
unchanged, so unlike Cartesia nothing is dropped: the full argument schemas are
compared. The rest pins the two gradbot defaults the session overrides.
"""

from __future__ import annotations

import gradbot
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
from rory_tools.prompt import GREETING, today_context

from rory_gradium import agent, tools

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


def test_instructions_are_the_shared_prompt_then_the_date_then_the_opening_line():
    """gradbot's scaffold tells the model to greet on ``[start]``; the shared
    prompt says the greeting already happened. The opening-line section wins
    and names the shared greeting verbatim."""
    instructions = agent.session_instructions()
    assert instructions.startswith(agent.AGENT_PROMPT)
    assert today_context() in instructions
    assert instructions.endswith(agent.OPENING_LINE)
    assert f'Open with exactly: "{GREETING}"' in agent.OPENING_LINE


def test_session_config_overrides_only_the_silence_nudge():
    config = agent.session_config()
    assert config.silence_timeout_s == 0.0
    assert config.assistant_speaks_first is True
    assert config.language == gradbot.Lang.En
    assert config.voice_id == agent.GRADIUM_VOICE_ID
    assert [t.name for t in config.tools] == [t.name for t in tools.TOOLS]
    # gradbot defaults, left alone.
    assert config.flush_duration_s == 0.5
    assert config.padding_bonus == 0.0


def test_health_answers_without_starting_a_gradbot_session(monkeypatch):
    from rory_gradium.web import app

    def boom(*args, **kwargs):
        raise AssertionError("gradbot.run must not be called on boot or health")

    monkeypatch.setattr(gradbot, "run", boom)
    for name in ("STRIPE_API_KEY", "GRADIUM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "test")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("missing", ["GRADIUM_API_KEY", "OPENAI_API_KEY", "STRIPE_API_KEY"])
def test_candidate_fails_closed_without_its_keys(monkeypatch, missing):
    """A missing key must stop the boot, not each call.

    Failing per-call would report a hundred timed-out simulations and a zero
    score — indistinguishable from an agent that cannot do the task.
    """
    from rory_gradium.web import app

    for name in ("STRIPE_API_KEY", "GRADIUM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "test")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(app) as client:
            client.get("/health")
