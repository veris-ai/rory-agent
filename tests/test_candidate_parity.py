"""Checks that belong to the shared half, not to any one transport.

Every candidate must sit the same exam, so everything except the transport has
to be identical: the same 16 tools, the same descriptions and required
arguments, the same system prompt, the same opening line. A benchmark that lets
those drift is measuring the drift.

That enforcement runs **inside each transport's own suite**, through
``rory_tools.parity`` — see ``<transport>/tests/test_parity_*.py``, for example
``pipecat/tests/test_parity_pipecat.py``. Every transport ships its own image,
and no environment is guaranteed to hold two voice SDKs together, so each one
is compared against the single shared declaration instead of against the
others. That gives the same guarantee transitively, and catches drift in the
transport's own test run.

What is left here is what has no transport in it at all.
"""

from __future__ import annotations

from rory_tools import SCHEMAS, TOOL_NAMES
from rory_tools.impl import IMPLEMENTATIONS
from rory_tools.prompt import GREETING, load_agent_prompt


def test_every_declared_tool_has_an_implementation():
    assert set(TOOL_NAMES) == set(IMPLEMENTATIONS)


def test_tool_names_are_unique_and_complete():
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))
    assert [s.name for s in SCHEMAS] == TOOL_NAMES


def test_there_is_exactly_one_prompt_and_one_greeting():
    """The single source both halves of the parity guarantee rest on.

    Each transport asserts it uses *these*; that is what makes "every transport
    matches the shared declaration" equivalent to "the transports match each
    other" without ever importing two of them together.
    """
    assert load_agent_prompt().strip()
    assert GREETING == "Thanks for calling Acme Energy, this is Rory — how can I help you today?"
