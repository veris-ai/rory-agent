"""Parity checks a transport runs against itself, inside its own environment.

Each transport ships its own image, and no environment is guaranteed to hold
two voice SDKs together, so transports are never compared with each other
directly. Instead each transport checks its own framework objects against the
single shared declaration in ``rory_tools.schemas``. Equality *between*
transports then follows transitively, because there is only one thing for them
to be equal to. A transport that reworded a description fails in its own test
run, before its own image is built.

A transport participates by exporting ``tool_surface()`` from its adapter,
reading the values back out of the objects it actually hands its framework. It
must not rebuild the surface from ``SCHEMAS``: that would assert the shared
declaration equals itself and prove nothing about what the model was sent.

The per-tool checks stay one assertion per tool so a transport's suite still
names the tool that drifted, rather than failing once for the whole surface.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import TOOL_NAMES, ToolSchema

# {tool name: {"description": str, "required": [str], "properties": {name: schema}}}
ToolSurface = Mapping[str, Mapping[str, Any]]


def assert_tool_names_match(surface: ToolSurface) -> None:
    """The transport declares every shared tool, and invents none of its own."""
    assert set(surface) == set(TOOL_NAMES), (
        f"tool names drifted: missing={sorted(set(TOOL_NAMES) - set(surface))} "
        f"unexpected={sorted(set(surface) - set(TOOL_NAMES))}"
    )


def assert_description_matches(surface: ToolSurface, schema: ToolSchema) -> None:
    """A reworded description is a different exam question."""
    assert surface[schema.name]["description"] == schema.description, (
        f"{schema.name}: the transport edited the tool description"
    )


def assert_required_arguments_match(surface: ToolSurface, schema: ToolSchema) -> None:
    assert list(surface[schema.name]["required"]) == list(schema.required), (
        f"{schema.name}: required arguments differ from the shared declaration"
    )


def assert_argument_schemas_match(surface: ToolSurface, schema: ToolSchema) -> None:
    """Argument names and their full schemas, as the model will receive them."""
    properties = surface[schema.name]["properties"]
    assert set(properties) == set(schema.properties), (
        f"{schema.name}: argument names differ from the shared declaration"
    )
    for name, expected in schema.properties.items():
        assert properties[name] == expected, (
            f"{schema.name}.{name}: argument schema differs from the shared declaration"
        )


def assert_shares_prompt_and_greeting(agent_module: Any) -> None:
    """The transport reads the one prompt and the one opening line.

    Identity against the shared module, not equality between two transports —
    the same guarantee, checkable without importing anyone else.
    """
    from .prompt import GREETING, load_agent_prompt

    assert agent_module.AGENT_PROMPT == load_agent_prompt(), (
        "transport is not using the shared agent_desc.txt"
    )
    assert agent_module.GREETING is GREETING, (
        "transport carries its own greeting instead of the shared one"
    )


def assert_uses_shared_dispatcher(module: Any) -> None:
    """Tool calls route through the shared gate.

    Checked by identity, so a transport that grew its own copy of the
    verification gate fails here rather than silently grading its candidate
    under different rules.
    """
    from .dispatch import dispatch

    assert module.dispatch is dispatch, (
        "transport is not calling rory_tools.dispatch — it has its own gate"
    )
