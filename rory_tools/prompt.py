"""Rory's system prompt, loaded from the one file every transport shares.

The prompt is not a transport detail. It carries the policy Rory is graded on —
verify before disclosing, never report a declined payment as success, transfer
rather than negotiate a shutoff — so two candidates reading different prompts
are not two measurements of the same agent. There is exactly one
``agent_desc.txt``, inside the rory-core package, and every transport loads it
from here.
"""

from __future__ import annotations

from pathlib import Path

def load_agent_prompt() -> str:
    """Read the shared prompt shipped inside the rory-core package."""
    return Path(__file__).with_name("agent_desc.txt").read_text()


# Transports with their own TTS speak this text verbatim. Speech-to-speech
# models (Gemini Live, for one) are asked for this wording and generate native
# audio, which is not guaranteed to be verbatim.
GREETING = "Thanks for calling Acme Energy, this is Rory — how can I help you today?"


def today_context() -> str:
    """The date sentence pinned to the snapshot's frozen clock.

    A bill is only "last month's" relative to now, and an instalment date has
    to be real — left to guess, a model picks a year out of its training data
    and Fineract refuses the transaction outright. ``rory_tools.clock`` reads
    the frozen instant the attempt restored, so this is the snapshot's today,
    not the wall clock's.
    """
    from .clock import now

    today = now()
    return (
        f"Today is {today:%A, %B %-d, %Y} ({today:%Y-%m-%d}). "
        f"The current month as YYYY-MM is {today:%Y-%m}. Use these "
        "for anything the caller describes as today, this week, or "
        "this month."
    )
