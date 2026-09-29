"""Rory's world-facing half: the tool surface, its implementations, and the
vendor clients behind them — everything that is the same whichever transport
is carrying the call.

A transport package is expected to use exactly three things from here:

    SCHEMAS      the 16 tools, in a framework-neutral shape to adapt
    CallSession  one per call, holding who the caller proved themselves to be
    dispatch     run one tool call and get the model-visible result back

Keeping the split at that line is what makes a comparison between transports
mean something: the candidates differ in how audio and function calls are
carried, and in nothing else.
"""

from .dispatch import dispatch
from .schemas import SCHEMAS, TOOL_NAMES, ToolSchema
from .session import CallSession, clients

__all__ = [
    "SCHEMAS",
    "TOOL_NAMES",
    "CallSession",
    "ToolSchema",
    "clients",
    "dispatch",
]
