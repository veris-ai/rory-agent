"""The Lex hook Lambda: keeps the caller's Lex conversation open after every turn.

Lex closes the dialog after an utterance that no intent fulfills, and a closed
dialog ends the conversation the stream belongs to. AWS's Connect samples answer
the fallback intent with ``ElicitIntent``, which asks for the caller's next turn.
This hook does only that. The agent is not called here: the pod takes the
transcript off the stream, which Lex sends before this hook runs, and streams the
agent's reply itself.

Lex requires a message with ``ElicitIntent``. The reply is not known here, so the
message is a fixed placeholder; the pod never speaks it, and acknowledges it as
played so Lex listens for the next turn.
"""

from __future__ import annotations

from typing import Any

PLACEHOLDER = "(the agent's reply streams from AgentCore)"


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return {
        "sessionState": {
            "dialogAction": {"type": "ElicitIntent"},
            "sessionAttributes": event["sessionState"].get("sessionAttributes", {}),
        },
        "messages": [{"contentType": "PlainText", "content": PLACEHOLDER}],
    }
