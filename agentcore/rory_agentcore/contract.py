"""Names the four pieces of the AgentCore candidate agree on.

The pod, the Lex fulfillment Lambda, the AgentCore Runtime and the tools Lambda
run in four different places. Everything they must spell the same way lives here.
"""

from __future__ import annotations

# Lex session attributes the pod sets on every turn. The fulfillment Lambda copies
# the twin settings into the session record the tools Lambda reads.
ATTR_STRIPE_API_BASE = "stripe_api_base"
ATTR_STRIPE_API_KEY = "stripe_api_key"
ATTR_FINERACT_API_BASE = "fineract_api_base"
ATTR_REFERENCE_TIME = "reference_time"
# Set only on the turn after a barge-in: the reply as the caller heard it.
ATTR_INTERRUPTED_REPLY = "interrupted_reply"

# The argument that carries the session into a tool call. The gateway target declares
# it; the model never sees it, because the runtime adds it after the model has chosen.
SESSION_ARG = "session_id"

# Gateway tool names are "<target>___<tool>".
GATEWAY_TARGET = "rory-tools"

# Appended to a reply the caller talked over, so the model reads its own history as
# cut off rather than said.
INTERRUPTED_MARKER = "… [interrupted by the caller]"

# Bedrock Converse histories must open with a user turn, and the pod speaks the
# greeting before the caller says anything. This stands in for the call connecting.
CALL_CONNECTED = "[call connected]"
