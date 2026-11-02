"""The runtime and Lambda modules build their AWS clients at import, from the environment they run in."""

import os

os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("SESSIONS_TABLE", "rory-agentcore-sessions-test")
