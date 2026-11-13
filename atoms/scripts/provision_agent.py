"""Push Rory's scriptable config onto a real Atoms agent.

Confirmed against the live API (not just docs, which don't enumerate a full
schema): ``PUT /agent/{id}/branches/{branchId}/draft`` accepts the same
camelCase fields shown on ``GET /agent/{id}`` — ``globalPrompt``,
``firstMessage``, ``voiceDetectionConfig``, ``smartTurnConfig``,
``sttMissDetectorConfig`` all round-tripped correctly in testing.

``tools`` is NOT one of them. Every shape tried (``tools``, ``workflowTools``,
``workflow.tools``, ``workflow_tools``, as a top-level field on the draft and
on a guessed ``.../draft/workflow`` route that turned out not to exist) was
rejected with ``{"status": false, "errors": ["No recognized config fields in
request body"]}``, or 404. Registering Rory's 16 tools as Atoms client tools
is a genuine manual step — see ``rory_atoms/tools.py``'s module docstring —
not a gap in this script.

Usage::

    SMALLEST_API_KEY=... uv run python -m atoms.scripts.provision_agent \\
        --agent-id <existing agent id, or omit to create one> \\
        --branch-name atoms-transport

Creates (or reuses) an agent, forks a branch from its live one, pushes the
config above, publishes, and activates — so the branch is what the agent
actually answers calls with. Printed at the end: the agent id to use as
``ATOMS_AGENT_ID``, and a reminder of the one step this cannot do for you.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import httpx

from rory_tools.prompt import GREETING, load_agent_prompt

BASE_URL = "https://api.smallest.ai/atoms/v1"

# The three turn-taking settings from atoms/README.md's benchmark-mapped
# table. smartTurnConfig/sttMissDetectorConfig are already on by default on a
# new agent — pushed anyway so a re-run against an existing, differently
# configured agent still lands on these values. releaseTimeInSecs is the one
# that actually needs moving from its 0.2s default.
TURN_TAKING_CONFIG: Dict[str, Any] = {
    "voiceDetectionConfig": {
        "confidence": 0.7,
        "minVolume": 0.6,
        "triggerTimeInSecs": 0.2,
        "releaseTimeInSecs": 0.7,
    },
    "smartTurnConfig": {"isEnabled": True, "waitTimeInSecs": 1.5},
    "sttMissDetectorConfig": {"isEnabled": True, "repeatPromptMessages": []},
}


def _client() -> httpx.Client:
    key = os.environ["SMALLEST_API_KEY"]
    return httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)


def _raise_for_status(resp: httpx.Response) -> Dict[str, Any]:
    body = resp.json()
    if resp.status_code >= 400 or body.get("status") is False:
        raise RuntimeError(f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {body}")
    return body["data"]


def create_agent(client: httpx.Client, name: str) -> str:
    data = _raise_for_status(client.post("/agent", json={"name": name}))
    return data  # POST /agent returns the bare id string in "data"


def default_branch_id(client: httpx.Client, agent_id: str) -> str:
    data = _raise_for_status(client.get(f"/agent/{agent_id}/branches"))
    for entry in data["branches"]:
        if entry["branch"]["isDefault"]:
            return entry["branch"]["_id"]
    raise RuntimeError(f"agent {agent_id} has no default branch")


def fork_branch(client: httpx.Client, agent_id: str, source_branch_id: str, name: str) -> str:
    data = _raise_for_status(client.post(
        f"/agent/{agent_id}/branches",
        json={"sourceBranchId": source_branch_id, "name": name},
    ))
    return data["_id"]


def push_draft(client: httpx.Client, agent_id: str, branch_id: str) -> None:
    body = {
        "globalPrompt": load_agent_prompt(),
        "firstMessage": GREETING,
        **TURN_TAKING_CONFIG,
    }
    _raise_for_status(client.put(f"/agent/{agent_id}/branches/{branch_id}/draft", json=body))


def publish_and_activate(client: httpx.Client, agent_id: str, branch_id: str, label: str) -> None:
    _raise_for_status(client.post(
        f"/agent/{agent_id}/branches/{branch_id}/draft/publish", json={"label": label}
    ))
    # Publishing can run an async security scan ("scanning" -> "committed");
    # poll the branch until it has no open draft before activating.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        data = _raise_for_status(client.get(f"/agent/{agent_id}/branches/{branch_id}"))
        if not data["hasOpenDraft"]:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"branch {branch_id} still had an open draft after 60s")
    _raise_for_status(client.post(f"/agent/{agent_id}/branches/{branch_id}/live"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", help="Reuse an existing agent instead of creating one")
    parser.add_argument("--agent-name", default="Rory (Acme Energy, rory-agent atoms transport)")
    parser.add_argument("--branch-name", default="atoms-transport")
    parser.add_argument("--label", default="Rory prompt and turn-taking")
    args = parser.parse_args()

    with _client() as client:
        agent_id = args.agent_id or create_agent(client, args.agent_name)
        print(f"agent: {agent_id}", file=sys.stderr)

        source_branch = default_branch_id(client, agent_id)
        branch_id = fork_branch(client, agent_id, source_branch, args.branch_name)
        print(f"branch: {branch_id}", file=sys.stderr)

        push_draft(client, agent_id, branch_id)
        publish_and_activate(client, agent_id, branch_id, args.label)
        print("published and activated", file=sys.stderr)

    print(json.dumps({"ATOMS_AGENT_ID": agent_id}))
    print(
        "\nOne step left this script cannot do: register the 16 tools below "
        "as CLIENT tools (not API tools) in the console's Tools tab — "
        "`uv run python -m rory_atoms.tools` to print them again.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
