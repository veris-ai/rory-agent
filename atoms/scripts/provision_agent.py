"""Push Rory's entire config onto an existing Atoms agent — prompt, greeting,
turn-taking settings, voice, and the 16 tools.

This configures an agent that already exists. It never creates one and
never runs automatically — ``--agent-id`` (or ``ATOMS_AGENT_ID``) is
required, with no fallback.

Everything goes through ``PUT /agent/{id}/branches/{branchId}/draft``: the
top-level settings as their own fields, and tools nested as
``{"type": "single_prompt", "singlePromptConfig": {"prompt": ...,
"tools": [...]}}`` — see ``rory_atoms/tools.py``'s module docstring for why
that nesting is required and the per-tool shape.

Usage::

    ATOMS_API_KEY=... ATOMS_AGENT_ID=... \\
        uv run python -m atoms.scripts.provision_agent [--branch-name NAME]

Forks a branch from whichever is currently live, pushes everything below,
publishes, and activates it — so the branch is what the agent actually
answers calls with.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

import httpx

from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from rory_atoms.tools import NATIVE_END_CALL_TOOL, TOOLS

BASE_URL = "https://api.smallest.ai/atoms/v1"

# The two-sentence turn-taking profile this agent runs with. Matches a
# blank agent's own defaults except waitTimeInSecs — pushed explicitly
# anyway so this script is a complete, standalone record of the intended
# configuration rather than something that only works by coincidence with
# whatever the platform currently defaults to.
VOICE_DETECTION_CONFIG: Dict[str, Any] = {
    "confidence": 0.7,
    "minVolume": 0.6,
    "triggerTimeInSecs": 0.2,
    "releaseTimeInSecs": 0.2,
}
SMART_TURN_CONFIG: Dict[str, Any] = {"isEnabled": True, "waitTimeInSecs": 2.0}

SESSION_TIMEOUT_CONFIG: Dict[str, Any] = {"timeoutTimeInSecs": 900}
LLM_IDLE_TIMEOUT_CONFIG: Dict[str, Any] = {
    "chatTimeoutTimeInSecs": 60,
    "webcallTimeoutTimeInSecs": 20,
    "telephonyTimeoutTimeInSecs": 20,
    "maxRetries": 3,
}

LANGUAGE_CONFIG: Dict[str, Any] = {
    "default": "en",
    "supported": ["en"],
    "switching": {
        "isEnabled": False,
        "minWordsForDetection": 2,
        "strongSignalThreshold": 0.7,
        "weakSignalThreshold": 0.3,
        "minConsecutiveForWeakThresholdSwitch": 2,
    },
}
SYNTHESIZER_CONFIG: Dict[str, Any] = {
    "voiceConfig": {"model": "waves_lightning_v3_1_pro", "voiceId": "blake", "gender": "male"},
    "speed": 1,
    "sampleRate": 24000,
    "numberPronunciationLanguage": "auto",
}


def _client() -> httpx.Client:
    key = os.environ["ATOMS_API_KEY"]
    return httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)


def _raise_for_status(resp: httpx.Response) -> Dict[str, Any]:
    body = resp.json()
    if resp.status_code >= 400 or body.get("status") is False:
        raise RuntimeError(f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {body}")
    return body["data"]


def live_branch_id(client: httpx.Client, agent_id: str) -> str:
    """The branch to fork from — whichever is currently live, so a re-run
    builds on the agent's actual current state rather than its original
    default branch."""
    data = _raise_for_status(client.get(f"/agent/{agent_id}/branches"))
    for entry in data["branches"]:
        if entry["isLive"]:
            return entry["branch"]["_id"]
    raise RuntimeError(f"agent {agent_id} has no live branch")


def fork_branch(client: httpx.Client, agent_id: str, source_branch_id: str, name: str) -> str:
    data = _raise_for_status(client.post(
        f"/agent/{agent_id}/branches",
        json={"sourceBranchId": source_branch_id, "name": name},
    ))
    return data["_id"]


def push_draft(client: httpx.Client, agent_id: str, branch_id: str) -> None:
    # Appended once, at provisioning time, because the prompt lives on the
    # agent rather than being rebuilt per call — the same limitation every
    # other statically-provisioned candidate in this repo has. Re-run this
    # script to refresh it.
    prompt = f"{load_agent_prompt()}\n\n{today_context()}"
    body = {
        "globalPrompt": prompt,
        "firstMessage": GREETING,
        "slmModel": "electron",
        "transcriberType": "pulse",
        "language": LANGUAGE_CONFIG,
        "synthesizer": SYNTHESIZER_CONFIG,
        "voiceDetectionConfig": VOICE_DETECTION_CONFIG,
        "smartTurnConfig": SMART_TURN_CONFIG,
        "sessionTimeoutConfig": SESSION_TIMEOUT_CONFIG,
        "llmIdleTimeoutConfig": LLM_IDLE_TIMEOUT_CONFIG,
        "type": "single_prompt",
        "singlePromptConfig": {"prompt": prompt, "tools": [*TOOLS, NATIVE_END_CALL_TOOL]},
    }
    _raise_for_status(client.put(f"/agent/{agent_id}/branches/{branch_id}/draft", json=body))


def publish_and_activate(client: httpx.Client, agent_id: str, branch_id: str, label: str) -> None:
    _raise_for_status(client.post(
        f"/agent/{agent_id}/branches/{branch_id}/draft/publish", json={"label": label}
    ))
    # Publishing can run an async security scan ("scanning" -> "committed");
    # poll the branch until it has no open draft before activating, or
    # activation can land on the pre-publish revision.
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
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("ATOMS_AGENT_ID"),
        help="The existing agent to configure. Required; falls back to ATOMS_AGENT_ID.",
    )
    parser.add_argument("--branch-name", default="rory-config")
    parser.add_argument("--label", default="Rory prompt, voice, turn-taking, tools")
    args = parser.parse_args()

    if not args.agent_id:
        parser.error("--agent-id is required (or set ATOMS_AGENT_ID) — this configures an existing agent, it never creates one")

    with _client() as client:
        source_branch = live_branch_id(client, args.agent_id)
        branch_id = fork_branch(client, args.agent_id, source_branch, args.branch_name)
        print(f"branch: {branch_id}", file=sys.stderr)

        push_draft(client, args.agent_id, branch_id)
        publish_and_activate(client, args.agent_id, branch_id, args.label)
        print(f"published and activated with {len(TOOLS) + 1} tools", file=sys.stderr)


if __name__ == "__main__":
    main()
