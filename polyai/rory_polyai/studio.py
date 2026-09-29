"""Agent Studio provisioning through the PolyAI ADK.

A PolyAI agent is not code this image runs. It is project configuration in
Agent Studio — persona, rules, greeting, Python functions — edited through a
local project directory with the ``poly`` CLI and executed in PolyAI's cloud.
The ADK is the only write path to that configuration, and the tool webhook
URL is baked into the function bodies, so Rory is pushed onto the project at
every boot, before ``/health`` answers:

    poly init        a fresh local copy of the project
    overlay          persona and rules from the shared prompt, the shared
                     greeting, barge-in, and one function per shared tool
    poly push        which lands on a working branch, not main
    poly branch merge
    poly deployments promote   sandbox -> pre-release -> live

The WebRTC gateway connects calls to the live deployment, which is why the
promotion runs all the way. A rebooted pod with the same webhook URL pushes
identical content and the ADK reports "No changes detected"; that is a
success here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from loguru import logger

from rory_tools import TOOL_NAMES
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import render_functions

REGION = os.environ.get("POLYAI_REGION", "us-1")

# Rebuilt on every provision: the ADK forbids reusing a project directory
# across projects, and a stale copy would push a stale webhook URL.
BASE_PATH = Path("/tmp/rory-polyai-studio")

PERSONA = "Rory, the billing-line customer service agent for Acme Energy, a regulated gas and electric utility."

AGENT_PROMPT = load_agent_prompt()


def rules_text() -> str:
    """The shared prompt, the frozen-clock date context, and the tool bindings.

    A global function only reaches the model when the rules reference it as
    ``{{fn:name}}``; without these lines the project has nothing binding the
    tools to the conversation. This is the one addition to the shared prompt.
    """
    bindings = "\n".join(f"- {{{{fn:{name}}}}}" for name in TOOL_NAMES)
    return (
        f"{AGENT_PROMPT}\n\n{today_context()}\n\n"
        "# Tools\n"
        "Use these functions for every account, bill and payment operation — never invent details:\n"
        f"{bindings}\n"
    )


def _run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    logger.info(f"[studio] $ {' '.join(args)}")
    t0 = time.monotonic()
    proc = subprocess.run(
        args,
        cwd=cwd,
        # The ADK's generated code trips Python 3.14 SyntaxWarnings, dozens of
        # lines per command; this is the ADK docs' own suppression.
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
        capture_output=True,
        text=True,
    )
    for stream, text in (("out", proc.stdout), ("err", proc.stderr)):
        for line in text.splitlines():
            logger.info(f"[studio] {stream}| {line}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"'{' '.join(args)}' failed with exit code {proc.returncode}")
    logger.info(f"[studio] exit {proc.returncode} in {time.monotonic() - t0:.1f}s")
    return proc


def _merge_yaml(path: Path, **fields) -> None:
    """Update fields of a platform-provisioned file the ADK only lets us edit, not author."""
    config = yaml.safe_load(path.read_text()) or {}
    config.update(fields)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def overlay(project_dir: Path, tool_webhook_url: str) -> None:
    settings = project_dir / "agent_settings"
    (settings / "persona.txt").write_text(PERSONA)
    (settings / "rules.txt").write_text(rules_text())

    _merge_yaml(
        project_dir / "voice" / "configuration.yaml",
        greeting={"welcome_message": GREETING, "language_code": "en-US"},
    )
    # Barge-in is on for every other candidate; Agent Studio defaults it off.
    _merge_yaml(project_dir / "voice" / "speech_recognition" / "asr_settings.yaml", barge_in=True)

    functions = project_dir / "functions"
    functions.mkdir(exist_ok=True)
    for name, source in render_functions(tool_webhook_url).items():
        (functions / f"{name}.py").write_text(source)


def _json_result(proc: subprocess.CompletedProcess, command: str) -> dict:
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{command} produced no JSON result (exit code {proc.returncode})")


def provision(tool_webhook_url: str) -> None:
    """Push Rory onto the Agent Studio project with this boot's webhook URL and promote it live.

    Blocking (subprocess + network); run it via ``asyncio.to_thread``. Auth is
    the ``poly`` CLI's own, ``POLY_ADK_KEY`` from the environment.
    """
    account_id = os.environ["POLYAI_ACCOUNT_ID"]
    project_id = os.environ["POLYAI_PROJECT_ID"]
    if BASE_PATH.exists():
        shutil.rmtree(BASE_PATH)
    _run([
        "poly", "init",
        "--region", REGION,
        "--account_id", account_id,
        "--project_id", project_id,
        "--base-path", str(BASE_PATH),
    ])

    # poly init creates <account>/<project> under the base path, but the
    # platform may normalize either id, so locate the one project it created.
    candidates = [p.parent for p in BASE_PATH.glob("*/*/project.yaml")]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one initialized project under {BASE_PATH}, found {candidates}")
    project_dir = candidates[0]

    overlay(project_dir, tool_webhook_url)

    # A rejected push exits 0 in the plain CLI; only --json reports failure
    # dependably.
    push = _json_result(_run(["poly", "push", "--json"], cwd=project_dir, check=False), "poly push")
    if not push.get("success") and push.get("message") != "No changes detected":
        raise RuntimeError(f"poly push failed: {push.get('message') or push}")

    message = "rory-polyai: bind tool webhooks for this boot"

    # push lands on a working branch, so the sandbox deployment still holds
    # the previous configuration until that branch is merged. Skipping the
    # merge fails silently: the live agent keeps the old rules with no tools
    # bound, answers plausibly, and never calls anything.
    if push.get("new_branch_id"):
        merge = _json_result(
            _run(["poly", "branch", "merge", message, "--json", "--force"], cwd=project_dir, check=False),
            "poly branch merge",
        )
        if not merge.get("success"):
            raise RuntimeError(f"poly branch merge failed: {merge.get('message') or merge}")

    # Promotion to live must pass through pre-release: --to live resolves
    # --from against the pre-release deployment list, and a fresh sandbox
    # hash is never there yet.
    for source, target in (("sandbox", "pre-release"), ("pre-release", "live")):
        _run([
            "poly", "deployments", "promote",
            "--from", source, "--to", target, "--force", "-m", message,
        ], cwd=project_dir)
    logger.info(f"[studio] project {account_id}/{project_id} pushed and promoted to live")
