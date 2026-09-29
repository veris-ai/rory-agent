"""What can be checked about the PolyAI transport without a platform key.

The rendered Agent Studio functions (run for real against a stub of the ADK's
``_gen`` namespace), the project overlay, the ``poly`` command sequence the
provisioner drives, the webhook's per-conversation sessions, the paced
microphone track, and the signaling offer. The WebRTC media legs and the
hosted turn model need a live call and are validated on the bench, not here.
"""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from rory_tools import SCHEMAS, TOOL_NAMES
from rory_tools.prompt import GREETING, today_context
from rory_polyai import agent, studio, tools, web

WEBHOOK = "https://rory.trycloudflare.com/tool"


# ---------------------------------------------------------------------------
# Rendered functions
# ---------------------------------------------------------------------------

def _run_rendered(name: str, **kwargs):
    """Call the rendered function with ``conv`` and the given arguments; return what it POSTed and returned."""
    import sys
    import urllib.request

    posted = []
    schema = next(s for s in SCHEMAS if s.name == name)
    source = tools.render_function(schema, WEBHOOK)

    gen = types.ModuleType("_gen")
    gen.func_description = lambda *_: (lambda fn: fn)
    gen.func_parameter = lambda *_: (lambda fn: fn)
    gen.Conversation = object
    sys.modules["_gen"] = gen

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"ok": true}'

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout: (posted.append((req.full_url, json.loads(req.data), timeout)), FakeResponse())[1]
    try:
        namespace = {}
        exec(compile(source, f"{name}.py", "exec"), namespace)
        returned = namespace[name](SimpleNamespace(id="CONV-1"), **kwargs)
    finally:
        urllib.request.urlopen = real_urlopen
        del sys.modules["_gen"]
    return posted, returned


def test_rendered_function_posts_the_envelope_and_returns_the_reply_text():
    posted, returned = _run_rendered("get_bill", bill_id="in_123")
    assert posted == [(WEBHOOK, {"name": "get_bill", "args": {"bill_id": "in_123"}, "conversation_id": "CONV-1"}, 30)]
    assert returned == '{"ok": true}'


def test_rendered_function_keeps_typed_arguments_typed():
    posted, _ = _run_rendered("make_payment", bill_id="in_1", payment_method_id="pm_1", amount_cents=21407)
    assert posted[0][1]["args"] == {"bill_id": "in_1", "payment_method_id": "pm_1", "amount_cents": 21407}
    posted, _ = _run_rendered("set_paperless_billing", enabled=True)
    assert posted[0][1]["args"] == {"enabled": True}


def test_rendered_function_drops_an_optional_argument_the_model_left_empty():
    posted, _ = _run_rendered("modify_payment_arrangement", amount_cents=6875, payment_date="")
    assert posted[0][1]["args"] == {"amount_cents": 6875}
    posted, _ = _run_rendered("modify_payment_arrangement", amount_cents=6875, payment_date="2026-03-12")
    assert posted[0][1]["args"] == {"amount_cents": 6875, "payment_date": "2026-03-12"}


def test_rendered_function_with_no_arguments_takes_only_conv():
    posted, _ = _run_rendered("get_account")
    assert posted[0][1] == {"name": "get_account", "args": {}, "conversation_id": "CONV-1"}


def test_rendered_functions_use_only_the_standard_library():
    for source in tools.render_functions(WEBHOOK).values():
        imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        assert imports == ["from _gen import *  # <AUTO GENERATED>", "import json", "import urllib.request"]


# ---------------------------------------------------------------------------
# Project overlay and provisioning
# ---------------------------------------------------------------------------

def _pulled_project(root: Path) -> Path:
    """The files ``poly init`` leaves that the overlay edits rather than authors."""
    project = root / "ws-acme" / "PROJECT-rory"
    (project / "agent_settings").mkdir(parents=True)
    (project / "voice" / "speech_recognition").mkdir(parents=True)
    (project / "project.yaml").write_text("region: us-1\n")
    (project / "voice" / "configuration.yaml").write_text(
        "greeting:\n  welcome_message: Hello from the template\n  language_code: en-GB\nstyle_prompt:\n  prompt: Keep it short.\n"
    )
    (project / "voice" / "speech_recognition" / "asr_settings.yaml").write_text("barge_in: false\ninteraction_style: balanced\n")
    return project


def test_overlay_writes_the_shared_prompt_greeting_bindings_and_one_function_per_tool(tmp_path):
    project = _pulled_project(tmp_path)
    studio.overlay(project, WEBHOOK)

    assert (project / "agent_settings" / "persona.txt").read_text() == studio.PERSONA
    rules = (project / "agent_settings" / "rules.txt").read_text()
    assert rules.startswith(studio.AGENT_PROMPT + "\n\n" + today_context())
    for name in TOOL_NAMES:
        assert f"{{{{fn:{name}}}}}" in rules

    voice = yaml.safe_load((project / "voice" / "configuration.yaml").read_text())
    assert voice["greeting"] == {"welcome_message": GREETING, "language_code": "en-US"}
    assert voice["style_prompt"] == {"prompt": "Keep it short."}  # untouched platform field survives
    asr = yaml.safe_load((project / "voice" / "speech_recognition" / "asr_settings.yaml").read_text())
    assert asr == {"barge_in": True, "interaction_style": "balanced"}

    rendered = sorted(p.stem for p in (project / "functions").glob("*.py"))
    assert rendered == sorted(TOOL_NAMES)
    assert WEBHOOK in (project / "functions" / "verify_caller.py").read_text()


class _FakePoly:
    """Records the ``poly`` commands the provisioner runs and answers like the CLI."""

    def __init__(self, base_path: Path, push_result: dict):
        self.base_path = base_path
        self.push_result = push_result
        self.commands: list[list[str]] = []

    def __call__(self, args, cwd=None, check=True):
        self.commands.append(args)
        stdout = ""
        if args[:2] == ["poly", "init"]:
            _pulled_project(self.base_path)
        elif args[:2] == ["poly", "push"]:
            stdout = json.dumps(self.push_result)
        elif args[:3] == ["poly", "branch", "merge"]:
            stdout = json.dumps({"success": True})
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


@pytest.fixture
def studio_env(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYAI_ACCOUNT_ID", "ws-acme")
    monkeypatch.setenv("POLYAI_PROJECT_ID", "PROJECT-rory")
    monkeypatch.setattr(studio, "BASE_PATH", tmp_path / "studio")
    return tmp_path / "studio"


def test_provision_inits_pushes_merges_and_promotes_through_pre_release(monkeypatch, studio_env):
    poly = _FakePoly(studio_env, {"success": True, "new_branch_id": "ADK-1"})
    monkeypatch.setattr(studio, "_run", poly)
    studio.provision(WEBHOOK)

    heads = [cmd[1:3] for cmd in poly.commands]
    assert heads == [
        ["init", "--region"],
        ["push", "--json"],
        ["branch", "merge"],
        ["deployments", "promote"],
        ["deployments", "promote"],
    ]
    assert poly.commands[0][2:] == ["--region", "us-1", "--account_id", "ws-acme", "--project_id", "PROJECT-rory", "--base-path", str(studio_env)]
    assert poly.commands[3][3:7] == ["--from", "sandbox", "--to", "pre-release"]
    assert poly.commands[4][3:7] == ["--from", "pre-release", "--to", "live"]
    # The overlay landed in the project poly init created.
    assert (studio_env / "ws-acme" / "PROJECT-rory" / "functions" / "get_account.py").exists()


def test_provision_skips_the_merge_when_nothing_changed(monkeypatch, studio_env):
    poly = _FakePoly(studio_env, {"success": False, "message": "No changes detected"})
    monkeypatch.setattr(studio, "_run", poly)
    studio.provision(WEBHOOK)
    assert [cmd[1] for cmd in poly.commands] == ["init", "push", "deployments", "deployments"]


def test_provision_fails_loudly_on_a_rejected_push(monkeypatch, studio_env):
    poly = _FakePoly(studio_env, {"success": False, "message": "validation failed: rules.txt"})
    monkeypatch.setattr(studio, "_run", poly)
    with pytest.raises(RuntimeError, match="validation failed"):
        studio.provision(WEBHOOK)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def test_webhook_keeps_one_session_per_conversation(monkeypatch):
    monkeypatch.setattr(tools, "LIVE_CALLS", {})
    seen = []

    def dispatch(session, name, args):
        if name == "verify_caller":
            session.account = {"customer_id": "cus_verified"}
        seen.append((name, session.account))
        return {"ok": name}

    monkeypatch.setattr(tools, "dispatch", dispatch)
    client = TestClient(web.app)  # no context manager: no lifespan, no tunnel, no provisioning
    assert client.post("/tool", json={"name": "verify_caller", "args": {"account_number": "4417-88231"}, "conversation_id": "CONV-1"}).json() == {"ok": "verify_caller"}
    assert client.post("/tool", json={"name": "get_account", "args": {}, "conversation_id": "CONV-1"}).json() == {"ok": "get_account"}
    assert client.post("/tool", json={"name": "get_account", "args": {}, "conversation_id": "CONV-2"}).json() == {"ok": "get_account"}
    assert seen == [
        ("verify_caller", {"customer_id": "cus_verified"}),
        ("get_account", {"customer_id": "cus_verified"}),
        ("get_account", None),  # a different conversation starts unverified
    ]
    assert set(tools.LIVE_CALLS) == {"CONV-1", "CONV-2"}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

def test_mic_track_frames_buffered_audio_and_fills_gaps_with_silence():
    track = agent.ActorAudioTrack()
    audio = (bytes(range(256)) * 4)[: agent.BYTES_PER_FRAME]
    track.push(audio)

    async def two_frames():
        return await track.recv(), await track.recv()

    first, second = asyncio.run(two_frames())
    assert bytes(first.planes[0])[: agent.BYTES_PER_FRAME] == audio
    assert first.sample_rate == agent.SAMPLE_RATE_HZ and first.samples == agent.SAMPLES_PER_FRAME
    assert bytes(second.planes[0])[: agent.BYTES_PER_FRAME] == b"\x00" * agent.BYTES_PER_FRAME
    assert second.pts == first.pts + agent.SAMPLES_PER_FRAME


def test_offer_carries_the_connector_token_and_project(monkeypatch):
    monkeypatch.setenv("POLYAI_AUTH_TOKEN", "tok")
    monkeypatch.setenv("POLYAI_ACCOUNT_ID", "ws-acme")
    monkeypatch.setenv("POLYAI_PROJECT_ID", "PROJECT-rory")
    offer = agent.offer_message("v=0 ...", "rory-1")
    assert offer == {
        "type": "offer",
        "sessionId": "",
        "data": {"type": "offer", "sdp": "v=0 ..."},
        "authToken": "tok",
        "callSid": "rory-1",
        "accountId": "ws-acme",
        "projectId": "PROJECT-rory",
    }


@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_distinguishes_failure_from_normal_end(monkeypatch, failed, code):
    ws = SimpleNamespace(client=None, client_state=WebSocketState.CONNECTED, accept=AsyncMock(), close=AsyncMock())
    runner = AsyncMock(side_effect=RuntimeError("broken pump") if failed else None)
    monkeypatch.setattr(web, "run_voice_ws_bot", runner)
    if failed:
        with pytest.raises(RuntimeError, match="broken pump"):
            asyncio.run(web.voice(ws))
    else:
        asyncio.run(web.voice(ws))
    assert ws.close.await_args.kwargs["code"] == code
