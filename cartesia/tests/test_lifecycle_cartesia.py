"""What can be checked about the Cartesia bridge without a platform key.

The pod's tool budget — 16 webhook tools created once at startup, shared by
every call's agent, deleted at shutdown — the webhook's bearer check and
live-call lookup, and the per-call agent teardown. The audio legs and the
hosted turn model need a live session and are validated on the bench, not
here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from rory_cartesia import agent, web
from rory_tools import TOOL_NAMES
from rory_tools.prompt import GREETING, today_context

PUBLIC = "https://rory.example.com/hooks/1"


class FakeCartesia:
    """``api.cartesia.ai`` behind ``httpx.MockTransport``: stores tools and agents, logs every request."""

    def __init__(self, tool_limit: int | None = None):
        self.tool_limit = tool_limit
        self.requests: list[tuple[str, str]] = []
        self.tools: dict[str, dict] = {}
        self.agents: dict[str, dict] = {}
        self.agent_configs: list[dict] = []

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), base_url=agent.CARTESIA_API)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if request.method == "POST" and path == "/v1/agents/tools":
            if self.tool_limit is not None and len(self.tools) >= self.tool_limit:
                return httpx.Response(
                    400,
                    json={"error_code": "tool_limit_reached", "message": "An account may store at most 100 tools"},
                )
            tool_id = f"tool_{len(self.requests)}"
            self.tools[tool_id] = json.loads(request.content)
            return httpx.Response(201, json={"id": tool_id})
        if request.method == "POST" and path == "/v1/agents":
            config = json.loads(request.content)["config"]
            agent_id = f"agent_{len(self.agent_configs) + 1}"
            self.agents[agent_id] = config
            self.agent_configs.append(config)
            return httpx.Response(201, json={"id": agent_id})
        if request.method == "DELETE" and path.startswith("/v1/agents/tools/"):
            tool_id = path.rsplit("/", 1)[1]
            referenced = {ref["id"] for config in self.agents.values() for ref in config["tools"]}
            assert tool_id not in referenced, f"{tool_id} deleted while an agent still references it"
            del self.tools[tool_id]
            return httpx.Response(204)
        if request.method == "DELETE" and path.startswith("/v1/agents/"):
            del self.agents[path.rsplit("/", 1)[1]]
            return httpx.Response(204)
        raise AssertionError(f"unexpected {request.method} {path}")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "test")
    monkeypatch.setenv("STRIPE_API_KEY", "test")
    monkeypatch.setenv("PUBLIC_BASE_URL", PUBLIC)
    monkeypatch.setattr(agent, "LIVE_CALL", None)


def test_agent_config_carries_the_shared_prompt_greeting_and_the_pods_tools():
    config = agent.agent_config(["tool_a", "tool_b"])
    assert config["instructions"] == f"{agent.AGENT_PROMPT}\n\n{today_context()}"
    assert config["initial_message"] is GREETING
    assert config["tools"] == [{"id": "tool_a"}, {"id": "tool_b"}]
    assert config["audio"]["input"]["noise_suppression"] == "off"


def test_two_calls_share_the_tools_created_once_at_startup_and_deleted_at_shutdown(monkeypatch, configured):
    fake = FakeCartesia()
    monkeypatch.setattr(web, "cartesia_api", fake.client)
    monkeypatch.setattr(agent, "_relay", AsyncMock())
    with TestClient(web.app) as client:
        tool_ids = client.app.state.tool_ids
        assert len(tool_ids) == 16
        assert set(fake.tools) == set(tool_ids)
        # One URL and one bearer per pod: nothing in a tool names a call.
        assert {tool["api_schema"]["url"] for tool in fake.tools.values()} == {
            f"{PUBLIC}/tool/{name}" for name in TOOL_NAMES
        }
        assert {tool["api_schema"]["authentication"]["token"]["secret_value"] for tool in fake.tools.values()} == {
            client.app.state.secret
        }
        for _ in range(2):
            with client.websocket_connect("/voice"):
                pass
        assert fake.requests.count(("POST", "/v1/agents/tools")) == 16
        assert [config["tools"] for config in fake.agent_configs] == [[{"id": t} for t in tool_ids]] * 2
        assert fake.agents == {}
        assert set(fake.tools) == set(tool_ids)  # still there for the next call
    assert fake.tools == {}
    deletes = [path for method, path in fake.requests if method == "DELETE"]
    assert deletes[:2] == ["/v1/agents/agent_1", "/v1/agents/agent_2"]
    assert len(deletes) == 18


def test_startup_deletes_the_partial_set_and_fails_the_boot_when_the_account_is_full(monkeypatch, configured):
    fake = FakeCartesia(tool_limit=5)
    monkeypatch.setattr(web, "cartesia_api", fake.client)
    with pytest.raises(RuntimeError, match="tool_limit_reached"):
        with TestClient(web.app):
            pass
    assert fake.requests.count(("POST", "/v1/agents/tools")) == 16
    assert len([path for method, path in fake.requests if method == "DELETE"]) == 5
    assert fake.tools == {}


def test_webhook_checks_the_pods_bearer_then_runs_against_the_live_call(monkeypatch, configured):
    fake = FakeCartesia()
    monkeypatch.setattr(web, "cartesia_api", fake.client)
    monkeypatch.setattr(agent, "dispatch", lambda session, name, args: {"echo": name, **args})
    with TestClient(web.app) as client:
        bearer = {"Authorization": f"Bearer {client.app.state.secret}"}
        assert client.post("/tool/get_account", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.post("/tool/get_account", headers=bearer).status_code == 404
        monkeypatch.setattr(agent, "LIVE_CALL", agent.LiveCall())
        response = client.post("/tool/verify_caller", headers=bearer, json={"account_number": "4417-88231"})
        assert response.status_code == 200
        assert response.json() == {"echo": "verify_caller", "account_number": "4417-88231"}
        # A tool with no arguments arrives with an empty body.
        assert client.post("/tool/get_account", headers=bearer).json() == {"echo": "get_account"}


@pytest.mark.parametrize("failed", [False, True])
def test_call_creates_and_deletes_its_agent_and_propagates_relay_failure(monkeypatch, configured, failed):
    fake = FakeCartesia()

    async def relay(actor_ws, agent_id, api_key):
        assert agent.LIVE_CALL is not None  # registered before any audio flows
        assert agent_id in fake.agents
        if failed:
            raise RuntimeError("text frame")

    monkeypatch.setattr(agent, "_relay", relay)

    async def run():
        async with fake.client() as api:
            await agent.run_voice_ws_bot(SimpleNamespace(client=None), api, ["tool_a"])

    if failed:
        with pytest.raises(RuntimeError, match="text frame"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert fake.agent_configs[0]["tools"] == [{"id": "tool_a"}]
    assert fake.agents == {}
    assert agent.LIVE_CALL is None


def test_a_second_concurrent_call_is_refused_and_leaves_the_live_one_alone(monkeypatch):
    live = agent.LiveCall()
    monkeypatch.setattr(agent, "LIVE_CALL", live)
    with pytest.raises(RuntimeError, match="already on a call"):
        asyncio.run(agent.run_voice_ws_bot(SimpleNamespace(client=None), object(), ["tool_a"]))
    assert agent.LIVE_CALL is live


@pytest.mark.parametrize("failed,code", [(False, 1000), (True, 1011)])
def test_voice_handler_distinguishes_failure_from_normal_end(monkeypatch, failed, code):
    state = SimpleNamespace(api=object(), tool_ids=["tool_a"])
    ws = SimpleNamespace(client=None, client_state=WebSocketState.CONNECTED,
                         app=SimpleNamespace(state=state), accept=AsyncMock(), close=AsyncMock())
    runner = AsyncMock(side_effect=RuntimeError("broken pump") if failed else None)
    monkeypatch.setattr(web, "run_voice_ws_bot", runner)
    if failed:
        with pytest.raises(RuntimeError, match="broken pump"):
            asyncio.run(web.voice(ws))
    else:
        asyncio.run(web.voice(ws))
    assert runner.await_args.args[1:] == (state.api, ["tool_a"])
    assert ws.close.await_args.kwargs["code"] == code
