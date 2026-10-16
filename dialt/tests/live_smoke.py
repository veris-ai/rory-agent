"""Opt-in live audio proof: real /voice ASGI + production Dialt + local HTTP fixtures.

Run from the repo root (billable Dialt inference; no real financial writes):
  uv run --no-sync python dialt/tests/live_smoke.py --output /tmp/dialt-live.json

No instructions, tools, dispatcher or vendor clients are mocked. A transparent
WebSocket relay records an allowlist of synthetic conversation evidence, never
credentials, resume tokens, the voice roster or other server internals.
"""

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

import httpx
import uvicorn
import websockets
from websockets.asyncio.server import serve

from live_backends import Backends

PRODUCTION = "wss://api.dialt.com/v1/realtime"
PHRASES = {
    "wrong_identity": "My name is Jane Smith and my postal code is nine nine nine nine nine. Please check my bill.",
    "account": "My account number is forty four seventeen, dash, eighty eight two thirty one.",
    "identity": "Let me correct the postal code. My name is Jane Smith and my mailing postal code is six zero six one four. Please tell me my latest bill amount.",
    "paperless": "Please turn on paperless billing for my account now.",
    "confirm": "Yes, I confirm. Please go ahead with that now.",
    "decline": "Please pay my latest bill of forty two dollars and seventeen cents, in full, now, using the Visa card ending four two four two. I authorize this payment.",
    "pay": "Please try the payment again now. I authorize paying forty two dollars and seventeen cents with my Visa ending four two four two.",
    "explain": "Please read out all the charges on my latest bill and explain each charge in detail.",
    "enable": "Stop. There is a television speaking in the background. Please turn on background speaker suppression now.",
    "disable": "Please turn off background speaker suppression now.",
}


@contextmanager
def server(app):
    """A finite test server with OS-assigned port and guaranteed shutdown."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="websockets"))
    thread = threading.Thread(target=instance.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    try:
        while not instance.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Local fixture/ASGI server did not start")
            time.sleep(.02)
        yield port
    finally:
        instance.should_exit = True
        thread.join(15)
        sock.close()
        assert not thread.is_alive(), "Local test server did not stop"


async def synthesize(text, cache):
    path = cache / (hashlib.sha256(text.encode()).hexdigest() + ".pcm")
    if path.exists():
        return path.read_bytes()
    audio = bytearray()
    async with websockets.connect(PRODUCTION, additional_headers={
        "Authorization": "Bearer " + os.environ["DIALT_API_KEY"],
    }) as ws:
        await ws.send(json.dumps({
            "type": "start", "session_id": str(uuid.uuid4()),
            "audio": {"sr": 16000, "output_sr": 24000},
            "mode": {"kind": "dialt", "instructions": "Read the greeting as written.",
                     "greeting": text, "voice": "us_male", "web_search": False},
        }))
        async with asyncio.timeout(60):
            async for raw in ws:
                if isinstance(raw, bytes):
                    audio.extend(raw)
                else:
                    event = json.loads(raw)
                    if event["type"] == "error":
                        raise RuntimeError("Dialt fixture speech generation failed")
                    if event["type"] == "ready":
                        assert event["audio"] == {"output_encoding": "pcm16", "output_sr": 24000}
                    if event["type"] == "done":
                        break
    assert len(audio) >= 48000, "Fixture speech was missing or truncated"
    path.write_bytes(audio)
    return bytes(audio)


class Evidence:
    def __init__(self, brain):
        self.brain = brain
        self.stage = "greeting"
        self.events = []
        self.started = time.monotonic()
        self.audio_bytes = 0
        self.last_audio = 0
        self.tool_names = {}
        self.injections = {}

    def record(self, direction, raw):
        if isinstance(raw, bytes):
            return
        data = json.loads(raw)
        kind = data.get("type")
        entry = {"type": kind, "stage": self.stage, "seconds": round(time.monotonic() - self.started, 3)}
        if kind == "ready":
            entry.update(audio=data.get("audio"), suppression=data.get("background_speaker_suppression"))
        elif kind in {"asr", "asr_correction", "utterance"}:
            entry["text"] = data.get("text")
            if "input_source" in data:
                entry["input_source"] = data["input_source"]
        elif kind == "inject_context_ack":
            self.injections[data["message_id"]] = data["accepted"]
            entry["accepted"] = data["accepted"]
        elif kind == "tool_call":
            self.tool_names[data["id"]] = data["name"]
            entry.update(name=data["name"], args=data["args"])
        elif kind == "tool_result":
            entry.update(name=self.tool_names.get(data["id"]), content=data["content"], outcome=data.get("outcome"), verified=data.get("verified"))
        elif kind == "background_speaker_suppression_updated":
            entry.update({key: data.get(key) for key in ("enabled", "agent_control", "source", "revision")})
        elif kind == "client_event" and data.get("event") == "playback_stopped":
            entry.update({key: data.get(key) for key in ("event", "remaining_ms", "discarded_ms", "barge_seq")})
        elif kind in {"interrupted", "canceled", "tool_cancel", "done", "response_done"}:
            entry.update({key: data[key] for key in ("clear", "status") if key in data})
        elif kind == "error":
            entry.update(code=data.get("code"), retryable=data.get("retryable"))
        else:
            return
        entry["direction"] = direction
        self.events.append(entry)

    async def until(self, predicate, description, timeout=60):
        async with asyncio.timeout(timeout):
            while not predicate():
                if any(e["type"] == "error" for e in self.events):
                    raise RuntimeError("Dialt sent an error; see sanitized evidence")
                await asyncio.sleep(.05)
        print(self.brain, description, "PASS", flush=True)

    async def quiet(self):
        await self.until(lambda: self.audio_bytes and time.monotonic() - self.last_audio > 1.5,
                         "playback drained")

    def result(self, name, predicate):
        return any(e["stage"] == self.stage and e["type"] == "tool_result" and e["name"] == name
                   and predicate(e) for e in self.events)


async def run_tier(brain, speech, evidence, typed_business):
    backend = Backends()
    evidence.backend = backend
    async def relay(client):
        async with websockets.connect(PRODUCTION, additional_headers={
            "Authorization": client.request.headers["Authorization"],
        }) as remote:
            evidence.remote = remote
            async def forward(source, destination, direction):
                async for raw in source:
                    evidence.record(direction, raw)
                    try:
                        await destination.send(raw)
                    except websockets.ConnectionClosedOK:
                        if direction == "sent":
                            # Let the reader drain remote output before the
                            # relay propagates a clean close to the adapter.
                            await asyncio.Future()
                        return
            tasks = [asyncio.create_task(forward(client, remote, "sent")),
                     asyncio.create_task(forward(remote, client, "received"))]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    with server(backend.app) as backend_port:
        # Always overwrite vendor settings. No caller-supplied financial
        # credentials or remote backend URLs can be used by this harness.
        os.environ.update(STRIPE_API_KEY="sk_test_disposable_fixture",
                          STRIPE_API_BASE=f"http://127.0.0.1:{backend_port}",
                          FINERACT_API_BASE=f"http://127.0.0.1:{backend_port}",
                          FINERACT_USER="fixture", FINERACT_PASSWORD="fixture",
                          RORY_REFERENCE_TIME="2026-10-16T12:00:00Z")
        from rory_dialt import agent
        from rory_dialt.web import app
        from rory_tools.session import clients
        # The same process runs both tiers, each with fresh HTTP clients.
        if clients._fineract:
            clients._fineract._http.close()
        clients._stripe = clients._fineract = None
        agent.DIALT_BRAIN = brain
        async with serve(relay, "127.0.0.1", 0) as proxy:
            agent.DIALT_URL = f"ws://127.0.0.1:{proxy.sockets[0].getsockname()[1]}"
            with server(app) as port:
                async with httpx.AsyncClient() as http:
                    assert (await http.get(f"http://127.0.0.1:{port}/health")).json() == {"status": "ok"}
                async with websockets.connect(f"ws://127.0.0.1:{port}/voice") as actor:
                    microphone = bytearray()
                    async def uplink():
                        while True:
                            frame = bytes(microphone[:960]) if microphone else b"\0" * 960
                            del microphone[:960]
                            await actor.send(frame)
                            await asyncio.sleep(.02)
                    async def downlink():
                        async for frame in actor:
                            assert isinstance(frame, bytes) and len(frame) % 2 == 0 and len(frame) <= 960
                            evidence.audio_bytes += len(frame)
                            evidence.last_audio = time.monotonic()
                    async def say(phrase, stage=None):
                        evidence.stage = stage or phrase
                        if typed_business and phrase not in {"enable", "disable"}:
                            message_id = str(uuid.uuid4())
                            await evidence.remote.send(json.dumps({
                                "type": "inject_context", "role": "user", "reply": True,
                                "message_id": message_id, "text": PHRASES[phrase],
                            }))
                            await evidence.until(lambda: message_id in evidence.injections, phrase + " typed acknowledgement")
                            assert evidence.injections[message_id], "Typed input rejected (busy or invalid)"
                            return
                        microphone.extend(speech[phrase])
                        await evidence.until(lambda: not microphone, phrase + " microphone sent")
                    async def action(phrase, tool, check):
                        await say(phrase)
                        await evidence.until(lambda: evidence.result(tool, check) or any(
                            e["stage"] == phrase and e["type"] == "response_done" and e.get("status") == "completed"
                            for e in evidence.events), phrase + " answered")
                        if not evidence.result(tool, check):
                            # A real caller must answer confirmation questions.
                            # One explicit confirmation is allowed, never an
                            # automatic replay after an uncertain write.
                            assert not any(e["stage"] == phrase and e["type"] == "tool_call" and e["name"] == tool for e in evidence.events)
                            await evidence.quiet()
                            await say("confirm", phrase)
                        await evidence.until(lambda: evidence.result(tool, check), phrase + " result")
                    tasks = [asyncio.create_task(uplink()), asyncio.create_task(downlink())]
                    try:
                        await evidence.quiet()
                        assert evidence.events[0]["suppression"]["agent_control"] is True
                        assert evidence.events[0]["suppression"]["enabled"] is False
                        await say("wrong_identity")
                        await evidence.until(lambda: any(e["stage"] == "wrong_identity" and e["type"] == "utterance" for e in evidence.events), "request for missing identity")
                        await evidence.quiet()
                        await say("account", "wrong_identity")
                        await evidence.until(lambda: evidence.result("verify_caller", lambda e: e["content"].get("verified") is False), "wrong identity denied")
                        await asyncio.sleep(2)
                        await evidence.quiet()
                        await say("identity")
                        await evidence.until(lambda: evidence.result("verify_caller", lambda e: e["content"].get("verified") is True), "identity verified")
                        await evidence.until(lambda: evidence.result("get_account", lambda e: e["outcome"] == "succeeded"), "account lookup")
                        await asyncio.sleep(2)
                        await evidence.quiet()
                        await action("paperless", "set_paperless_billing", lambda e: e["content"].get("paperless_billing") is True)
                        assert backend.customer["metadata"]["paperless"] == "true"
                        await asyncio.sleep(2)
                        await evidence.quiet()
                        for stage, outcome in (("decline", "unknown"), ("pay", "succeeded")):
                            await action(stage, "make_payment", lambda e: e["outcome"] == outcome)
                            if stage == "decline":
                                assert backend.payments == ["declined"]
                                assert backend.invoice["amount_remaining"] == 4217
                                assert backend.invoice["status"] == "open"
                                assert evidence.result("make_payment", lambda e: e["verified"] is False and "error" in e["content"])
                            else:
                                assert evidence.result("make_payment", lambda e: e["verified"] is True and e["content"].get("amount_paid_cents") == 4217)
                            await asyncio.sleep(2)
                            await evidence.quiet()
                        assert backend.payments == ["declined", "paid"]
                        assert backend.invoice["status"] == "paid" and backend.invoice["amount_remaining"] == 0
                        # Start caller speech during a real streamed answer.
                        await say("explain")
                        previous = evidence.audio_bytes
                        await evidence.until(lambda: evidence.audio_bytes > previous + 24000, "explanation audio started")
                        await say("enable")
                        await evidence.until(lambda: any(e["stage"] == "enable" and e["type"] == "background_speaker_suppression_updated" and e.get("enabled") is True and e.get("source") == "agent" for e in evidence.events), "agent suppression enabled")
                        await asyncio.sleep(2)
                        await evidence.quiet()
                        await say("disable")
                        await evidence.until(lambda: any(e["stage"] == "disable" and e["type"] == "background_speaker_suppression_updated" and e.get("enabled") is False and e.get("source") == "agent" for e in evidence.events), "agent suppression disabled")
                        await asyncio.sleep(2)
                        await evidence.quiet()
                        assert any(e["type"] == "interrupted" for e in evidence.events), "No live interruption observed"
                        assert any(e.get("event") == "playback_stopped" for e in evidence.events)
                        assert "set_background_speaker_suppression" not in evidence.tool_names.values()
                        for task in tasks:
                            assert not task.done(), "Audio pump ended unexpectedly"
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
    return {"payments": backend.payments, "paperless": backend.customer["metadata"]["paperless"],
            "remaining_cents": backend.invoice["amount_remaining"], "http_requests": backend.requests}


async def main(args):
    if not os.environ.get("DIALT_API_KEY"):
        raise SystemExit("DIALT_API_KEY is required; inference is billable")
    cache = Path(args.audio_cache)
    cache.mkdir(parents=True, exist_ok=True)
    speech = {name: await synthesize(text, cache) for name, text in PHRASES.items()
              if not args.typed_business or name in {"enable", "disable"}}
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(Path("dialt").rglob("*.py"))},
              "scope": "production Dialt + actual ASGI /voice + disposable HTTP fixtures, not hosted Veris twins",
              "input": "typed business turns bypass ASR; spoken suppression/interruption" if args.typed_business else "all spoken through /voice",
              "runs": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for brain in args.brains:
            evidence = Evidence(brain)
            run = {"brain": brain, "passed": False, "events": evidence.events}
            report["runs"].append(run)
            try:
                run["backend"] = await run_tier(brain, speech, evidence, args.typed_business)
                run["passed"] = True
            except Exception as exc:
                run["failure"] = type(exc).__name__
                run["failed_stage"] = evidence.stage
                print(brain, evidence.stage, "FAILED", type(exc).__name__, flush=True)
            finally:
                run["output_pcm_bytes"] = evidence.audio_bytes
                backend = evidence.backend
                run["backend"] = {"payments": backend.payments, "paperless": backend.customer["metadata"]["paperless"],
                                  "remaining_cents": backend.invoice["amount_remaining"], "http_requests": backend.requests}
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("Sanitized evidence:", output, flush=True)
    if not all(run["passed"] for run in report["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-cache", default="/tmp/rory-dialt-live-audio")
    parser.add_argument("--brains", nargs="+", choices=["fast", "smart"], default=["fast", "smart"])
    parser.add_argument("--typed-business", action="store_true", help="Bypass business-turn ASR via acknowledged user text; not full voice E2E")
    asyncio.run(main(parser.parse_args()))
