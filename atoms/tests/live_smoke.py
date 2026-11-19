"""Opt-in live audio proof: real /voice ASGI + production Atoms + local HTTP fixtures.

Run from the repo root (billable Atoms and TTS inference; no real financial
writes):

    uv run --no-sync python atoms/tests/live_smoke.py --output /tmp/atoms-live.json

Needs ``ATOMS_API_KEY`` and ``ATOMS_AGENT_ID`` pointed at a provisioned agent
(prompt and turn-taking settings already configured on it — see
``atoms/README.md``) plus ``SMALLEST_API_KEY`` to synthesize the test
phrases via Lightning TTS. Rory's 16 tools need nothing provisioned ahead of
time: ``rory_atoms.agent`` declares them itself over ``session.update`` on
every call.

Only raw PCM16 audio crosses ``/voice``, not typed JSON events, so there is
nothing to relay-and-record. Tool-call evidence instead comes from
monkeypatching ``rory_atoms.agent.dispatch`` to record every call without
changing its behavior.
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

import httpx
import uvicorn
import websockets

from live_backends import Backends

TTS_URL = "https://api.smallest.ai/waves/v1/tts"
FRAME_BYTES = 960  # 20 ms @ 24 kHz mono PCM16, matching the actor's own chunking
PHRASES = {
    "wrong_identity": "My name is Jane Smith and my postal code is nine nine nine nine nine. Please check my bill.",
    "account": "My account number is forty four seventeen, dash, eighty eight two thirty one.",
    "identity": "Let me correct the postal code. My name is Jane Smith and my mailing postal code is six zero six one four. Please tell me my latest bill amount.",
    "paperless": "Please turn on paperless billing for my account now.",
    "confirm": "Yes, I confirm. Please go ahead with that now.",
    "decline": "Please pay my latest bill of forty two dollars and seventeen cents, in full, now, using the Visa card ending four two four two. I authorize this payment.",
    "pay": "Please try the payment again now. I authorize paying forty two dollars and seventeen cents with my Visa ending four two four two.",
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


def synthesize(text, cache):
    path = cache / (hashlib.sha256(text.encode()).hexdigest() + ".pcm")
    if path.exists():
        return path.read_bytes()
    resp = httpx.post(
        TTS_URL,
        headers={"Authorization": f"Bearer {os.environ['SMALLEST_API_KEY']}", "Accept": "audio/wav"},
        json={"text": text, "voice_id": "magnus", "sample_rate": 24000, "output_format": "pcm"},
        timeout=60,
    )
    resp.raise_for_status()
    audio = resp.content
    assert len(audio) >= 24000, "Fixture speech was missing or truncated"
    path.write_bytes(audio)
    return audio


class Evidence:
    """Records every dispatched tool call (via the monkeypatch) and the audio
    actually heard back, since /voice carries nothing else to observe.

    No silence-gap heuristic here, confirmed against a real call: Atoms
    streams ``output_audio.delta`` roughly once a second continuously, turn
    or no turn, so "time since the last audio byte" never ages past ~1s and
    can never signal a quiet agent. Pacing between utterances is a fixed
    sleep instead; the real correctness gate is waiting for the expected
    tool call to actually show up in ``calls``.
    """

    def __init__(self):
        self.stage = "greeting"
        self.calls = []  # (stage, name, args, result)
        self.started = time.monotonic()
        self.audio_bytes = 0
        self.last_audio = 0.0

    def record_call(self, name, args, result):
        self.calls.append({"stage": self.stage, "name": name, "args": args, "result": result})

    def result(self, name, predicate):
        return any(c["stage"] == self.stage and c["name"] == name and predicate(c["result"]) for c in self.calls)

    async def until(self, predicate, description, timeout=60):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(.05)
        print(description, "PASS", flush=True)


async def run(speech, evidence):
    backend = Backends()

    with server(backend.app) as backend_port:
        # Always overwrite vendor settings. No caller-supplied financial
        # credentials or remote backend URLs can be used by this harness.
        os.environ.update(STRIPE_API_KEY="sk_test_disposable_fixture",
                          STRIPE_API_BASE=f"http://127.0.0.1:{backend_port}",
                          FINERACT_API_BASE=f"http://127.0.0.1:{backend_port}",
                          FINERACT_USER="fixture", FINERACT_PASSWORD="fixture",
                          RORY_REFERENCE_TIME="2026-10-16T12:00:00Z")
        from rory_atoms import agent
        from rory_atoms.web import app
        from rory_tools.session import clients

        if clients._fineract:
            clients._fineract._http.close()
        clients._stripe = clients._fineract = None

        real_dispatch = agent.dispatch

        def recording_dispatch(session_state, name, args):
            result = real_dispatch(session_state, name, args)
            evidence.record_call(name, args, result)
            return result

        agent.dispatch = recording_dispatch
        try:
            with server(app) as port:
                async with httpx.AsyncClient() as http:
                    assert (await http.get(f"http://127.0.0.1:{port}/health")).json() == {"status": "ok"}
                async with websockets.connect(f"ws://127.0.0.1:{port}/voice") as actor:
                    microphone = bytearray()

                    async def uplink():
                        while True:
                            frame = bytes(microphone[:FRAME_BYTES]) if microphone else b"\0" * FRAME_BYTES
                            del microphone[:FRAME_BYTES]
                            await actor.send(frame)
                            await asyncio.sleep(.02)

                    async def downlink():
                        async for frame in actor:
                            assert isinstance(frame, bytes)
                            evidence.audio_bytes += len(frame)
                            evidence.last_audio = time.monotonic()

                    async def say(phrase, stage=None):
                        evidence.stage = stage or phrase
                        microphone.extend(speech[phrase])
                        await evidence.until(lambda: not microphone, phrase + " microphone sent")

                    async def action(phrase, tool, check):
                        await say(phrase)
                        try:
                            await evidence.until(lambda: evidence.result(tool, check), phrase + " result", timeout=15)
                        except TimeoutError:
                            # The prompt requires a read-back and a yes before
                            # most writes (paperless, a payment) — a real
                            # caller hears that question and answers it.
                            await asyncio.sleep(PACING_S)
                            await say("confirm", phrase)
                            await evidence.until(lambda: evidence.result(tool, check), phrase + " result", timeout=60)

                    # Fixed pacing between utterances, sized from a real call
                    # (greeting ~4s, a short follow-up question ~4-5s): this
                    # is scheduling, not a correctness check. Every stage's
                    # real gate is the tool-call result wait below it.
                    PACING_S = 6

                    tasks = [asyncio.create_task(uplink()), asyncio.create_task(downlink())]
                    try:
                        await evidence.until(lambda: evidence.audio_bytes > 0, "greeting started", timeout=30)
                        await asyncio.sleep(PACING_S)
                        await say("wrong_identity")
                        await asyncio.sleep(PACING_S)
                        await say("account", "wrong_identity")
                        await evidence.until(
                            lambda: evidence.result("verify_caller", lambda r: r.get("verified") is False),
                            "wrong identity denied", timeout=60,
                        )
                        await asyncio.sleep(PACING_S)
                        await say("identity")
                        await evidence.until(
                            lambda: evidence.result("verify_caller", lambda r: r.get("verified") is True),
                            "identity verified", timeout=60,
                        )
                        await evidence.until(
                            lambda: evidence.result("get_account", lambda r: "error" not in r),
                            "account lookup", timeout=60,
                        )
                        await asyncio.sleep(PACING_S)
                        await action("paperless", "set_paperless_billing", lambda r: r.get("paperless_billing") is True)
                        assert backend.customer["metadata"]["paperless"] == "true"
                        await asyncio.sleep(PACING_S)
                        # A declined card comes back as {"error": ...} with no
                        # "paid" key at all (dispatch() converts VENDOR_ERRORS
                        # that way) — not {"paid": False}.
                        for stage, check in (
                            ("decline", lambda r: "error" in r),
                            ("pay", lambda r: r.get("paid") is True),
                        ):
                            await action(stage, "make_payment", check)
                            await asyncio.sleep(PACING_S)
                        assert backend.payments == ["declined", "paid"]
                        assert backend.invoice["status"] == "paid" and backend.invoice["amount_remaining"] == 0
                        for task in tasks:
                            assert not task.done(), "Audio pump ended unexpectedly"
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            agent.dispatch = real_dispatch
    return {"payments": backend.payments, "paperless": backend.customer["metadata"]["paperless"],
            "remaining_cents": backend.invoice["amount_remaining"], "http_requests": backend.requests}


async def main(args):
    missing = [name for name in ("SMALLEST_API_KEY", "ATOMS_API_KEY", "ATOMS_AGENT_ID") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required; inference and TTS are billable")
    cache = Path(args.audio_cache)
    cache.mkdir(parents=True, exist_ok=True)
    speech = {name: synthesize(text, cache) for name, text in PHRASES.items()}

    evidence = Evidence()
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(Path("atoms").rglob("*.py"))},
              "scope": "production Atoms + actual ASGI /voice + disposable HTTP fixtures, not hosted Veris twins",
              "passed": False, "calls": evidence.calls}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["backend"] = await run(speech, evidence)
        report["passed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__
        report["failed_stage"] = evidence.stage
        print(evidence.stage, "FAILED", type(exc).__name__, flush=True)
        raise
    finally:
        report["output_pcm_bytes"] = evidence.audio_bytes
        report["calls"] = evidence.calls
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("Evidence:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-cache", default="/tmp/rory-atoms-live-audio")
    asyncio.run(main(parser.parse_args()))
