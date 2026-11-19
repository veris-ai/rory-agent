"""Prism tool rounds against a fake session.

The parity tests check the declared surface. These drive the tool loop the way
Rime's tools guides run it: each call runs on its own task through the
dispatcher as Prism makes it (writes one at a time), each result is submitted,
and the reply that made the calls is continued once it has completed and all of
them are in. When Prism refuses the continuation, a nudge reports each result,
with no guessing about what Prism has said since. A round whose reply did not
complete is dropped. A slow lookup gets still-working lines and a deadline. No
call is ever run twice.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from rimelabs_sdk import RimeInputError
from rimelabs_sdk.realtime import (
    RealtimeAdmissionTimeout,
    RealtimeFault,
    ResponseEnded,
    ResponseRef,
    RimeRealtimeError,
    ToolCall,
    ToolCallRef,
)

from rory_rime_prism import agent


class FakeSession:
    """Records what the loop sends.

    ``refuse`` maps a response id to the error its continuation raises.
    ``refuse_nudge`` is the error every nudge raises.
    """

    def __init__(self, refuse: dict[str, Exception] | None = None, refuse_nudge: Exception | None = None):
        self.refuse = refuse or {}
        self.refuse_nudge = refuse_nudge
        self.log: list[tuple] = []
        self.idle = asyncio.Event()  # Prism takes a nudge only once it is idle
        self.idle.set()

    async def submit_tool_result(self, call, output):
        self.log.append(("result", call.call_id, json.loads(output)))

    async def continue_reply(self, parent):
        self.log.append(("continue", parent.response_id))
        if parent.response_id in self.refuse:
            raise self.refuse[parent.response_id]

    async def request_reply(self, *, instruction=None, tool_call=None):
        await self.idle.wait()
        self.log.append(("line", instruction) if tool_call is None else ("report", tool_call.call_id))
        if self.refuse_nudge is not None:
            raise self.refuse_nudge


def _call(response_id: str, call_id: str, name: str = "get_account") -> ToolCall:
    ref = ToolCallRef(session_id="sess", response_id=response_id, call_id=call_id)
    return ToolCall(call=ref, item_id=f"item_{call_id}", name=name, arguments={"n": call_id})


def _ended(response_id: str, status: str = "completed") -> ResponseEnded:
    return ResponseEnded(response=ResponseRef(session_id="sess", response_id=response_id), status=status)


def _superseded(code: str = "tool_continuation_unavailable") -> RimeRealtimeError:
    return RimeRealtimeError(RealtimeFault(code=code, message="round closed", scope="tool_roundtrip"))


class FakeDispatch:
    """The shared dispatcher, recording which calls ran and how many at once.

    Each call takes ``lookup_s``."""

    def __init__(self, lookup_s: float) -> None:
        self.lookup_s = lookup_s
        self.ran: list[str] = []
        self.running: dict[str, int] = {"write": 0, "read": 0}
        self.most: dict[str, int] = {"write": 0, "read": 0}
        self._lock = threading.Lock()

    def __call__(self, _session, name, args):
        kind = "write" if name in agent.WRITE_TOOLS else "read"
        with self._lock:
            self.ran.append(args["n"])
            self.running[kind] += 1
            self.most[kind] = max(self.most[kind], self.running[kind])
        time.sleep(self.lookup_s)
        with self._lock:
            self.running[kind] -= 1
        return {"tool": name, "ok": True}


async def _settle(loop: asyncio.Task, rounds: agent.ToolRounds) -> None:
    """Wait until every event is taken and every task the loop started is done."""
    deadline = time.monotonic() + 2
    while not loop.done() and time.monotonic() < deadline:
        if rounds._events.empty() and not asyncio.all_tasks() - {asyncio.current_task(), loop}:
            return
        await asyncio.sleep(0.01)


def _drive(session, steps, monkeypatch, lookup_s: float = 0.0) -> FakeDispatch:
    """Run the loop through ``steps``: events to notice, seconds to let it run, or
    "idle" to let Prism take nudges. Then wait for the loop's work to finish."""
    fake = FakeDispatch(lookup_s)
    monkeypatch.setattr(agent, "dispatch", fake)

    async def go():
        rounds = agent.ToolRounds(session, object())
        loop = asyncio.create_task(rounds.run())
        for step in steps:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif step == "idle":
                session.idle.set()
            else:
                rounds.notice(step)
        await _settle(loop, rounds)
        if loop.done():
            return loop.result()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)

    asyncio.run(go())
    return fake


def _kinds(session) -> list:
    return [entry[:2] for entry in session.log]


def test_the_calls_of_a_reply_run_side_by_side_then_the_reply_continues(monkeypatch):
    session = FakeSession()
    fake = _drive(session, [
        _call("resp_1", "call_a", "get_account"),
        _call("resp_1", "call_b", "get_payment_arrangement"),
        _ended("resp_1"),
    ], monkeypatch, lookup_s=0.1)
    assert sorted(fake.ran) == ["call_a", "call_b"]
    assert fake.most["read"] == 2
    assert sorted(_kinds(session)[:2]) == [("result", "call_a"), ("result", "call_b")]
    assert _kinds(session)[2:] == [("continue", "resp_1")]


def test_writes_run_one_at_a_time(monkeypatch):
    """Prism can ask for the same write again while the first is still running; the
    shared tools refuse the repeat only once the first has finished."""
    session = FakeSession()
    fake = _drive(session, [
        _call("resp_1", "call_a", "create_payment_arrangement"),
        _call("resp_2", "call_b", "create_payment_arrangement"),
        _call("resp_2", "call_c", "set_paperless_billing"),
        _ended("resp_1"),
        _ended("resp_2"),
    ], monkeypatch, lookup_s=0.1)
    assert sorted(fake.ran) == ["call_a", "call_b", "call_c"]
    assert fake.most["write"] == 1


def test_a_lookup_does_not_wait_for_a_slow_write(monkeypatch):
    session = FakeSession()
    fake = _drive(session, [
        _call("resp_1", "call_pay", "create_payment_arrangement"),
        _call("resp_2", "call_look", "get_account"),
        _ended("resp_1"),
        _ended("resp_2"),
    ], monkeypatch, lookup_s=0.1)
    assert fake.most == {"write": 1, "read": 1}
    # Both ran at once: the lookup started before the write finished.
    assert sorted(fake.ran) == ["call_look", "call_pay"]



def test_a_reply_without_tool_calls_is_not_continued(monkeypatch):
    session = FakeSession()
    _drive(session, [_ended("resp_1")], monkeypatch)
    assert session.log == []


def test_a_round_the_caller_spoke_over_is_reported(monkeypatch):
    """The caller speaks while a call runs, and the round closes. Without a report,
    Prism mentions the result only if the caller asks."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1")], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1"), ("report", "call_a")]


@pytest.mark.parametrize("code", sorted(agent.SUPERSEDED_CODES))
def test_a_round_prism_refuses_to_continue_is_reported(monkeypatch, code):
    session = FakeSession(refuse={"resp_1": _superseded(code)})
    fake = _drive(session, [_call("resp_1", "call_pay", "make_payment"), _ended("resp_1")], monkeypatch)
    assert fake.ran == ["call_pay"]
    assert _kinds(session) == [("result", "call_pay"), ("continue", "resp_1"), ("report", "call_pay")]


def test_a_refused_round_reports_each_of_its_results(monkeypatch):
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [
        _call("resp_1", "call_a", "get_account"),
        _call("resp_1", "call_b", "get_payment_arrangement"),
        _ended("resp_1"),
    ], monkeypatch)
    assert _kinds(session)[2:] == [("continue", "resp_1"), ("report", "call_a"), ("report", "call_b")]


def test_the_report_does_not_guess_what_prism_has_said_since(monkeypatch):
    """Prism answers the caller with a reply that makes its own call, and that
    round continues. The older round is still reported: Rime's guide reports
    every result whose round closed, and Prism may repeat itself."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    fake = _drive(session, [
        _call("resp_1", "call_a"),
        _ended("resp_1"),
        0.05,
        _call("resp_2", "call_b", "get_payment_arrangement"),
        _ended("resp_2"),
    ], monkeypatch)
    assert sorted(fake.ran) == ["call_a", "call_b"]
    assert _kinds(session) == [
        ("result", "call_a"),
        ("continue", "resp_1"),
        ("report", "call_a"),
        ("result", "call_b"),
        ("continue", "resp_2"),
    ]


def test_a_round_whose_reply_did_not_complete_is_dropped(monkeypatch):
    """The result still goes in, so Prism has it if the caller asks."""
    session = FakeSession()
    fake = _drive(session, [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1", status="cancelled")], monkeypatch)
    assert fake.ran == ["call_a"]
    assert _kinds(session) == [("result", "call_a")]


def test_any_other_continuation_error_fails_the_call(monkeypatch):
    fault = RealtimeFault(code="invalid_tool_continuation", message="no parent", scope="event")
    session = FakeSession(refuse={"resp_1": RimeRealtimeError(fault)})
    with pytest.raises(ExceptionGroup) as failure:
        _drive(session, [_call("resp_1", "call_a"), _ended("resp_1")], monkeypatch)
    assert failure.group_contains(RimeRealtimeError, match="invalid_tool_continuation")


def test_a_refused_report_leaves_the_call_running(monkeypatch):
    session = FakeSession(
        refuse={"resp_1": RimeInputError("the caller spoke first")},
        refuse_nudge=_superseded("nudge_superseded"),
    )
    fake = _drive(session, [
        _call("resp_1", "call_a"),
        _ended("resp_1"),
        0.05,
        _call("resp_2", "call_b"),
        _ended("resp_2"),
    ], monkeypatch)
    assert sorted(fake.ran) == ["call_a", "call_b"]
    assert _kinds(session)[-2:] == [("result", "call_b"), ("continue", "resp_2")]


def test_a_slow_lookup_gets_a_still_working_line_and_then_the_report(monkeypatch):
    """Rime's slow-tool pattern. The line closes the round, so the result is
    reported with a nudge."""
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.05, 10.0))
    session = FakeSession(refuse={"resp_1": RimeInputError("the line closed the round")})
    _drive(session, [_call("resp_1", "call_pay", "create_payment_arrangement"), _ended("resp_1")],
           monkeypatch, lookup_s=0.2)
    assert _kinds(session) == [
        ("line", agent.STILL_WORKING),
        ("result", "call_pay"),
        ("continue", "resp_1"),
        ("report", "call_pay"),
    ]


def test_a_fast_lookup_gets_no_line(monkeypatch):
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.5, 10.0))
    session = FakeSession()
    _drive(session, [_call("resp_1", "call_a"), _ended("resp_1")], monkeypatch, lookup_s=0.05)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]


def test_the_still_working_lines_stop_when_prism_refuses_one(monkeypatch):
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.02, 0.02))
    session = FakeSession(refuse_nudge=_superseded("nudge_superseded"))
    _drive(session, [_call("resp_1", "call_a"), _ended("resp_1")], monkeypatch, lookup_s=0.2)
    assert _kinds(session) == [("line", agent.STILL_WORKING), ("result", "call_a"), ("continue", "resp_1")]


def test_a_lookup_past_its_deadline_gets_an_error_result(monkeypatch):
    monkeypatch.setattr(agent, "LOOKUP_DEADLINE_S", 0.05)
    session = FakeSession()
    _drive(session, [_call("resp_1", "call_a"), _ended("resp_1")], monkeypatch, lookup_s=0.3)
    assert session.log[0] == ("result", "call_a", {"error": "The lookup timed out"})
    assert _kinds(session)[1:] == [("continue", "resp_1")]


def _run_control(session, steps, monkeypatch) -> bool:
    """Run the whole control task, greeting and tool rounds, through ``steps`` (as in
    ``_drive``). Returns whether the call was still running at the end."""
    monkeypatch.setattr(agent, "dispatch", FakeDispatch(0.0))

    async def go():
        rounds = agent.ToolRounds(session, object())
        control = asyncio.create_task(agent._control(rounds))
        for step in steps:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif step == "idle":
                session.idle.set()
            else:
                rounds.notice(step)
        await asyncio.sleep(0.05)
        running = not control.done()
        control.cancel()
        await asyncio.gather(control, return_exceptions=True)
        return running

    return asyncio.run(go())


def test_the_greeting_opens_the_call(monkeypatch):
    session = FakeSession()
    assert _run_control(session, [0.02], monkeypatch)
    assert session.log == [("line", agent.GREETING_INSTRUCTION)]


def test_tool_calls_run_while_the_greeting_waits(monkeypatch):
    session = FakeSession()
    session.idle.clear()
    assert _run_control(session, [
        0.02,
        _call("resp_1", "call_a", "verify_caller"),
        _ended("resp_1"),
        0.1,
        "idle",
    ], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1"), ("line", agent.GREETING_INSTRUCTION)]


@pytest.mark.parametrize("refusal", [
    _superseded("nudge_superseded"),  # the caller spoke first
    RealtimeAdmissionTimeout("readiness expired"),  # the wait ran out
])
def test_a_refused_greeting_leaves_the_call_running(monkeypatch, refusal):
    session = FakeSession(refuse_nudge=refusal)
    assert _run_control(session, [0.02, _call("resp_1", "call_a"), _ended("resp_1"), 0.1], monkeypatch)
    assert _kinds(session) == [("line", agent.GREETING_INSTRUCTION), ("result", "call_a"), ("continue", "resp_1")]
