"""Prism tool rounds against a fake session.

The parity tests check the declared surface. These drive the tool loop: calls run
one at a time through the dispatcher, each result is submitted, and the reply that
made them is continued. A round that closes before it uses its result is reported
with a nudge once Prism is idle, unless a newer round is open or Prism has answered
the caller since the result went in. No call is ever run twice.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from rimelabs_sdk import RimeInputError
from rimelabs_sdk.realtime import (
    InputReady,
    RealtimeFault,
    ResponseEnded,
    ResponseRef,
    ResponseStarted,
    RimeRealtimeError,
    ToolCall,
    ToolCallRef,
)

from rory_rime_prism import agent


class FakeSession:
    """Records what the loop sends.

    ``refuse`` maps a response id to the error its continuation raises.
    ``refuse_report`` is the error a nudge that reports a result raises.
    """

    def __init__(self, refuse: dict[str, Exception] | None = None, refuse_report: Exception | None = None):
        self.refuse = refuse or {}
        self.refuse_report = refuse_report
        self.log: list[tuple] = []
        self.rounds: agent.ToolRounds | None = None
        self.before_continue: list = []  # events that arrive after the results went in

    async def submit_tool_result(self, call, output):
        self.log.append(("result", call.call_id, json.loads(output)))

    async def continue_reply(self, parent):
        while self.before_continue:
            self.rounds.notice(self.before_continue.pop(0))
        self.log.append(("continue", parent.response_id))
        if parent.response_id in self.refuse:
            raise self.refuse[parent.response_id]
        self.rounds.notice(_started(f"{parent.response_id}_cont", "tools"))

    async def request_reply(self, *, instruction=None, tool_call=None):
        if tool_call is None:
            self.log.append(("line", instruction))
            return
        self.log.append(("report", tool_call.call_id))
        if self.refuse_report is not None:
            raise self.refuse_report


def _call(response_id: str, call_id: str, name: str = "get_account") -> ToolCall:
    ref = ToolCallRef(session_id="sess", response_id=response_id, call_id=call_id)
    return ToolCall(call=ref, item_id=f"item_{call_id}", name=name, arguments={"n": call_id})


def _ended(response_id: str, status: str = "completed") -> ResponseEnded:
    return ResponseEnded(response=ResponseRef(session_id="sess", response_id=response_id), status=status)


def _started(response_id: str, cause: str) -> ResponseStarted:
    return ResponseStarted(response=ResponseRef(session_id="sess", response_id=response_id), cause=cause)


def _superseded(code: str = "tool_continuation_unavailable") -> RimeRealtimeError:
    return RimeRealtimeError(RealtimeFault(code=code, message="round closed", scope="tool_roundtrip"))


def _run(events, session, monkeypatch, lookup_s: float = 0.0):
    ran: list[str] = []

    def fake_dispatch(_session, name, args):
        ran.append(args["n"])
        time.sleep(lookup_s)
        return {"tool": name, "ok": True}

    monkeypatch.setattr(agent, "dispatch", fake_dispatch)

    async def go():
        rounds = agent.ToolRounds(session, object())
        session.rounds = rounds
        for event in events:
            rounds.notice(event)
        loop = asyncio.create_task(rounds.run())
        while not rounds._events.empty() and not loop.done():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        if loop.done():
            return loop.result()
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)

    asyncio.run(go())
    return ran


def _kinds(session) -> list:
    return [entry[:2] for entry in session.log]


def test_calls_run_in_order_then_the_reply_continues(monkeypatch):
    session = FakeSession()
    events = [_call("resp_1", "call_a", "verify_caller"), _call("resp_1", "call_b"), _ended("resp_1"), InputReady()]
    ran = _run(events, session, monkeypatch)
    assert ran == ["call_a", "call_b"]
    assert session.log == [
        ("result", "call_a", {"tool": "verify_caller", "ok": True}),
        ("result", "call_b", {"tool": "get_account", "ok": True}),
        ("continue", "resp_1"),
    ]


def test_a_reply_without_tool_calls_is_not_continued(monkeypatch):
    session = FakeSession()
    _run([_ended("resp_1"), InputReady()], session, monkeypatch)
    assert session.log == []


def test_a_dropped_round_is_reported_once_prism_is_idle(monkeypatch):
    """The smoke-run case: the caller said "okay" during verify_caller, Prism said
    "please hold", and the result came in after. Prism said nothing more, and the
    caller hung up. The result is now reported once Prism is idle."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    events = [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1"), _ended("resp_2"), InputReady()]
    ran = _run(events, session, monkeypatch)
    assert ran == ["call_a"]
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1"), ("report", "call_a")]


def test_a_cancelled_reply_is_not_continued_and_its_result_is_reported(monkeypatch):
    session = FakeSession()
    events = [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1", status="cancelled"), InputReady()]
    ran = _run(events, session, monkeypatch)
    assert ran == ["call_a"]
    assert _kinds(session) == [("result", "call_a"), ("report", "call_a")]


def test_a_round_superseded_by_a_newer_round_is_not_reported(monkeypatch):
    """The first smoke run's case: the reply to the caller's "okay" made its own
    call. The newer round continues with every result in the history, so reporting
    the old one would repeat the answer. No call runs twice."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    events = [
        _call("resp_1", "call_a"),
        _ended("resp_1"),
        InputReady(),
        _call("resp_2", "call_b", "get_payment_arrangement"),
        _ended("resp_2"),
        InputReady(),
    ]
    ran = _run(events, session, monkeypatch)
    assert ran == ["call_a", "call_b"]
    assert _kinds(session) == [
        ("result", "call_a"),
        ("continue", "resp_1"),
        ("result", "call_b"),
        ("continue", "resp_2"),
    ]


def test_no_report_while_a_newer_round_is_still_open(monkeypatch):
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    events = [_call("resp_1", "call_a"), _ended("resp_1"), InputReady(), _call("resp_2", "call_b"), InputReady()]
    _run(events, session, monkeypatch)
    assert ("report", "call_a") not in _kinds(session)


def test_no_report_when_prism_answered_the_caller_after_the_result(monkeypatch):
    """The caller asked again after the result went in, and Prism's answer had the
    result in its history."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    session.before_continue = [_started("resp_2", "speech")]
    _run([_call("resp_1", "call_a"), _ended("resp_1"), _ended("resp_2"), InputReady()], session, monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]


def test_an_answer_that_started_before_the_result_still_gets_the_report(monkeypatch):
    """The premature-claim case: the caller said "okay" during the 9 s
    create_payment_arrangement, and Prism answered before the result was in."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    events = [
        _call("resp_1", "call_pay", "create_payment_arrangement"),
        _ended("resp_1"),
        _started("resp_2", "speech"),
        _ended("resp_2"),
        InputReady(),
    ]
    _run(events, session, monkeypatch)
    assert _kinds(session)[-1] == ("report", "call_pay")


@pytest.mark.parametrize("code", sorted(agent.SUPERSEDED_CODES))
def test_prism_refusing_the_continuation_drops_the_round(monkeypatch, code):
    session = FakeSession(refuse={"resp_1": _superseded(code)})
    ran = _run([_call("resp_1", "call_pay", "make_payment"), _ended("resp_1"), InputReady()], session, monkeypatch)
    assert ran == ["call_pay"]
    assert [entry[0] for entry in session.log] == ["result", "continue", "report"]


def test_any_other_continuation_error_fails_the_call(monkeypatch):
    fault = RealtimeFault(code="invalid_tool_continuation", message="no parent", scope="event")
    session = FakeSession(refuse={"resp_1": RimeRealtimeError(fault)})
    with pytest.raises(RimeRealtimeError, match="invalid_tool_continuation"):
        _run([_call("resp_1", "call_a"), _ended("resp_1")], session, monkeypatch)


def test_a_report_the_caller_talks_over_is_skipped(monkeypatch):
    session = FakeSession(
        refuse={"resp_1": RimeInputError("the caller spoke first")},
        refuse_report=_superseded("nudge_superseded"),
    )
    events = [_call("resp_1", "call_a"), _ended("resp_1"), InputReady(), _call("resp_2", "call_b"), _ended("resp_2")]
    ran = _run(events, session, monkeypatch)
    assert ran == ["call_a", "call_b"]
    assert _kinds(session)[-2:] == [("result", "call_b"), ("continue", "resp_2")]


def test_a_slow_lookup_gets_a_still_working_line_and_then_the_report(monkeypatch):
    """Rime's slow-tool pattern. The line closes the round, so the result is
    reported with a nudge."""
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.05, 10.0))
    session = FakeSession(refuse={"resp_1": RimeInputError("the line closed the round")})
    events = [_call("resp_1", "call_pay", "create_payment_arrangement"), _ended("resp_1"), InputReady()]
    _run(events, session, monkeypatch, lookup_s=0.2)
    assert _kinds(session) == [
        ("line", agent.STILL_WORKING),
        ("result", "call_pay"),
        ("continue", "resp_1"),
        ("report", "call_pay"),
    ]


def test_a_fast_lookup_gets_no_line(monkeypatch):
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.5, 10.0))
    session = FakeSession()
    _run([_call("resp_1", "call_a"), _ended("resp_1"), InputReady()], session, monkeypatch, lookup_s=0.05)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]
