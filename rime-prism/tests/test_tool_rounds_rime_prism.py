"""Prism tool rounds against a fake session.

The parity tests check the declared surface. These drive the tool loop: each call
runs on its own task through the dispatcher as Prism makes it (writes one at a
time), each result is submitted, and the reply that made the calls is continued
once all of them are in. A round that closes before it uses its results is
reported with a nudge, unless a newer round is open or Prism has answered the
caller since (an answer that had said nothing when the result went in counts).
A nudge still waiting is withdrawn when Prism starts answering the
caller. No call is ever run twice.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from rimelabs_sdk import RimeInputError
from rimelabs_sdk.realtime import (
    MessageStarted,
    OutputRef,
    RealtimeAdmissionTimeout,
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
        self.during_submit: list = []  # events that arrive while a result is on its way in
        self.idle = asyncio.Event()  # Prism takes a nudge only once it is idle
        self.idle.set()

    async def submit_tool_result(self, call, output):
        while self.during_submit:
            self.rounds.notice(self.during_submit.pop(0))
        self.log.append(("result", call.call_id, json.loads(output)))

    async def continue_reply(self, parent):
        while self.before_continue:
            self.rounds.notice(self.before_continue.pop(0))
        self.log.append(("continue", parent.response_id))
        if parent.response_id in self.refuse:
            raise self.refuse[parent.response_id]
        self.rounds.notice(_started(f"{parent.response_id}_cont", "tools"))

    async def request_reply(self, *, instruction=None, tool_call=None):
        await self.idle.wait()
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


def _message(response_id: str) -> MessageStarted:
    """The reply's first spoken output."""
    ref = ResponseRef(session_id="sess", response_id=response_id)
    return MessageStarted(output=OutputRef(response=ref, item_id=f"msg_{response_id}", output_index=0, content_index=0))


def _superseded(code: str = "tool_continuation_unavailable") -> RimeRealtimeError:
    return RimeRealtimeError(RealtimeFault(code=code, message="round closed", scope="tool_roundtrip"))


class FakeDispatch:
    """The shared dispatcher, recording which calls ran and how many at once.

    Each call takes ``lookup_s``, or its own time in ``slow`` (by call id)."""

    def __init__(self, lookup_s: float, slow: dict[str, float] | None = None) -> None:
        self.lookup_s = lookup_s
        self.slow = slow or {}
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
        time.sleep(self.slow.get(args["n"], self.lookup_s))
        with self._lock:
            self.running[kind] -= 1
        return {"tool": name, "ok": True}


def _drive(session, steps, monkeypatch, lookup_s: float = 0.0, slow: dict[str, float] | None = None) -> FakeDispatch:
    """Run the loop through ``steps``: events to notice, seconds to let it run, or
    "idle" to let Prism take nudges. Then wait for every open round to finish."""
    fake = FakeDispatch(lookup_s, slow)
    monkeypatch.setattr(agent, "dispatch", fake)

    async def go():
        rounds = agent.ToolRounds(session, object())
        session.rounds = rounds
        loop = asyncio.create_task(rounds.run())
        for step in steps:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif step == "idle":
                session.idle.set()
            else:
                rounds.notice(step)
        deadline = time.monotonic() + 2
        while (rounds._rounds or not rounds._events.empty()) and not loop.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
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


def test_a_dropped_round_is_reported(monkeypatch):
    """The caller speaks while a call runs, Prism answers them before the result is
    in, and the round closes. Without a report, Prism never mentions the result."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1"), _ended("resp_2")], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1"), ("report", "call_a")]


def test_a_cancelled_reply_is_not_continued_and_its_result_is_reported(monkeypatch):
    session = FakeSession()
    _drive(session, [_call("resp_1", "call_a", "verify_caller"), _ended("resp_1", status="cancelled")], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("report", "call_a")]


def test_a_round_superseded_by_a_newer_round_is_not_reported(monkeypatch):
    """Prism answers the caller with a reply that makes its own call. The newer
    round continues with every result in the history, so reporting the old one would
    repeat the answer. No call runs twice."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    fake = _drive(session, [
        _call("resp_1", "call_a"),
        _ended("resp_1"),
        _call("resp_2", "call_b", "get_payment_arrangement"),
        _ended("resp_2"),
    ], monkeypatch)
    assert sorted(fake.ran) == ["call_a", "call_b"]
    assert sorted(_kinds(session)) == [
        ("continue", "resp_1"),
        ("continue", "resp_2"),
        ("result", "call_a"),
        ("result", "call_b"),
    ]


def test_no_report_while_a_newer_round_is_still_open(monkeypatch):
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [_call("resp_1", "call_a"), _ended("resp_1"), _call("resp_2", "call_b")], monkeypatch)
    assert ("report", "call_a") not in _kinds(session)


def test_no_report_when_prism_answered_the_caller_after_the_result(monkeypatch):
    """The caller asked again after the result went in, and Prism's answer had the
    result in its history."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    session.before_continue = [_started("resp_2", "speech")]
    _drive(session, [_call("resp_1", "call_a"), _ended("resp_1"), _ended("resp_2")], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]


def test_an_answer_that_spoke_before_the_result_still_gets_the_report(monkeypatch):
    """Prism was already answering the caller out loud when a write's result went in,
    so the answer could not use it, and the report still goes out."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [
        _call("resp_1", "call_pay", "create_payment_arrangement"),
        _ended("resp_1"),
        _started("resp_2", "speech"),
        _message("resp_2"),
        0.1,  # the result goes in while the answer plays
        _ended("resp_2"),
    ], monkeypatch, lookup_s=0.05)
    assert _kinds(session)[-1] == ("report", "call_pay")


def test_no_report_when_the_result_went_in_before_the_answer_said_anything(monkeypatch):
    """The caller barges in on the reply that made a write; Prism starts answering
    them, and the write's result goes in before that answer says a word. The answer
    uses the result, so a report would make the caller hear it twice."""
    session = FakeSession()
    _drive(session, [
        _started("resp_1", "speech"),
        _call("resp_1", "call_pay", "modify_payment_arrangement"),
        _ended("resp_1", status="cancelled"),
        _started("resp_2", "speech"),
        0.1,  # the result goes in before the answer says anything
        _message("resp_2"),
        _ended("resp_2"),
    ], monkeypatch, lookup_s=0.05)
    assert _kinds(session) == [("result", "call_pay")]


def test_the_reply_that_made_the_call_has_already_spoken(monkeypatch):
    """The call is that reply's output, so its prompt was fixed before the result.
    A result that goes in while it still runs is reported when it is cancelled."""
    session = FakeSession()
    _drive(session, [
        _started("resp_1", "speech"),
        _call("resp_1", "call_a", "verify_caller"),
        0.1,  # the result goes in while the reply still runs
        _ended("resp_1", status="cancelled"),
    ], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("report", "call_a")]


@pytest.mark.parametrize("code", sorted(agent.SUPERSEDED_CODES))
def test_prism_refusing_the_continuation_drops_the_round(monkeypatch, code):
    session = FakeSession(refuse={"resp_1": _superseded(code)})
    fake = _drive(session, [_call("resp_1", "call_pay", "make_payment"), _ended("resp_1")], monkeypatch)
    assert fake.ran == ["call_pay"]
    assert [entry[0] for entry in session.log] == ["result", "continue", "report"]


def test_any_other_continuation_error_fails_the_call(monkeypatch):
    fault = RealtimeFault(code="invalid_tool_continuation", message="no parent", scope="event")
    session = FakeSession(refuse={"resp_1": RimeRealtimeError(fault)})
    with pytest.raises(ExceptionGroup) as failure:
        _drive(session, [_call("resp_1", "call_a"), _ended("resp_1")], monkeypatch)
    assert failure.group_contains(RimeRealtimeError, match="invalid_tool_continuation")


def test_a_report_the_caller_talks_over_is_skipped(monkeypatch):
    session = FakeSession(
        refuse={"resp_1": RimeInputError("the caller spoke first")},
        refuse_report=_superseded("nudge_superseded"),
    )
    fake = _drive(session, [
        _call("resp_1", "call_a"),
        _ended("resp_1"),
        0.1,
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


def test_a_held_report_is_withdrawn_when_prism_answers_the_caller(monkeypatch):
    """The report waits for Prism to be idle, the caller speaks, and Prism's answer
    makes a new call. Taken after that, the report would close the new round, and
    the caller would hear the result again."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    session.idle.clear()
    _drive(session, [
        _call("resp_1", "call_a", "create_payment_arrangement"),
        _ended("resp_1"),
        0.05,  # the report waits for Prism to be idle
        _started("resp_2", "speech"),
        _call("resp_2", "call_b", "get_payment_arrangement"),
        _ended("resp_2"),
        "idle",
    ], monkeypatch)
    assert _kinds(session) == [
        ("result", "call_a"),
        ("continue", "resp_1"),
        ("result", "call_b"),
        ("continue", "resp_2"),
    ]


def test_a_held_still_working_line_is_withdrawn_when_prism_answers_the_caller(monkeypatch):
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.02, 10.0))
    session = FakeSession()
    session.idle.clear()
    _drive(session, [
        _call("resp_1", "call_pay", "create_payment_arrangement"),
        0.1,  # the line waits for Prism to be idle
        _started("resp_2", "speech"),
        "idle",
        0.3,
        _ended("resp_1"),
    ], monkeypatch, lookup_s=0.3)
    assert ("line", agent.STILL_WORKING) not in _kinds(session)
    assert _kinds(session) == [("result", "call_pay"), ("continue", "resp_1")]


def _run_control(session, steps, monkeypatch) -> bool:
    """Run the whole control task, greeting and tool rounds, through ``steps`` (as in
    ``_drive``). Returns whether the call was still running at the end."""
    monkeypatch.setattr(agent, "dispatch", FakeDispatch(0.0))

    async def go():
        rounds = agent.ToolRounds(session, object())
        session.rounds = rounds
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


def test_the_caller_speaking_first_withdraws_the_greeting_and_tools_run_meanwhile(monkeypatch):
    """The greeting waits for Prism to be idle while Prism answers the caller with a
    call. The call runs without waiting for the greeting, and the answer withdraws
    it, so it does not play after the answer."""
    session = FakeSession()
    session.idle.clear()
    assert _run_control(session, [
        0.02,
        _started("resp_1", "speech"),
        _call("resp_1", "call_a", "verify_caller"),
        _ended("resp_1"),
        0.1,
        "idle",
    ], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]


def test_a_greeting_whose_wait_runs_out_leaves_the_call_running(monkeypatch):
    """The caller talks over the opening until Prism's readiness wait runs out. The
    SDK keeps the session usable, and so does the call."""

    class OpeningTimesOut(FakeSession):
        async def request_reply(self, **request):
            raise RealtimeAdmissionTimeout("readiness expired")

    session = OpeningTimesOut()
    assert _run_control(session, [0.02, _call("resp_1", "call_a"), _ended("resp_1"), 0.1], monkeypatch)
    assert _kinds(session) == [("result", "call_a"), ("continue", "resp_1")]


def test_a_round_whose_calls_finish_out_of_order_reports_the_result_that_went_in_last(monkeypatch):
    """Prism makes a slow call, then a fast one. The fast result goes in, Prism
    answers the caller with it, and then the slow result goes in. That answer could
    not have used the slow result, so it is reported."""
    session = FakeSession(refuse={"resp_1": RimeInputError("the caller spoke first")})
    _drive(session, [
        _call("resp_1", "call_slow", "get_account"),
        _call("resp_1", "call_fast", "get_payment_arrangement"),
        _ended("resp_1"),
        0.05,  # the fast result goes in
        _started("resp_2", "speech"),
        _message("resp_2"),
        _ended("resp_2"),
    ], monkeypatch, slow={"call_slow": 0.2})
    assert _kinds(session)[-1] == ("report", "call_slow")


def test_an_older_round_still_running_does_not_hold_back_a_newer_report(monkeypatch):
    """The caller barges in on a slow lookup, and Prism's answer makes a fast call
    whose round the caller also closes. The older round cannot continue after the
    newer reply, so it does not cover the newer result: each is reported."""
    session = FakeSession(refuse={"resp_2": RimeInputError("the caller spoke first")})
    _drive(session, [
        _call("resp_1", "call_slow", "get_account"),
        _ended("resp_1", status="cancelled"),
        _call("resp_2", "call_fast", "get_payment_arrangement"),
        _ended("resp_2"),
    ], monkeypatch, slow={"call_slow": 0.3})
    assert _kinds(session) == [
        ("result", "call_fast"),
        ("continue", "resp_2"),
        ("report", "call_fast"),
        ("result", "call_slow"),
        ("report", "call_slow"),
    ]


def test_no_still_working_line_once_prism_has_answered_the_caller(monkeypatch):
    """Prism answers the caller while a slow write's first line is still counting
    down. The caller is not waiting in silence, and a line taken now would close
    any round that answer opened."""
    monkeypatch.setattr(agent, "STILL_WORKING_WAITS_S", (0.1, 10.0))
    session = FakeSession()
    _drive(session, [
        _call("resp_1", "call_pay", "create_payment_arrangement"),
        _ended("resp_1"),
        0.03,
        _started("resp_2", "speech"),
        _message("resp_2"),
        _ended("resp_2"),
    ], monkeypatch, lookup_s=0.3)
    assert _kinds(session) == [("result", "call_pay"), ("continue", "resp_1")]


def test_an_answer_whose_first_words_arrive_while_the_result_goes_in_counts_as_using_it(monkeypatch):
    """Which came first on Prism's side, the result or the answer's first words, is
    unknown, so the answer counts as having used the result."""
    session = FakeSession()
    session.during_submit = [_message("resp_2")]
    _drive(session, [
        _started("resp_1", "speech"),
        _call("resp_1", "call_pay", "modify_payment_arrangement"),
        _ended("resp_1", status="cancelled"),
        _started("resp_2", "speech"),
    ], monkeypatch, lookup_s=0.05)
    assert _kinds(session) == [("result", "call_pay")]
