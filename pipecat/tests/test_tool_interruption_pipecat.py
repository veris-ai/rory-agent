"""A caller who barges in during a tool call is told what the call really did.

Rory's handlers run their blocking vendor call on a thread, which an
interruption cannot stop. With Pipecat's default ``cancel_on_interruption=True``
the payment went through while the LLM was told the call was CANCELLED. These
tests run the real handler through a real Pipecat pipeline, with a scripted LLM
in place of OpenAI and the dispatcher replaced by a slow fake vendor call.
"""

from __future__ import annotations

import asyncio
import json
import time

from pipecat.frames.frames import (
    EndFrame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner

from rory_pipecat import agent, tools
from rory_tools import CallSession

VENDOR_CALL_SECS = 0.3


class ScriptedLLM(LLMService):
    """Calls make_payment on its first inference and answers "ok" after that."""

    def __init__(self):
        super().__init__(run_in_parallel=False)
        self.seen: list[dict] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        self.seen.append(frame.context.get_messages()[-1])
        if len(self.seen) == 1:
            await self.run_function_calls(
                [FunctionCallFromLLM("make_payment", "call_1", {}, frame.context)]
            )
        else:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame("ok"))
            await self.push_frame(LLMFullResponseEndFrame())


def _call(monkeypatch, *, interrupt: bool):
    charged = []

    def slow_vendor(session, name, args):
        time.sleep(VENDOR_CALL_SECS)
        charged.append(name)
        return {"status": "paid"}

    monkeypatch.setattr(tools, "dispatch", slow_vendor)
    llm = ScriptedLLM()
    agent._register_tools(llm)
    context = LLMContext([{"role": "user", "content": "pay my bill"}])
    user, assistant = LLMContextAggregatorPair(context)
    worker = PipelineWorker(
        Pipeline([user, llm, assistant]), params=PipelineParams(), app_resources=CallSession()
    )

    async def drive():
        await worker.queue_frames([LLMRunFrame()])
        if interrupt:
            await asyncio.sleep(VENDOR_CALL_SECS / 3)
            await worker.queue_frames(
                [
                    InterruptionFrame(),
                    LLMMessagesAppendFrame([{"role": "user", "content": "wait"}], run_llm=True),
                ]
            )
        await asyncio.sleep(VENDOR_CALL_SECS * 2)
        await worker.queue_frames([EndFrame()])

    async def main():
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.gather(runner.run(), drive())

    asyncio.run(main())
    return charged, llm.seen, context.get_messages()


def _contents(messages, role):
    return [m["content"] for m in messages if m.get("role") == role]


def test_every_tool_runs_to_completion_through_an_interruption():
    llm = ScriptedLLM()
    agent._register_tools(llm)
    assert {n: f.cancel_on_interruption for n, f in llm._functions.items()} == {
        n: False for n in tools.HANDLERS
    }


def test_an_uninterrupted_call_settles_like_a_synchronous_one(monkeypatch):
    charged, seen, messages = _call(monkeypatch, interrupt=False)
    assert charged == ["make_payment"]
    assert _contents(messages, "tool") == [json.dumps({"status": "paid"})]
    assert seen[-1] == {"role": "tool", "content": '{"status": "paid"}', "tool_call_id": "call_1"}


def test_an_interrupted_call_reports_its_real_result(monkeypatch):
    charged, seen, messages = _call(monkeypatch, interrupt=True)
    assert charged == ["make_payment"]
    assert "CANCELLED" not in _contents(messages, "tool")
    # The result arrives after the caller spoke, and the LLM is run on it.
    [late] = [json.loads(c) for c in _contents(messages, "developer")]
    assert late["status"] == "finished" and late["tool_call_id"] == "call_1"
    assert "paid" in json.dumps(late["result"])
    assert seen[-1]["role"] == "developer"
