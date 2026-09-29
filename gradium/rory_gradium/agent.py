"""Rory on gradbot — Gradium speech in and out, around an OpenAI-compatible LLM.

Listens on ``/voice`` for one bidirectional PCM16 stream from the Veris actor.
For each connection it starts its own gradbot session: gradbot's Rust
multiplexer runs Gradium streaming STT, the LLM and Gradium streaming TTS
concurrently, owns turn-taking and barge-in, and hands each tool call back to
this process, which runs it through the shared dispatcher. The agent speaks
first.

No vendor holds the session: the event loop, the turn state and the tools live
in this pod, and only the individual STT, LLM and TTS requests leave it, so
nothing dials in. Unlike the other cascades (pipecat, livekit) the pipeline is
not assembled from Python services — gradbot ships it as one engine behind
``run()`` / ``send_audio()`` / ``receive()``.

Two gradbot defaults are overridden, both to keep the comparison like-for-like:

- gradbot opens the call by sending the LLM a literal ``[start]`` message, and
  its own scaffold tells the model to greet. The shared prompt says the greeting
  already happened, so the session instructions append an opening-line section
  that resolves the contradiction and names the shared greeting.
- gradbot nudges a silent caller and hangs up after three nudges. No other Rory
  transport does, so ``silence_timeout_s`` is ``0``.

And one gradbot behaviour is guarded against: a second or two into a pending
tool call gradbot re-prompts the model, which often issues the same call again
before the first result reaches it. ``ToolCalls`` answers such a re-issued call
with the earlier result instead of running the tool twice.

Everything below the transport is shared with the other candidates — same
prompt, same greeting, same frozen-clock date context, same tools, same
dispatcher.

Sample-rate note: gradbot decodes PCM input at 24 kHz, the actor's rate, but
encodes PCM output at 48 kHz, which is halved back to 24 kHz on the way out.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Set, Tuple

import gradbot
from loguru import logger
from starlette.websockets import WebSocket, WebSocketDisconnect

from rory_tools import CallSession, dispatch
from rory_tools.prompt import GREETING, load_agent_prompt, today_context

from .tools import TOOLS

# The Veris actor speaks and listens at 24 kHz PCM16; gradbot's PCM output is 48 kHz.
ACTOR_RATE_HZ = 24000
GRADBOT_OUTPUT_RATE_HZ = 48000

# Gradium flagship voice "Harper".
GRADIUM_VOICE_ID = os.environ.get("GRADIUM_VOICE_ID", "4SZHfMpw-p46Ywgs")
GRADIUM_LLM_MODEL = os.environ.get("GRADIUM_LLM_MODEL", "gpt-4.1-mini")

# Roughly one heartbeat per second at 20 ms frames.
LOG_EVERY_N_FRAMES = 50

AGENT_PROMPT = load_agent_prompt()

OPENING_LINE = f"""# Opening line
Disregard the note above about having already greeted the caller: on this
platform your first turn *is* the greeting. The call has just connected and
nobody has spoken. Open with exactly: "{GREETING}" Then stop and wait for the
caller to respond. Do not greet again after that."""


def session_instructions() -> str:
    """The prompt gradbot wraps in its own scaffold, resolved per call.

    The date context is resolved per call, off the frozen clock the attempt
    restored with its snapshot.
    """
    return f"{AGENT_PROMPT}\n\n{today_context()}\n\n{OPENING_LINE}"


def session_config() -> gradbot.SessionConfig:
    return gradbot.SessionConfig(
        voice_id=GRADIUM_VOICE_ID,
        instructions=session_instructions(),
        language=gradbot.Lang.En,
        tools=TOOLS,
        assistant_speaks_first=True,
        silence_timeout_s=0.0,
    )


@dataclass
class _Issued:
    """A tool call already dispatched, and how many prompts gradbot had built when its result went back."""

    result: "asyncio.Future[str]"
    answered_at_push: int | None = None


class ToolCalls:
    """One call's tool calls: through the shared dispatcher, one at a time, never twice by accident.

    gradbot re-prompts the model a second or two into a pending tool call, and
    the model often issues the same call again before the first result reaches
    it. Dispatching that would run the tool twice — a second verification
    attempt, a second payment. So a call with the same tool and arguments as an
    earlier one is answered with the earlier result while that call is still
    running, or when the generation issuing it was built from a prompt gradbot
    pushed before that result went back.

    The prompt, not the generation's start, is what dates a generation: gradbot
    emits ``push_to_llm`` when it builds a prompt and ``llm_started`` when that
    prompt starts generating, and on the bench a prompt pushed while
    ``get_account`` was running only started generating after the result had
    gone back — and re-issued the call. A generation belongs to the latest push
    before its ``llm_started``, and a tool call to the generation running when it
    arrives. An identical call from a generation built after the result went
    back is the model's own repeat, and runs.

    Each call runs in its own task, as in gradbot's reference server, so a
    re-issued call is read the moment gradbot emits it; the lock keeps dispatch
    sequential, since tools share account state.
    """

    def __init__(self, session: CallSession) -> None:
        self._session = session
        self._lock = asyncio.Lock()
        # Prompts gradbot has pushed, and which of them the running generation was built from.
        self._pushes = 0
        self._generation_prompt = 0
        self._issued: Dict[Tuple[str, str], _Issued] = {}
        self._tasks: Set[asyncio.Task] = set()

    def pushed_to_llm(self) -> None:
        self._pushes += 1

    def llm_started(self) -> None:
        self._generation_prompt = self._pushes

    def start(self, call: gradbot.ToolCallInfo, handle: gradbot.ToolCallHandlePy) -> None:
        # Arguments arrive as a JSON string; a no-argument tool can arrive with it empty.
        args = json.loads(call.args_json or "{}")
        key = (call.tool_name, json.dumps(args, sort_keys=True))
        earlier = self._issued.get(key)
        if earlier is not None and (
            earlier.answered_at_push is None or self._generation_prompt <= earlier.answered_at_push
        ):
            logger.info(
                f"[gb->a] tool {call.tool_name} ({call.call_id}) re-issued before the model saw "
                "the earlier result — answering with that result"
            )
            work = self._answer_again(earlier, handle)
        else:
            issued = _Issued(result=asyncio.get_running_loop().create_future())
            self._issued[key] = issued
            work = self._run(call, args, issued, handle)
        task = asyncio.create_task(work, name=f"tool {call.tool_name} ({call.call_id})")
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, call: gradbot.ToolCallInfo, args: dict, issued: _Issued, handle: gradbot.ToolCallHandlePy) -> None:
        async with self._lock:
            # The vendor clients are blocking; a worker thread keeps the audio moving.
            result = json.dumps(await asyncio.to_thread(dispatch, self._session, call.tool_name, args), default=str)
        logger.info(f"[gb->a] tool {call.tool_name} ({call.call_id}) -> {result}")
        issued.answered_at_push = self._pushes
        issued.result.set_result(result)
        await handle.send(result)

    @staticmethod
    async def _answer_again(earlier: _Issued, handle: gradbot.ToolCallHandlePy) -> None:
        await handle.send(await earlier.result)

    def _finished(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.opt(exception=task.exception()).error(f"[gb->a] {task.get_name()} failed")


async def run_voice_ws_bot(actor_ws: WebSocket) -> None:
    """One actor connection ↔ one gradbot session with Rory's tools.

    A fresh ``CallSession`` per connection: it holds the account
    ``verify_caller`` matched and dies with the call.
    """
    tool_calls = ToolCalls(CallSession())
    t_start = time.monotonic()
    input_handle, output_handle = await gradbot.run(
        gradium_api_key=os.environ["GRADIUM_API_KEY"],
        llm_api_key=os.environ["OPENAI_API_KEY"],
        llm_model_name=GRADIUM_LLM_MODEL,
        session_config=session_config(),
        input_format=gradbot.AudioFormat.Pcm,
        output_format=gradbot.AudioFormat.Pcm,
    )
    logger.info(
        f"[voice] gradbot session started voice={GRADIUM_VOICE_ID} model={GRADIUM_LLM_MODEL} tools={len(TOOLS)}"
    )
    up = asyncio.create_task(_pump_actor_to_gradbot(actor_ws, input_handle), name="actor->gradbot")
    down = asyncio.create_task(_pump_gradbot_to_actor(output_handle, actor_ws, tool_calls), name="gradbot->actor")
    try:
        done, _ = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in (up, down):
            if not task.done():
                task.cancel()
        await asyncio.gather(up, down, return_exceptions=True)
        await tool_calls.close()
        logger.info(f"[voice] handler exit duration={time.monotonic() - t_start:.1f}s")


async def _pump_actor_to_gradbot(actor_ws: WebSocket, input_handle: gradbot.SessionInputHandle) -> None:
    """Binary PCM16 frames from the actor → gradbot's STT input, untouched."""
    n_frames = 0
    n_bytes = 0
    try:
        while True:
            frame = await actor_ws.receive_bytes()
            n_frames += 1
            n_bytes += len(frame)
            if n_frames == 1:
                logger.info(f"[a->gb] first frame bytes={len(frame)}")
            elif n_frames % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[a->gb] forwarded {n_frames} frames ({n_bytes} bytes)")
            await input_handle.send_audio(frame)
    except WebSocketDisconnect as exc:
        logger.info(
            f"[a->gb] actor disconnected after {n_frames} frames "
            f"({n_bytes} bytes): code={getattr(exc, 'code', '?')}"
        )
        await input_handle.close()
    except KeyError as exc:
        # Starlette raises KeyError('bytes') when it got a text frame instead of
        # a binary one — a protocol mismatch on the actor side, not a bug here.
        logger.error(
            f"[a->gb] non-binary frame after {n_frames} binary frames — "
            f"actor protocol mismatch? ({exc})"
        )
        raise


async def _pump_gradbot_to_actor(
    output_handle: gradbot.SessionOutputHandle, actor_ws: WebSocket, tool_calls: ToolCalls
) -> None:
    """gradbot messages → audio back to the actor; tool calls → ``ToolCalls``.

    Audio streams to the actor as the multiplexer emits it. On barge-in gradbot
    simply stops emitting, so there is nothing to flush here. gradbot's own
    ``reset_asr`` tool is handled inside gradbot and never arrives.
    """
    # audioop.ratecv carries fractional-sample state between calls; keeping it
    # across chunks is what stops a click at every chunk boundary.
    resample_state = None
    n_chunks = 0
    n_bytes = 0
    n_tool_calls = 0
    while True:
        msg = await output_handle.receive()
        if msg is None:
            logger.info(
                f"[gb->a] session ended — {n_chunks} audio chunks / {n_bytes} bytes / {n_tool_calls} tool calls"
            )
            return

        if msg.msg_type == "audio":
            if not msg.data:
                continue
            pcm24, resample_state = audioop.ratecv(msg.data, 2, 1, GRADBOT_OUTPUT_RATE_HZ, ACTOR_RATE_HZ, resample_state)
            n_chunks += 1
            n_bytes += len(pcm24)
            if n_chunks == 1:
                logger.info(f"[gb->a] first audio chunk bytes={len(pcm24)}")
            elif n_chunks % LOG_EVERY_N_FRAMES == 0:
                logger.info(f"[gb->a] streamed {n_chunks} audio chunks ({n_bytes} bytes)")
            await actor_ws.send_bytes(pcm24)

        elif msg.msg_type == "tts_text":
            logger.info(f"[gb->a] rory_said: {msg.text}")

        elif msg.msg_type == "stt_text":
            logger.info(f"[gb->a] caller_said: {msg.text}")

        elif msg.msg_type == "event":
            logger.info(f"[gb->a] event: {msg.event.event_type}")
            if msg.event.event_type == "push_to_llm":
                tool_calls.pushed_to_llm()
            elif msg.event.event_type == "llm_started":
                tool_calls.llm_started()

        elif msg.msg_type == "tool_call":
            n_tool_calls += 1
            tool_calls.start(msg.tool_call, msg.tool_call_handle)
