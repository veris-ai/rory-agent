# rory-gpt-live

Rory on OpenAI's GPT-Live API (`gpt-live-1`). Transport only — the tools,
vendor clients and prompt live in rory-core.

One GPT-Live websocket per call. The live model listens and speaks at the same
time (full duplex; it stops itself when the caller talks over it) and runs no
tools of its own: tool use and reasoning are *delegated* to a Responses model
configured in `session.start`. The pod bridges the Veris actor's PCM16 to the
session, answers the backend's function calls through the shared dispatcher,
and continues the backend with `response.create`. Nothing dials in.

## The wire, relative to OpenAI Realtime

GPT-Live is a different API (`wss://api.openai.com/v1/live/sessions`, not
`v1/realtime`), so this is not the Realtime transport with a new model id:

- `session.start` carries the whole configuration once (`model`,
  `instructions`, `audio.output.voice`, `delegation`); it is immutable after
  `session.started`.
- Audio in is `session.input_audio.append`, audio out is
  `session.output_audio.delta`, both raw PCM16 mono 24 kHz base64. There is no
  input buffer to commit, no VAD to configure and no output-done event.
- Transcripts arrive as timed fragments (`session.input_transcript.delta`,
  `session.output_transcript.delta`) with no turn boundary; the bridge logs one
  line per speaker change.
- Tools live on the backend: `delegation.responses.tools` holds Rory's 16
  function tools, `session.delegation.created` announces a backend run, its
  Responses events arrive nested in `response.event` envelopes, a completed
  call is the nested `response.output_item.done` with a `function_call` item,
  and the answer is `response.item.create` (`function_call_output`) followed by
  `response.create`.
- There is no `response.create` for the live model itself; the greeting is a
  `session.instructions.append` sent after `session.started`.
- The session is billed per second (`session.usage.updated`, and the final
  figure on `session.closed`); the backend model is billed separately as
  Responses usage, reported in the nested `response.completed`.

`parallel_tool_calls` is off: Rory's tools share account state, so the backend
issues one call per run and the bridge answers it before continuing.

## Running it

```
docker build --platform linux/amd64 -f gpt-live/Dockerfile -t rory-gpt-live .
```

The image serves `/voice` and `/health` on `:8008`. The session's transcripts
and the nested Responses tool calls travel on the GPT-Live websocket, so a
trace has to be read from there rather than from SDK instrumentation.

## Environment

| Variable | Default | Role |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. The project behind it must have GPT-Live access; a key without it fails every call at `session.start` with `Voice session access denied`. |
| `LIVE_MODEL` | `gpt-live-1` | The live model. |
| `LIVE_BACKEND_MODEL` | `gpt-5.6-terra` | The Responses model tool use is delegated to (the model used in the benchmark run). |
| `LIVE_VOICE` | `marin` | Voice name. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q gpt-live/tests
```

`test_parity_gpt_live.py` runs the shared parity assertions, pins the session
shape, and checks the container fails closed at boot without `OPENAI_API_KEY`.
`test_delegation_bridge.py` drives the receive loop with a scripted socket:
audio is forwarded, a delegated function call is dispatched and answered, and
`session.closed` ends the loop.
