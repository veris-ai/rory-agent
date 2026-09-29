# rory-gradium

Rory on [gradbot](https://github.com/gradium-ai/gradbot), Gradium's open-source
voice-agent engine. Transport only — the tools, vendor clients and prompt live
in rory-core.

gradbot is a Rust multiplexer behind a Python API. It runs Gradium streaming
speech-to-text, an OpenAI-compatible LLM (`gpt-4.1-mini`) and Gradium streaming
text-to-speech concurrently, and owns turn-taking and barge-in. Each `/voice`
connection starts its own session; tool calls come back to this process and run
through the shared dispatcher. Only the STT, LLM and TTS requests leave the pod,
so nothing dials in.

## How gradbot shapes the call

- **Instructions are wrapped.** gradbot puts the prompt inside its own system
  scaffold (speaking style, transcription errors, interruptions) and adds an
  internal `reset_asr` tool it handles itself.
- **The first turn is a literal `[start]`.** gradbot's scaffold tells the model
  to greet, while the shared prompt says the greeting already happened. The
  session appends an opening-line section that resolves this and names the
  shared greeting.
- **Endpointing is a flush.** Once transcripts stop for 0.5 s, gradbot pushes
  `flush_duration_s` (0.5 s) of silence through the recogniser before handing the
  turn to the LLM — roughly a one-second end of turn, left on gradbot's defaults.
- **Silence nudges are off.** gradbot re-engages a quiet caller and hangs up
  after three tries by default; no other Rory transport does, so
  `silence_timeout_s` is `0`.

- **Pending tools get re-issued.** A second or two into a pending tool call,
  gradbot re-prompts the model, which often issues the same call again before
  the first result reaches it. The bridge answers a call identical to an
  earlier one (same tool, same arguments) with the earlier result while that
  call is still running, or when the generation issuing it was built from a
  prompt gradbot pushed (`push_to_llm`) before that result went back — a
  prompt can start generating well after it was pushed. An identical call from
  a prompt built after the result went back is a deliberate repeat and runs. Tool calls run in their own
  tasks, one at a time, as in gradbot's reference server.

gradbot's PCM output is 48 kHz; the bridge halves it to the actor's 24 kHz.

## Barge-in

gradbot owns the interruption, in Rust (`gradbot_lib/src/multiplex.rs` and
`llm.rs`, v0.2.0). When STT text or voice activity arrives while the agent is
`Processing`, the multiplexer sets `user_interrupted`, bumps the turn index,
and emits an `interrupted` event. The LLM→TTS task sends one more audio chunk
flagged `interrupted=true` and exits; anything from the old turn still queued
behind it is skipped by turn index, so the bridge never receives it. TTS is
paced 300 ms ahead of real time, so up to 300 ms of audio past the interruption
can already be at the actor. The bridge forwards each chunk as it arrives and
has nothing of its own to flush.

The history is trimmed by the engine: a text segment is only recorded as
`transmitted` when the audio carrying it has been sent, and the assistant
message kept for the next prompt is built from those segments alone, so it
reflects what was spoken at TTS-segment granularity, not the full LLM reply.
gradbot's scaffold tells the model an interrupted message ends with "—", but
v0.2.0 appends no dash; the truncation itself is what the model sees. The
Python API exposes no way to read or edit the history, so nothing here
touches it — and nothing needs to.

## Trace evidence

The LLM request is made from Rust, not a Python SDK, so OpenTelemetry
instrumentation in the pod sees no model calls for this candidate. Tool calls
are logged by the bridge as they run.

## Running it

```
docker build --platform linux/amd64 -f gradium/Dockerfile -t rory-gradium .
```

The image serves `/voice` and `/health` on `:8008`. gradbot ships wheels for linux x86_64, not
aarch64, so build for `linux/amd64`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `GRADIUM_API_KEY` | — | Required. Gradium STT and TTS. |
| `OPENAI_API_KEY` | — | Required. The LLM behind gradbot. |
| `GRADIUM_LLM_MODEL` | `gpt-4.1-mini` | OpenAI model name. |
| `GRADIUM_VOICE_ID` | `4SZHfMpw-p46Ywgs` | Gradium voice ("Harper"). |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run --package rory-gradium pytest -q gradium/tests
```

`test_parity_gradium.py` runs the shared parity assertions, pins the opening
line and the silence override, and checks the container fails closed at boot
without its keys. `test_lifecycle_gradium.py` stubs `gradbot.run`: actor
frames reach gradbot untouched, 48 kHz audio leaves at 24 kHz, tool calls run
through the shared dispatcher, and a re-issued call is answered with the
earlier result.
