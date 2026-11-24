# rory-grok-voice

Rory on xAI's Grok speech-to-speech API. Transport only — the tools, vendor
clients and prompt live in rory-core.

One `grok-voice-3.0` realtime websocket per call: the model listens,
reasons, calls Rory's tools and speaks inside that one session. The pod bridges
the Veris actor's PCM16 to it, runs each tool call locally through the shared
dispatcher, and posts the result back on the same socket. Nothing dials in.

## The wire, relative to OpenAI Realtime

Grok speaks the OpenAI Realtime protocol — the same event names in both
directions, so this transport is the Realtime one with two deltas in
`session.update`:

- `voice`, `instructions` and `turn_detection` sit at the session's top level
  rather than under `audio`.
- Caller transcripts only arrive when `audio.input.transcription.model` is
  `grok-transcribe`. Without it the caller's side of the call never reaches
  the log or the trace.

End-of-turn detection stays on xAI's `server_vad` default with nothing pinned,
unlike the other Rory transports, which pin 800 ms of silence. Both legs run at
the actor's 24 kHz, so no audio is resampled.

## Barge-in

Grok generates audio faster than real time, and the `voice_ws` protocol cannot
take audio back once it is sent. So the bridge holds agent audio and sends it to
the actor at the rate it is spoken. On `input_audio_buffer.speech_started` it
drops what it has not sent and sends `conversation.item.truncate` with
`audio_end_ms` at the amount it did send, so the model's context matches what
the caller heard.

## Tool calls

Tools run in a worker, so a slow tool never stops the bridge reading Grok's
events. Only a response that finishes with `status: "completed"` runs its tools.
Every `call_id` gets a `function_call_output`, including calls with invalid JSON.

Once all outputs are sent and the current audio has played, the bridge sends one
`response.create`. If the caller spoke during the round, server VAD's own
response uses the outputs, and the bridge only asks if no such response arrives.

`error` events are logged and do not end the call. A session that is really over
closes its websocket.

## Prompt and greeting

The greeting is a `force_message`, so it is spoken word for word without calling
the model. The prompt is the shared one with one sentence changed: callers may
read an account number with or without the dash, and the agent still sends it to
`verify_caller` in the dashed form the lookup needs.

## Running it

```
docker build --platform linux/amd64 -f grok-voice/Dockerfile -t rory-grok-voice .
```

The image serves `/voice` and `/health` on `:8008`. Grok's frames follow the
OpenAI Realtime protocol, so a trace reader built for Realtime reads this
transport's tool calls too.

## Environment

| Variable | Default | Role |
|---|---|---|
| `XAI_API_KEY` | — | Required. |
| `GROK_VOICE_MODEL` | `grok-voice-3.0` | The speech-to-speech model in the websocket URL. |
| `GROK_VOICE` | `eve` | Voice name. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q grok-voice/tests
```

`test_parity_grok_voice.py` runs the shared parity assertions, pins the
session shape and the one prompt difference, and checks the container fails
closed at boot without `XAI_API_KEY`. `test_cancelled_tool_call_grok_voice.py`
covers the tool round and `test_barge_in_grok_voice.py` covers playout and
truncation.
