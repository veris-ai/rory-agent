# rory-phonic

Rory on Phonic's speech-to-speech API. Transport only — the tools, vendor
clients and prompt live in rory-core.

One Phonic conversation websocket per call. The pod connects to
`wss://api.phonic.ai/v1/sts/ws`, sends one `config` carrying the shared system
prompt, greeting and Rory's 16 tools, and then bridges the Veris actor's PCM16
to it. Phonic speaks the greeting first. Each tool call runs locally through the
shared dispatcher, and its result always goes back on the same socket as a
`tool_call_output`. Nothing dials in.

## The wire

Phonic is its own protocol, not OpenAI Realtime-compatible. The transport uses:

- The pod sends `config` first, then reads `ready_to_start_conversation`. Phonic sends nothing before the config.
- `audio_chunk` both ways, base64 PCM16. The first chunk from the actor also
  carries `iso_date_time`, which Phonic requires once per conversation.
- `input_text` for the caller's transcript and `audio_chunk.text` for Rory's.
- `tool_call` → `tool_call_output` on the same `tool_call_id`, with the tool
  declared inline as `custom_websocket`, `execution_mode: "sync"`,
  `uninterruptible: true`. Without the flag, caller speech during a call cancels
  it. A call that has already acted (a payment, a new plan) is then asked for
  again and recorded twice.

Audio is `pcm_24000` both ways, so neither leg is resampled. Turn-taking is
pinned to 800 ms of end-of-turn silence (`vad_min_silence_duration_ms`). Phonic's
other VAD settings stay on its defaults.

The tool output timeout is 30 s, because the vendor clients are blocking and a
slow call can take a while.

The model is Phonic's `phonic-1` at `intelligence_level: high`. Phonic's built-in
`choose_not_to_respond` tool is enabled with a 3 s response delay. It sits outside
the shared 16 tools, so the parity surface is unchanged. The judge will not see a
tool call for it; a turn where the agent chose not to respond shows up only as
silence in the trace and a log line.

## Running it

```
docker build --platform linux/amd64 -f phonic/Dockerfile -t rory-phonic .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `PHONIC_API_KEY` | — | Required. Sent as `Authorization: Bearer`. |
| `PHONIC_VOICE` | `sabrina` | Phonic voice id. |
| `PHONIC_MODEL` | `phonic-1` | Phonic speech-to-speech model. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q phonic/tests
```

`test_parity_phonic.py` runs the shared parity assertions, pins the config
shape above, and checks the container fails closed at boot without
`PHONIC_API_KEY`.
