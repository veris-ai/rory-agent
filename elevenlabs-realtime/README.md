# rory-elevenlabs-realtime

Rory on [Eleven Realtime](https://elevenlabs.io/blog/realtime), using ElevenLabs'
standalone, OpenAI-compatible Realtime WebSocket API. The prompt, greeting,
16 tools, verification gate, and vendor clients come from `rory-core`.

## Models

| `ELEVENLABS_REALTIME_MODEL` | Use |
|---|---|
| `eleven_realtime_v1_max` (default) | Complex instructions and multi-step tool use, suited to Rory's billing and payment workflow. |
| `eleven_realtime_v1_mini` | Lower cost and latency. |

ElevenLabs describes these as cascaded systems optimized together. Selecting
one model configures recognition, turn-taking, reasoning, and speech.

## Running

From the repository root, export `ELEVENLABS_API_KEY` and `STRIPE_API_KEY`,
then run:

```bash
uv sync
uv run uvicorn rory_elevenlabs_realtime.web:app --host 0.0.0.0 --port 8008
```

Or build the standalone image from the repository root:

```bash
docker build --platform linux/amd64 -f elevenlabs-realtime/Dockerfile -t rory-elevenlabs-realtime .
```

The server exposes `GET /health` and `WS /voice` on `:8008`. Audio is raw
PCM16, 24 kHz mono in both directions, with no resampling. Missing provider
or billing credentials fail startup before the health probe succeeds.

## Environment

| Variable | Default | Role |
|---|---|---|
| `ELEVENLABS_API_KEY` | — | Required ElevenLabs API key. |
| `ELEVENLABS_REALTIME_MODEL` | `eleven_realtime_v1_max` | Max or Mini model ID. |
| `ELEVENLABS_REALTIME_VOICE` | `brian` | ElevenLabs voice. |
| `STRIPE_API_KEY` | — | Required billing sandbox credential. |
| `STRIPE_API_BASE` / `FINERACT_API_BASE` | Vendor API URLs | Injected URLs for the Acme Energy twins; see the root README. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Session lifecycle

Each call mints an ephemeral client secret with its model, shared prompt,
current frozen-clock date, tools, and audio configuration at
`POST /v1/convai/realtime/client_secrets`. It then authenticates to
`wss://api.elevenlabs.io/v1/convai/realtime` using the same WebSocket
subprotocol as the documented browser SDK flow. The initial token-scoped
configuration is authoritative; a dashboard agent ID is unnecessary.

Rory greets first by adding a user text item requesting the shared greeting,
then sending `response.create`; ElevenLabs needs conversation input before it
will generate a reply. ElevenLabs requires semantic VAD; this transport uses
medium eagerness and interrupts responses when the caller speaks. The bridge
streams audio deltas immediately and sends
`conversation.item.truncate` at the amount forwarded when the caller interrupts
an active response.

Tools run sequentially through the shared dispatcher on worker threads, after
`response.done` confirms a completed response. Cancelled responses execute no
tools. Tool outputs are returned as `function_call_output` items, followed by
one `response.create` to continue. Provider errors propagate to a failed
voice connection, and ending either audio pump cancels the other.

This transport connects directly to the Realtime API. The existing
[`elevenlabs`](../elevenlabs/) implementation uses a dashboard-hosted
ElevenAgents agent. Knowledge bases, deployment channels, and other
ElevenAgents platform features belong to that hosted integration.

References: [API compatibility guide](https://elevenlabs.io/docs/eleven-agents/guides/openai-realtime-compatibility),
[client secret schema](https://elevenlabs.io/docs/eleven-agents/api-reference/conversations/create-client-secret).

## Tests

```bash
uv run pytest -q elevenlabs-realtime/tests
```

The offline tests cover tool and prompt parity, authentication, audio routing,
greeting and disconnect lifecycle, sequential tool dispatch, cancelled calls,
barge-in, and provider failures. A live call requires ElevenLabs credentials
and access to Realtime models for the workspace.
