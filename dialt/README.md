# Rory on Dialt

Dialt owns listening, turn detection, reasoning, speech and interruption
detection over one WebSocket per call. Rory supplies the unchanged shared
prompt, greeting, frozen-clock date context and 16 business tools. Tool calls
execute sequentially through `rory_tools.dispatch` with a fresh `CallSession`
per connection. No separate speech or language-model credentials are needed.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DIALT_API_KEY` | required | Backend Dialt API credential |
| `DIALT_BRAIN` | `fast` | `fast` = Dialt Fast; `smart` = Dialt Smart |
| `DIALT_VOICE` | `us_female` | Stock voice key from Dialt's voice catalogue |
| `STRIPE_API_KEY` | required | Shared Rory billing backend credential |
| `STRIPE_API_BASE`, `FINERACT_API_BASE` | shared defaults | Benchmark-injected twin URLs |

Use the same image with `DIALT_BRAIN=fast` or `DIALT_BRAIN=smart` to evaluate
the two public tiers. Invalid tier names fail at boot. Credentials belong in
the runtime environment, never in the image or source tree.

```bash
uv sync --package rory-dialt --no-dev
uv run --no-sync uvicorn rory_dialt.web:app --host 0.0.0.0 --port 8008

docker build --platform linux/amd64 -f dialt/Dockerfile -t rory-dialt .
# Keep the runtime credential file outside the repository.
docker run --rm --env-file /path/to/runtime.env -p 8008:8008 rory-dialt
```

The benchmark polls `GET /health` and connects to `WS /voice`: raw PCM16,
24 kHz mono, binary frames in both directions. The bridge resamples input
continuously to Dialt's 16 kHz and requests 24 kHz output, checking the server's
`ready` format before sending microphone audio. The shared greeting starts
the call. Server turn detection is used without an endpointing override.

Output is paced in 20 ms slices. Hard interruptions discard locally queued
audio; soft interruptions let it drain. Both report playback state to Dialt,
and late audio for an interrupted response is ignored. `voice_ws` has no remote
playback receipts or clear-buffer command: audio already sent to the actor cannot be recalled,
and sent bytes are not proof of physical playback. Normal Dialt closure drains
local audio; errors fail the call. There is no automatic reconnect or tool
replay, to avoid repeating account or payment operations.
Cancelled tools are skipped if still queued. An already running blocking
vendor request cannot be undone; its result is not sent after cancellation.

Tool results preserve the shared payload. Explicit negative verification or
payment results report `failed`; successful results report `succeeded` and
`verified: true`. Generic errors report `unknown`, unverified: a write may have
committed before its response or subsequent refresh failed. This conservative
mapping also leaves generic vendor refusals unverified rather than guessing
from exception text whether a write happened.

## Background speaker suppression

This transport requests agent control **on**, with suppression initially
**off**, using the session-only beta start-mode fields
`agent_background_speaker_suppression: true` and
`background_speaker_suppression: false`. The managed
`set_background_speaker_suppression` tool is declared and executed by Dialt;
it is not added to Rory's business-tool manifest or dispatcher.

These are beta WebSocket extensions, not public Python SDK fields. The bridge
uses raw WebSocket messages rather than modifying or extending the SDK's
contract. The server authorizes the request; startup rejection fails the call
instead of silently running a different configuration. Suppression can remove
quiet caller words and reduce recognition accuracy, including in clean audio.
Agent control lets the model enable or disable it during the call; it does
not guarantee better benchmark results.

## References and tests

- [Dialt WebSocket guide](https://dialt.com/docs/api/websocket/)
- [Python SDK reference](https://dialt.com/docs/api/python/)
- [Realtime API contract](https://dialt.com/docs/api/reference/realtime/)

```bash
uv sync
uv run pytest dialt/tests -q
```

### Opt-in live checks (billable)

With `DIALT_API_KEY` set, the following starts finite loopback ASGI and HTTP
fixture servers, calls the actual `/health` and `/voice` routes, and connects
through a transparent observing relay to production Dialt. It runs both tiers
by default (`--brains fast` or `--brains smart` selects one). It never changes
Rory's prompt, tools, dispatcher or vendor clients. All Stripe/Fineract URLs
and credentials are overwritten with disposable local fixtures; no real
financial writes occur. The fixtures implement only this scenario, **not the
complete hosted Veris twin benchmark**.

```bash
uv sync --package rory-dialt --no-dev --group test
uv run --no-sync python dialt/tests/live_smoke.py --output /tmp/dialt-voice.json

# Separately labelled diagnostic: bypass ASR for business turns using the
# public acknowledged user-text injection API. Suppression requests and
# barge-in still use audio through /voice. This is NOT full voice E2E.
uv run --no-sync python dialt/tests/live_smoke.py --typed-business --output /tmp/dialt-typed.json
```

The scenario checks rejected and accepted identity, account lookup, paperless
billing, a declined payment followed by an explicitly authorized successful
payment, interruption and agent-controlled suppression on/off. One spoken
confirmation is supplied if the model asks before an action. A timeout or
unexpected result fails the run; recognition or model variability is not
silently retried. Both tiers are attempted even when one fails.

JSON evidence includes source hashes, synthetic transcripts, tool results,
suppression/playback receipts and disposable backend state. It excludes API
keys, resume tokens and server-internal configuration. Synthetic caller PCM
is cached in `/tmp/rory-dialt-live-audio` (override with `--audio-cache`) for
repeatability and to avoid paying to regenerate it on every run. Keep evidence
and cached audio outside the repository. Local tests additionally force clean
and abnormal closure, mixed response/turn identifiers, queued tool cancellation
and uncertain writes; these deterministic peer tests are not production proof.
