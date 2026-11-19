# rory-atoms

Rory on Smallest AI's Atoms platform, over its unified realtime websocket.
Transport only — the tools, vendor clients and prompt live in rory-core.

One realtime session per call against `wss://api.smallest.ai/atoms/v1/agent/connect`:
the agent listens, reasons, calls Rory's tools and speaks inside that one
session. The pod bridges the Veris actor's PCM16 to it, runs each tool call
through the shared dispatcher, and posts the result back on the same socket.
Nothing dials in.

> [!IMPORTANT]
> Unlike every other speech-to-speech candidate in this repo, Atoms takes
> **no configuration at all on the connection**. The prompt, greeting, voice,
> turn-taking settings, and tools all live on the Atoms agent itself — run
> `atoms/scripts/provision_agent.py` (see **Required agent configuration**
> below) against an already-provisioned agent id before pointing
> `ATOMS_AGENT_ID` at it.

## The wire

Confirmed against Smallest's public docs (the WebSocket SDK page, the
Realtime Agent WebSocket API reference, and the client-tools page):

- Connect with `agent_id`, `mode=webcall`, and `input_audio_format`/
  `output_audio_format` both pinned to `pcm_24000` — the actor's native
  rate, so neither leg is resampled. The API key rides as an `Authorization`
  header on the connection, not a URL query parameter.
- The server's first message is `session.created` directly — there is no
  configure step to wait on, because there is nothing left to configure
  per-connection.
- Audio streams as `input_audio_buffer.append` (client) /
  `output_audio.delta` (server, field `audio`). Output audio is paced to at
  most 0.3s ahead of when it should play, and an `interruption` event drops
  whatever is still queued rather than draining it.
- Tool calls arrive as `function_call` (`name`, `call_id`, `arguments` as a
  JSON string) and are answered with `function_call.result`
  (`call_id`, `output` as a JSON string) — every call gets an answer, even
  one whose arguments arrived truncated by a barge-in. Each call runs as its
  own task rather than blocking the rest of the session.
- `session.closed` with reason `client_requested` or `ended` is a normal end
  of call — any other reason (or the websocket itself closing abnormally) is
  treated as this bridge's own failure, not the agent hanging up, and is
  raised rather than logged as a clean exit.
- A failed connect or handshake is retried once before giving up.

## Required agent configuration

`atoms/scripts/provision_agent.py` configures an **existing** agent — it
never creates one and never runs automatically; `--agent-id` (or
`ATOMS_AGENT_ID`) is required with no fallback:

```bash
ATOMS_API_KEY=... ATOMS_AGENT_ID=... uv run python -m atoms.scripts.provision_agent
```

It forks a branch from whichever is currently live, pushes the prompt
(`rory_tools/agent_desc.txt`, with a date line appended at provisioning
time — the prompt lives on the agent rather than being rebuilt per call, so
re-run this to refresh it), the greeting, voice and turn-taking settings,
and Rory's 16 tools as Atoms client tools (plus the platform's own
call-ending tool) — then publishes and activates the branch.

Turn-taking is set to: VAD confidence 0.7, min volume 0.6, trigger 0.2s,
release 0.2s; smart turn detection on, wait 2.0s; a 900s session cap; an
idle prompt after 20s, up to 3 times. See `rory_atoms/tools.py`'s module
docstring for the per-tool shape pushed (`llmParameters`, `text`/`number`/
`boolean` types — not JSON Schema's) and the one JSON-Schema constraint
(`make_payment.amount_cents`'s `minimum`) that has nowhere to go in it.

## Running it

```
docker build --platform linux/amd64 -f atoms/Dockerfile -t rory-atoms .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `ATOMS_API_KEY` | — | Required. Sent as an `Authorization: Bearer` header on the Atoms websocket connection and REST calls. |
| `ATOMS_AGENT_ID` | — | Required. The provisioned agent (see above) this pod connects every call to. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q atoms/tests
```

`test_parity_atoms.py` runs the shared parity assertions against
`provisioning_payload()` — what `provision_agent.py` actually pushes to the
agent, since there's no per-call wire-sent tool array to read back the way
other transports' parity tests do — including an explicit, separate check
for `make_payment`'s one unrepresentable constraint rather than silently
passing or failing it. Also checks the connect URL and headers, and that
the container fails closed at boot without `ATOMS_API_KEY` or
`ATOMS_AGENT_ID`.

`test_pumps_atoms.py` scripts the receive loop directly: concurrent
tool-call dispatch with every call answered (including a truncated one),
paced audio delivery, a dropped queue on interruption, and normal vs
abnormal session endings.

`scripts/provision_agent.py` and `tests/live_smoke.py` are not run by the
test suite — the first needs a real agent id and writes to it; the second
is an opt-in, billable, end-to-end proof. `live_smoke.py` boots the real
`/voice` ASGI app against disposable Stripe/Fineract fixtures
(`tests/live_backends.py`), bridges to a real provisioned Atoms agent, and
drives a full conversation with synthesized speech. `/voice` carries only
raw audio, so tool-call evidence comes from monkeypatching
`rory_atoms.agent.dispatch` instead of reading the wire:

```bash
ATOMS_API_KEY=... ATOMS_AGENT_ID=... \
  uv run --no-sync python atoms/tests/live_smoke.py --output /tmp/atoms-live.json
```
