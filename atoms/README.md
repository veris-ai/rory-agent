# rory-atoms

Rory on Smallest AI's Atoms platform, over its unified realtime websocket.
Transport only — the tools, vendor clients and prompt live in rory-core.

One realtime session per call against `wss://api.smallest.ai/atoms/v1/agent/connect`:
the agent listens, reasons, calls Rory's tools and speaks inside that one
session. The pod bridges the Veris actor's PCM16 to it, declares Rory's 16
tools fresh on every call, runs each tool call through the shared dispatcher,
and posts the result back on the same socket. Nothing dials in.

> [!IMPORTANT]
> Prompt, greeting, voice and turn-taking settings live on the Atoms agent
> itself, provisioned and frozen ahead of time — `ATOMS_AGENT_ID` points at
> an already-configured agent; this pod never writes to it. Tools are the one
> thing **not** stored on the agent: they are declared per session instead
> (see below), so each call always carries the exact, current schema.

## The wire

- Connect with `agent_id`, `mode=webcall`, and `input_audio_format`/
  `output_audio_format` both pinned to `pcm_24000` — the actor's native
  rate, so neither leg is resampled. The API key rides as an `Authorization`
  header on the connection, not a URL query parameter.
- The server's first message is `session.created`. Right after it, this
  bridge sends one `session.update` declaring Rory's 16 tools — each one's
  `parameters` is `ToolSchema.parameters()` verbatim, not translated into a
  lossier shape — and waits for `session.updated` before any audio moves.
- Audio streams as `input_audio_buffer.append` (client) /
  `output_audio.delta` (server, field `audio`). Output audio is paced to at
  most 0.3s ahead of when it should play; the pacing clock restarts whenever
  a real silence gap has already passed the point it was scheduled out to,
  rather than drifting from a start time set before the gap. An
  `interruption` event drops whatever is still queued rather than draining it.
- Tool calls arrive as `function_call` (`name`, `call_id`, `arguments` as a
  JSON string) and are answered with `function_call.result`
  (`call_id`, `output` as a JSON string) — every call gets an answer, even
  one whose arguments arrived truncated by a barge-in. Calls are dispatched
  one at a time, in the order they arrived: they share one `CallSession`,
  and running two at once would race its mutation of account/attempt state.
- `session.closed` with reason `client_requested` or `ended` is a normal end
  of call. Any other `session.closed` reason, any `error` outside the
  non-fatal set below, or the websocket closing without a `session.closed`
  we recognized as normal — regardless of its numeric close code — is this
  bridge's own failure, not the agent hanging up, and is raised rather than
  logged as a clean exit.
- `error` events for `tool_timeout`, `unknown_call_id`, or `result_too_large`
  are scoped to one call, not the session: logged, and the call continues.
  Any other `error` is treated as a setup/session failure and raised.
- A failed connect or handshake (including a missing `session.updated`
  acknowledgment) is retried once before giving up.
- The Atoms `call_id` from `session.created` is logged for every call.

## Required agent configuration

The Atoms agent behind `ATOMS_AGENT_ID` is provisioned and frozen outside
this repo — prompt, greeting, voice and turn-taking settings all live on it
already. This pod only ever connects to it; it never writes to the agent or
stores anything on it. `rory_atoms/tools.py` is the one place the tool
schemas themselves live, declared per session rather than pushed anywhere
ahead of time.

## Running it

```
docker build --platform linux/amd64 -f atoms/Dockerfile -t rory-atoms .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `ATOMS_API_KEY` | — | Required. Sent as an `Authorization: Bearer` header on the Atoms websocket connection. |
| `ATOMS_AGENT_ID` | — | Required. The provisioned agent (see above) this pod connects every call to. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q atoms/tests
```

`test_parity_atoms.py` runs the shared parity assertions against
`tools.TOOLS` — the exact object sent in every call's `session.update` — and
checks the connect URL, headers, and that the container fails closed at boot
without `ATOMS_API_KEY` or `ATOMS_AGENT_ID`.

`test_connect_atoms.py` scripts the connect handshake: that `session.update`
declares every tool verbatim, that a missing `session.updated` acknowledgment
is treated as a setup failure, and that a failed handshake is retried once.

`test_pumps_atoms.py` scripts the receive loop directly: sequential,
arrival-ordered tool-call dispatch with every call answered (including a
truncated one), paced audio delivery that correctly resumes after a silence
gap, a dropped queue on interruption, non-fatal vs. fatal `error` codes, and
normal vs. abnormal session endings.

`tests/live_smoke.py` is not run by the test suite — it is an opt-in,
billable, end-to-end proof against a real Atoms agent. `/voice` carries only
raw audio, so tool-call evidence comes from monkeypatching
`rory_atoms.agent.dispatch` instead of reading the wire:

```bash
ATOMS_API_KEY=... ATOMS_AGENT_ID=... \
  uv run --no-sync python atoms/tests/live_smoke.py --output /tmp/atoms-live.json
```
