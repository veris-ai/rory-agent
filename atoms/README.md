# rory-atoms

Rory on Smallest AI's Atoms platform, over its unified realtime websocket.
Transport only — the tools, vendor clients and prompt live in rory-core.

One realtime session per call against `wss://api.smallest.ai/atoms/v1/agent/connect`:
the agent listens, reasons, calls Rory's tools and speaks inside that one
session. The pod bridges the Veris actor's PCM16 to it, declares Rory's 16
tools fresh on every call, runs each tool call through the shared dispatcher,
and posts the result back on the same socket. Nothing dials in.

> [!IMPORTANT]
> The agent behind `ATOMS_AGENT_ID` is provisioned and frozen by Smallest AI
> ahead of time, outside this repo — this pod only ever connects to it; it
> never writes to it. That provisioning is what actually sets the prompt,
> greeting, voice and turn-taking behavior for the run; nothing in this repo
> re-derives or re-asserts those values. Tools are the one thing **not**
> stored on the agent: they are declared per session instead (see below), so
> each call always carries the exact, current schema.

## The wire

- Connect with `agent_id`, `mode=webcall`, and `input_audio_format`/
  `output_audio_format` both pinned to `pcm_24000` — the actor's native
  rate, so neither leg is resampled. The API key rides as an `Authorization`
  header on the connection, not a URL query parameter.
- The server's first message is `session.created`, within a connect
  deadline of its own — a hung connect fails setup rather than blocking the
  call forever. Right after it, this bridge sends `session.update`
  declaring Rory's 16 tools — each one's `parameters` is
  `ToolSchema.parameters()` verbatim, not translated into a lossier shape —
  and resends it on a timer until `session.updated` lists every one of
  them, up to a capped number of attempts and the same overall deadline. The
  resend is purely time-gated, not triggered by every incoming event:
  resending on every event (including routine greeting audio streamed
  during the handshake) could exhaust the whole send budget in well under a
  second, long before a real acknowledgment might arrive.
- Caller audio sent during that connect-and-handshake window is read (so it
  never piles up waiting for us) and dropped (there's nowhere ready to send
  it yet) rather than forwarded once the connection comes up — which would
  otherwise burst into Atoms all at once, looking nothing like real-time
  speech. Conversely, any `output_audio.delta` Atoms streams during that
  same window (the start of its own greeting, say) is buffered and played
  once the actor connection is live, rather than lost.
- Audio streams as `input_audio_buffer.append` (client) /
  `output_audio.delta` (server, field `audio`). Output audio is paced to at
  most 0.3s ahead of when it should play; the pacing clock restarts whenever
  a real silence gap has already passed the point it was scheduled out to,
  rather than drifting from a start time set before the gap. An
  `interruption` event drops whatever is still queued rather than draining it.
- Tool calls arrive as `function_call` (`name`, `call_id`, `arguments` as a
  JSON string) and are answered with `function_call.result`
  (`call_id`, `output` as a JSON string, serialized without escaping
  non-ASCII text) — every call gets an answer, even one whose arguments
  arrived truncated by a barge-in. Each call runs as its own task, so a
  slow one never stops the receive loop from reading audio or an
  interruption in the meantime, but the actual dispatch is serialized
  behind a lock in arrival order: calls share one `CallSession`, and
  running two at once would race its mutation of account/attempt state.
- `session.closed` with reason `client_requested` or `ended` is a normal end
  of call. Any other `session.closed` reason, or the connection ending
  without a `session.closed` we recognized as normal — regardless of its
  numeric close code — is this bridge's own failure, not the agent hanging
  up, and is raised rather than logged as a clean exit. This matters
  because the `websockets` client library does not raise an exception at
  all for some close codes (notably a clean 1000 or 1001): the loop has to
  check explicitly that a normal `session.closed` actually arrived, not
  just that nothing was raised.
- An `error` event reaching the receive loop (`tool_timeout`,
  `unknown_call_id`, `result_too_large`, and the like) is always scoped to
  one call or one tool call, never the session — it is logged and the call
  continues, whatever its code. Only a setup-time failure (a bad connect, a
  rejected tool declaration) ends the call before any of this runs.
- A failed connect or handshake is retried once, after a jittered 2-4s
  delay — except when retrying plainly cannot help: a 4xx on the connection
  upgrade itself, or tools rejected because they are stored on the agent
  instead of declared per session.
- The Atoms `call_id` from `session.created` is logged for every call.

## Required agent configuration

The Atoms agent behind `ATOMS_AGENT_ID` is provisioned and frozen by
Smallest AI outside this repo. This pod only ever connects to it; it never
writes to the agent or stores anything on it. `rory_atoms/tools.py` is the
one place the tool schemas themselves live, declared per session rather
than pushed anywhere ahead of time.

## Running it

```
docker build --platform linux/amd64 -f atoms/Dockerfile -t rory-atoms .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `ATOMS_API_KEY` | — | Required. Sent as an `Authorization: Bearer` header on the Atoms websocket connection. Needs agent-management scope — the same connection path is also how a tool declaration would be rejected as agent-managed, so the key can't be scoped down to connect-only. |
| `ATOMS_AGENT_ID` | — | Required. The provisioned agent (see above) this pod connects every call to. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `STRIPE_API_BASE` | — | The billing twin's base URL; the bench injects this. |
| `FINERACT_API_BASE` | — | The arrangements twin's base URL; the bench injects this. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |
| `RORY_REFERENCE_TIME` | wall-clock time | A frozen instant for anything date-relative Rory's tools compute — set by whatever harness is running the call, not by this transport. |

## Tests

```bash
cd .. && uv run pytest -q atoms/tests
```

`test_parity_atoms.py` runs the shared parity assertions against
`tools.TOOLS` — the exact object sent in every call's `session.update` — and
checks the connect URL, headers, and that the container fails closed at boot
without `ATOMS_API_KEY` or `ATOMS_AGENT_ID`.

`test_connect_atoms.py` scripts the connect handshake: that `session.update`
declares every tool verbatim with its configured timeout, that the resend is
time-gated rather than firing on every event, that greeting audio arriving
before the handshake finishes is buffered rather than dropped, that a
missing `session.created` or `session.updated` is a setup failure within the
connect deadline, and that a retry is skipped outright for a non-retryable
failure (a 4xx upgrade, tools rejected as agent-managed).

`test_pumps_atoms.py` scripts the receive loop directly: arrival-ordered
tool-call dispatch that does not block the loop while in flight, every call
answered (including a truncated one), paced audio delivery that correctly
resumes after a silence gap, a dropped queue on interruption (proven with a
real unsent-audio window, not just a vacuous same-tick push), non-fatal
`error` codes, and normal vs. abnormal session endings — including the
silent-stop case `websockets` 15 doesn't raise an exception for at all.

`test_web_close_codes.py` is a real loopback through the ASGI app and a
local fake Atoms: proof of what close code the actor side actually
receives for 1000, 1001, an abrupt disconnect, and a genuine normal end,
since that silent-stop behavior can only be confirmed end to end.

`tests/live_smoke.py` is not run by the test suite — it is an opt-in,
billable, end-to-end proof against a real Atoms agent. `/voice` carries only
raw audio, so tool-call evidence comes from monkeypatching
`rory_atoms.agent.dispatch` instead of reading the wire:

```bash
ATOMS_API_KEY=... ATOMS_AGENT_ID=... \
  uv run --no-sync python atoms/tests/live_smoke.py --output /tmp/atoms-live.json
```
