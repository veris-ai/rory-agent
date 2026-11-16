# rory-atoms

Rory on Smallest AI's Atoms platform, over its unified realtime websocket.
Transport only — the tools, vendor clients and prompt live in rory-core.

One realtime session per call against `wss://api.smallest.ai/atoms/v1/agent/connect`:
the agent listens, reasons, calls Rory's tools and speaks inside that one
session. The pod bridges the Veris actor's PCM16 to it, runs each tool call
locally through the shared dispatcher (and the harness below), and posts the
result back on the same socket. Nothing dials in.

> [!IMPORTANT]
> Unlike every other speech-to-speech candidate in this repo, Atoms takes
> **no configuration at all on the connection**. The prompt, greeting, voice,
> turn-taking settings, and tools all live on the Atoms agent itself — run
> `atoms/scripts/provision_agent.py` (see **Required agent configuration**
> below) before pointing `ATOMS_AGENT_ID` at it, or every call will run
> against an agent with no tools at all.

## The wire

Confirmed against Smallest's public docs (the WebSocket SDK page, the
Realtime Agent WebSocket API reference, and the client-tools page):

- Connect with `token`, `agent_id`, `mode=webcall`, and
  `input_audio_format`/`output_audio_format` both pinned to `pcm_24000` — the
  actor's native rate, so neither leg is resampled.
- The server's first message is `session.created` directly — there is no
  configure step to wait on, because there is nothing left to configure
  per-connection.
- Audio streams as `input_audio_buffer.append` (client) /
  `output_audio.delta` (server, field `audio`). There is no commit message
  needed in this mode: the docs describe the server as handling VAD and turn
  boundaries on its own.
- Tool calls arrive as `function_call` (`name`, `call_id`, `arguments` as a
  JSON string) and are answered with `function_call.result`
  (`call_id`, `output` as a JSON string). There is no response-resume message
  to send afterward — Atoms' documented client message list has nothing like
  it, so the agent is expected to continue on its own once it has the
  result.
- `interruption` announces a barge-in; the server owns stopping its own
  generation, so this bridge only stops receiving further
  `output_audio.delta` for that turn — there is nothing documented to send
  back.

## Required agent configuration

Before pointing `ATOMS_AGENT_ID` at a real agent, it needs the following —
all of it scriptable, confirmed by actually running this against a live
agent rather than from docs (which don't enumerate the full schema):

```bash
SMALLEST_API_KEY=... uv run python -m atoms.scripts.provision_agent
```

This creates an agent (or reuses one via `--agent-id`), forks a branch from
whichever is currently live, and pushes everything below through
`PUT /agent/{id}/branches/{branchId}/draft` — then publishes and activates
it. It prints the resulting id to use as `ATOMS_AGENT_ID`.

1. **Prompt and greeting** — `rory_tools/agent_desc.txt` as `globalPrompt`,
   `rory_tools.prompt.GREETING` as `firstMessage`.
2. **Turn-taking settings** — mapped from a benchmark report's
   turn-taking-settings findings (the same report this transport implements
   the harness side of), via the `voiceDetectionConfig`/`smartTurnConfig`/
   `sttMissDetectorConfig` fields `GET /agent/{id}` already shows on every
   agent:
   | Benchmark setting | Atoms field | Value this script pushes |
   |---|---|---|
   | 0.6–0.8 s silence before taking the turn | `voiceDetectionConfig.releaseTimeInSecs` | 0.7 (default is 0.2) |
   | MaAI (a model decides whose turn it is) | `smartTurnConfig.isEnabled` | `true` (already the default) |
   | R2's "I can't hear you" hang-up | `sttMissDetectorConfig.isEnabled` | `true` (already the default) |

   Two of the benchmark's R1 findings have **no matching Atoms field** as of
   this writing and are not configurable from here: barge-in requiring
   transcribed words rather than audio alone, and an "addressee gate" that
   drops speech not directed at the agent (e.g. a TV in the background).
   `allowInterruptions` / `interruptionBackoffTimer` are the nearest levers
   but do not reproduce either one. Flag this as a known gap rather than
   silently dropping it.
3. **Tools, as client tools, not API tools** — the one that took the most
   digging. A bare top-level `tools` field (or `workflowTools`, or nested
   under a plain `workflow` key, or a guessed `.../draft/workflow` route) is
   rejected with `{"status": false, "errors": ["No recognized config fields
   in request body"]}` or 404s. What actually works: nesting under
   `singlePromptConfig` alongside the type discriminator this agent needs —
   `{"type": "single_prompt", "singlePromptConfig": {"prompt": ...,
   "tools": [...]}}`. See `rory_atoms/tools.py`'s module docstring for why
   client tools matter (not API tools) and the exact per-tool shape
   (`llmParameters`, `text`/`number`/`boolean` types — not JSON Schema's).

## The harness

`rory_tools/harness.py` (shared — applies to every transport in this repo,
not only this one) ports a benchmark finding: a rule layer between a tool
call and its execution was the largest lever measured on Atoms, ahead of
turn-taking settings and MaAI. It is wired into `rory_tools.dispatch`, so it
runs here exactly as it does for every other candidate. See that module's
docstring for exactly which of the benchmark's four harness-v2 rules ported
as-is, which had to be weakened because no transport in this repo (this one
included) exposes a transcript to check against, and which don't apply to
Rory's actual 16 tools at all.

## Running it

```
docker build --platform linux/amd64 -f atoms/Dockerfile -t rory-atoms .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `SMALLEST_API_KEY` | — | Required. Rides as the `token` query parameter on the Atoms websocket URL. |
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
passing or failing it. Also checks the connect URL, and that the container
fails closed at boot without `SMALLEST_API_KEY` or `ATOMS_AGENT_ID`.
`test_pumps_atoms.py` scripts the receive loop directly: tool-call dispatch
and reply, a truncated-arguments call skipped rather than dispatched, audio
bridging, a non-fatal `interruption`, and a fatal `error` event.

`scripts/provision_agent.py` is a dev tool, not part of the image — it needs
a real `SMALLEST_API_KEY` and talks to the live Atoms API, so it isn't run
by the test suite.
