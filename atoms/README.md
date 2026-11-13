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
> and turn-taking settings live on the Atoms agent itself. This pod cannot
> provision that for you — see **Required agent configuration** below before
> the first call will behave correctly, and before scoring this candidate
> against anything.

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

Before pointing `ATOMS_AGENT_ID` at a real agent, push all of the following
onto it (console, or the agent's versioned draft → publish → activate flow —
**not** the legacy `PATCH /workflow/{workflowId}`, which bypasses versioning):

1. **Prompt and greeting** — `rory_tools/agent_desc.txt` as the global
   prompt, `rory_tools.prompt.GREETING` as the first message.
2. **Tools, as client tools, not API tools.** Run
   `python -m rory_atoms.tools` from this package to print the exact 16
   definitions (name, description, parameters, wait-for-response, timeout).
   Registering them as Atoms "API tools" instead would have Atoms call them
   directly over HTTP, bypassing `rory_tools.dispatch` — and the harness
   below — entirely, silently grading a different, ungated agent.
3. **Turn-taking settings**, on the agent's Speech tab — mapped from a
   benchmark report's turn-taking-settings findings (the same report this
   transport implements the harness side of):
   | Benchmark setting | Atoms field | Suggested value |
   |---|---|---|
   | 0.6–0.8 s silence before taking the turn | **Release time** | 0.6–0.8 s (repo's existing cross-transport parity target) |
   | MaAI (a model decides whose turn it is) | **Smart turn detection** | On |
   | Noise robustness | **Voice detection: Confidence** | tune up from the Standard default (0.7) against your own noise conditions |
   | R2's "I can't hear you" hang-up | **Handle unrecognised speech** | On |

   Two of the benchmark's R1 findings have **no matching Atoms field** as of
   this writing and are not configurable from here: barge-in requiring
   transcribed words rather than audio alone, and an "addressee gate" that
   drops speech not directed at the agent (e.g. a TV in the background).
   `Allow interruptions` / `Interruption backoff timer` are the nearest
   levers but do not reproduce either one. Flag this as a known gap rather
   than silently dropping it.

There is no public, documented REST field for pushing 2 or 3
programmatically as of this writing (the versioning-lifecycle docs show
`globalPrompt`/`firstMessage`/`synthesizer`/`language`, not a `tools` or
speech-settings field) — this is a manual console step until Smallest
documents one.

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
`provisioning_payload()` (what a human actually pushes to the agent, since
there is no wire-sent tool array to read back the way other transports'
parity tests do), checks the connect URL, and checks the container fails
closed at boot without `SMALLEST_API_KEY` or `ATOMS_AGENT_ID`.
`test_pumps_atoms.py` scripts the receive loop directly: tool-call dispatch
and reply, a truncated-arguments call skipped rather than dispatched, audio
bridging, a non-fatal `interruption`, and a fatal `error` event.
