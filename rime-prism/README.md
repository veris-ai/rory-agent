# rory-rime-prism

Rory on Rime's Prism speech-to-speech model. Transport only — the tools, vendor
clients and prompt live in rory-core.

One Prism session per call, opened through the Rime Python SDK
(`rimelabs-sdk`), which is how Rime recommends a Python backend use Prism. The
session carries the shared system prompt and Rory's 16 tools. The SDK's
conversation runner then bridges the Veris actor's PCM16 to it: the actor's
audio is the runner's source, and a paced player that writes to the actor is its
playback. Nothing dials in.

## How it uses Prism

- **Turn-taking is Prism's.** Prism decides when the caller has finished, and
  caller speech over a reply interrupts it. Interruption and turn detection stay
  on Prism's defaults (eagerness `auto`, a 5 s end-of-turn timeout). Rime's
  guidance is to add no turn detector or VAD of your own.
- **Audio.** The actor runs PCM16 at 24 kHz both ways. Prism listens at 16 kHz,
  and the SDK resamples the actor's audio itself. Prism replies at 24 kHz, so
  that leg is not resampled.
- **Playback reports.** Prism sends a reply faster than it plays. The player
  sends it to the actor in 100 ms slices, at most 0.5 s ahead of real time, so
  a barge-in can still cut it. Audio sent to the actor counts as heard, and the
  runner reports that to Prism, which keeps only the heard words of an
  interrupted reply.
- **Greeting.** Prism has no welcome message. A nudge (`request_reply`) asks it
  to say the shared greeting. Prism generates the audio, so it can paraphrase.
  Tool calls run while it waits, and Prism refuses it if the caller speaks
  first.
- **Tools.** Each tool is a `ToolDefinition` built from the shared schema, with
  no `prism_filler` set, so Prism says its own short phrase ("One moment.")
  while a tool runs. As in Rime's tools guide, each call starts on its own task
  as soon as Prism makes it, runs through the shared dispatcher, and submits
  its result; the reply that made the calls is continued once all of them are
  in. Lookups run side by side. Writes (`WRITE_TOOLS`: payments, plans,
  extensions, paperless) run one at a time, because Prism can ask for the same
  write again while the first is still running, and the shared tools refuse
  the repeat only once the first has finished. No call is ever run again.
- **Slow tools.** As in Rime's slow-tool guide, a lookup still running after
  4 s gets a nudge for one short "still working" line, and another 10 s later.
  The lines stop if Prism refuses one. At 30 s the lookup gets an error result.
- **Results of closed rounds.** As in Rime's guide to late results, when
  Prism refuses to continue a round (the caller spoke first, a still-working
  line closed it, or a wait ran out), a nudge reports each of its results
  (`request_reply(tool_call=...)`). Prism keeps a result but mentions it only
  if asked, so without the report the caller hears silence after the tool's
  phrase. A round whose reply did not complete is dropped, as in Rime's tools
  guide.

## Running it

```
docker build --platform linux/amd64 -f rime-prism/Dockerfile -t rory-rime-prism .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `RIME_API_KEY` | — | Required. The SDK sends it as `Authorization: Bearer`. |
| `PRISM_VOICE` | `aurora` | Prism voice: `aurora`, `wawona`, `brando` or `astra`. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q rime-prism/tests
```

`test_parity_rime_prism.py` runs the shared parity assertions, pins the session
settings above, and checks the container fails closed at boot without
`RIME_API_KEY`. `test_tool_rounds_rime_prism.py` drives the tool loop against a
fake session, and `test_playback_rime_prism.py` pins what the player reports.
