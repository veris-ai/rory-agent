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
- **Tools.** Each tool is a `ToolDefinition` built from the shared schema, with
  no `prism_filler` set, so Prism says its own short phrase ("One moment.")
  while a tool runs. Calls run one at a time through the shared dispatcher,
  each result is submitted, and the reply that made them is continued. No call
  is ever run again.
- **Slow tools.** A lookup still running after 4 s gets a nudge for one short
  "still working" line, and another 10 s later, as in Rime's slow-tool guide.
  Only `create_payment_arrangement` (about 9 s) runs that long.
- **Results of closed rounds.** A round can close before it uses its result:
  the caller spoke first (even an "okay"), the reply that made the call was
  cancelled, or a still-working line closed it. Prism keeps the result but
  mentions it only if asked. Once Prism is idle, a nudge reports the result
  (`request_reply(tool_call=...)`), as Rime's guide says. It is skipped when a
  newer round is open, or when Prism has answered the caller since the result
  went in. Prism takes a nudge only once it is idle, so a report or a
  still-working line still waiting is withdrawn if Prism starts answering the
  caller first. In the smoke runs, an unreported result left a verified caller
  in silence until they hung up, and a nudge Prism took after starting to
  answer the caller closed the round that answer opened, so the caller heard
  the same answer twice.

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
