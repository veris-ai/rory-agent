# rory-hydra

Rory on Smallest AI's Hydra speech-to-speech model. Transport only — the
tools, vendor clients and prompt live in rory-core.

One `hydra-v1.1` realtime websocket per call: the model listens, reasons,
calls Rory's tools and speaks inside that one session. The pod bridges the
Veris actor's PCM16 to it, runs each tool call locally through the shared
dispatcher, and posts the result back on the same socket. Nothing dials in.

## The wire

Hydra's handshake is its own: connect, wait for `session.created`, send
exactly one `session.configure` (persona, voice, tools — frozen for the rest
of the call; only `tools` can change later via `session.update`), then wait
for `session.configured` before sending any audio — frames sent earlier are
silently dropped. From there, tool calls reuse OpenAI-Realtime-shaped event
names: `response.function_call_arguments.done` arrives from the vendor,
`conversation.item.create` with a `function_call_output` item answers it, and
a `response.create` after the completed tool round asks Hydra to narrate the
result — the same sequence the tool-calling docs describe.

There is no transcript on the wire in either direction — audio is the only
payload — so unlike Gemini Live this transport has no way to send a text
turn requesting the greeting. Instead, `generate_initial_response` is left at
its default (off) and the opening line is forced the way the Grok Voice
transport forces it: a `response.create` with a per-response `instructions`
override, sent once right after `session.configured`. Whether Hydra honours
that override has not been confirmed against a live session — if it doesn't,
the caller hears Hydra's own persona-driven opening instead of the pinned
greeting, not a broken call.

## Audio

The actor speaks and listens at 24 kHz PCM16. Hydra's input is strict PCM16
mono at 16 kHz regardless of model version, so the inbound leg is always
resampled with `audioop.ratecv`. Output rate is fixed per model version and
announced in `session.configured.session.output_audio_sample_rate`:
`hydra-v1.1` (the default here) outputs 24 kHz, matching the actor exactly, so
the outbound leg is untouched; `HYDRA_MODEL=hydra-v1.0` outputs 48 kHz and
would resample on the way out too. The announced rate is checked against the
documented table at connect time and the boot fails loudly on a mismatch
rather than shipping distorted audio.

Hydra's own endpointing and barge-in are not configurable over the documented
API (no `turn_detection`/silence-threshold field), so the ~0.8 s silence pin
the other candidates share for turn-taking parity has no equivalent knob here.

## Barge-in

A response Hydra cancels because the caller spoke over it arrives as
`response.done` with `status: "cancelled"` and
`status_details.reason: "interrupted"`. This bridge does not resume it — the
caller is already mid-turn, and their utterance drives the next response
through Hydra's own server-side turn detection, the same assumption every
other speech-to-speech candidate in this repo makes.

## Running it

```
docker build --platform linux/amd64 -f hydra/Dockerfile -t rory-hydra .
```

The image serves `/voice` and `/health` on `:8008`.

## Environment

| Variable | Default | Role |
|---|---|---|
| `SMALLEST_API_KEY` | — | Required. Rides as a query parameter on the Hydra websocket URL. |
| `HYDRA_MODEL` | `hydra-v1.1` | The model (and output sample rate) in the websocket URL. |
| `HYDRA_VOICE` | `wren` | Voice name — one of `wren`, `sloane`, `marlowe`, `reed`, `knox`, `tate`. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q hydra/tests
```

`test_parity_hydra.py` runs the shared parity assertions, pins the session
shape and the output-rate check above, and checks the container fails closed
at boot without `SMALLEST_API_KEY`. `test_cancelled_tool_call_hydra.py`
scripts the receive loop to prove a barge-in-truncated tool call is skipped
rather than dispatched, and that bridging continues afterward.
