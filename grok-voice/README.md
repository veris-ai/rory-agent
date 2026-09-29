# rory-grok-voice

Rory on xAI's Grok speech-to-speech API. Transport only — the tools, vendor
clients and prompt live in rory-core.

One `grok-voice-think-fast-2.0` realtime websocket per call: the model listens,
reasons, calls Rory's tools and speaks inside that one session. The pod bridges
the Veris actor's PCM16 to it, runs each tool call locally through the shared
dispatcher, and posts the result back on the same socket. Nothing dials in.

## The wire, relative to OpenAI Realtime

Grok speaks the OpenAI Realtime protocol — the same event names in both
directions, so this transport is the Realtime one with two deltas in
`session.update`:

- `voice`, `instructions` and `turn_detection` sit at the session's top level
  rather than under `audio`.
- Caller transcripts only arrive when `audio.input.transcription.model` is
  `grok-transcribe`. Without it the caller's side of the call never reaches
  the log or the trace.

Endpointing is pinned to 800 ms of end-of-turn silence like every other Rory
transport; `threshold` and `prefix_padding_ms` stay on xAI's defaults rather
than borrowing OpenAI's numbers. Both legs run at the actor's 24 kHz, so no
audio is resampled.

## Barge-in

Grok ignores `conversation.item.truncate` on an item whose response is still
generating, which can make it look unimplemented. On a settled item truncate
works, and on a barge-in Grok settles the item itself before it reports the
speech, so the bridge has nothing to send. Probed against
`grok-voice-think-fast-2.0` with the bridge's session shape (server VAD,
`silence_duration_ms: 800`, PCM16 24 kHz both ways).

**Truncate on a settled item works.** Count to twenty, `response.cancel`
after 2630 ms of audio, `response.done` with `status: "cancelled"`, then
`conversation.item.truncate {item_id, content_index: 0, audio_end_ms: 2630}`
→ `conversation.item.truncated {item_id, transcript: " One. Two. Three.",
content_index: 0, audio_end_ms: 2630}`. Asked to repeat exactly what it had
said: "One. Two. Three." The model's context is cut at the truncate point.

**Under server VAD Grok stops the response itself, before `speech_started`.**
Five runs: caller speech (2 s of real speech PCM) streamed into
`input_audio_buffer.append` at real time while a count to one hundred was in
flight. The order was identical every time:

1. `response.output_audio.delta` … (Grok streams ~2.5× faster than real time)
2. `response.done` for the count — `status: "completed"`, `output: []`, no
   `output_audio.done` / `output_item.done` / `transcript.done` for the item,
   `usage.output_audio_seconds` equal to the audio delivered so far
3. `input_audio_buffer.speech_started` — same read as the `response.done`,
   ~1.8–2.2 s after the caller's speech began
4. `conversation.item.added` (user item), `…input_audio_transcription.updated`
5. `response.created` for the barge-in turn, ~2 s later
6. `input_audio_buffer.speech_stopped`, `input_audio_buffer.committed`

Grok never emitted `status: "cancelled"` on its own, and the model's context
ended where generation stopped: asked afterwards to repeat the numbers it had
said, it gave the first seven for ~6.8 s of delivered audio.

**Truncate at the forwarded length is a silent no-op.** The Realtime
transport sends truncate on `speech_started` with
`audio_end_ms` = bytes forwarded to the actor. This bridge forwards every
delta, so forwarded = delivered = where Grok stopped generating; that truncate,
sent at `speech_started`, drew no `conversation.item.truncated` and no
`error`, and the follow-up recalled the same seven numbers as with no
truncate. Under Grok's ordering the Realtime block would also never fire:
`response.done` clears the live item one event before `speech_started`.

**Truncate below the delivered length at `speech_started` works** —
`audio_end_ms: 3000` → `conversation.item.truncated {transcript: " One...
two... three... four"}`, follow-up "One... two... three... four". It is only
useful for a client that stops playout before the delivered audio, which the
actor does not; the Realtime transport rests on the same assumption.

**Truncate targets the latest assistant item, whatever `item_id` says.** The
same 3000 ms truncate sent after the barge-in reply had completed returned
`conversation.item.truncated` echoing the count item's id but with the
reply's transcript cut (`" Hello, sorry to interrupt, I have a question
about"`), and the follow-up repeated the reply. A truncate, if ever needed,
must go out on `speech_started`, before the barge-in turn's item exists.

xAI's docs list neither `response.cancel` nor `conversation.item.truncate`
in their supported or unsupported event tables; the behaviour above is
observed, not documented.

## Running it

```
docker build --platform linux/amd64 -f grok-voice/Dockerfile -t rory-grok-voice .
```

The image serves `/voice` and `/health` on `:8008`. Grok's frames follow the
OpenAI Realtime protocol, so a trace reader built for Realtime reads this
transport's tool calls too.

## Environment

| Variable | Default | Role |
|---|---|---|
| `XAI_API_KEY` | — | Required. |
| `GROK_VOICE_MODEL` | `grok-voice-think-fast-2.0` | The speech-to-speech model in the websocket URL. |
| `GROK_VOICE` | `eve` | Voice name. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q grok-voice/tests
```

`test_parity_grok_voice.py` runs the shared parity assertions, pins the two
session-shape deltas above, and checks the container fails closed at boot
without `XAI_API_KEY`.
