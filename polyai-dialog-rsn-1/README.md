# rory-polyai-dialog-rsn-1

Rory, a customer-service voice agent for Acme Energy, on PolyAI's
**Dialog-RSN-1** realtime API. Dialog-RSN-1 is an audio-native dialog model:
it hears the caller's raw audio, decides when their turn is over, and writes
either a reply or one tool call in the same generation. The service voices
each reply itself with a Cartesia voice and sends the audio at the pace it
plays. Rory explains bills, takes payments, and sets up payment arrangements
end to end, calling **two real vendor APIs**: Stripe for billing and Apache
Fineract for payment arrangements.

This is a separate candidate from [`polyai/`](../polyai/). That one pushes Rory
onto an Agent Studio project at boot and runs whatever model the project is
set to in Studio. This one opens Dialog-RSN-1 directly, one session per call,
with its own prompt and the shared tools sent in `session.update`.

The transport and its prompt live here. The tools, the vendor clients and the
verification gate are shared with the other implementations and live at the
repo root. See the layout section in the [root README](../README.md).

## What it does

One process runs inside the container: uvicorn serving
`rory_polyai_dialog_rsn_1/web.py`, which exposes `WS /voice` and `/health` on
`:8008`.

Each `/voice` connection opens its own websocket to
`wss://api.studio.poly.ai/v1/realtime`, which speaks the OpenAI Realtime wire
format (GA schema):

1. Wait for `session.created`, send `session.update` with this candidate's
   prompt plus the frozen-clock date context, the 16 shared tools, 24 kHz PCM16 input,
   server turn detection and the voice, then wait for `session.updated`. A
   refused update applies none of its fields, so anything else ends the call.
2. Send a `response.create` whose one-off instructions ask for the shared
   greeting, so Rory speaks first.
3. Forward the caller's PCM16 frames as `input_audio_buffer.append`, and the
   reply's `response.output_audio.delta` bytes straight back. Both legs are
   24 kHz, so nothing is resampled.
4. Collect `response.function_call_arguments.done`. When the response ends
   `completed`, run each call through `rory_tools.dispatch` on a worker
   thread, answer with a `function_call_output`, then send `response.create`,
   since a tool result never starts a turn on its own. A call from a response
   the caller cancelled by barging in is never run.

```mermaid
flowchart LR
    caller["caller<br/>(voice_ws actor)"]

    subgraph container ["one container"]
        direction LR
        web["rory_polyai_dialog_rsn_1<br/>:8008 /voice"]
        tools["rory_tools<br/>dispatch + CallSession"]
    end

    subgraph polyai ["PolyAI"]
        rsn["Dialog-RSN-1<br/>turn-taking, ASR, reply, tool calls"]
        tts["Cartesia voice"]
    end

    stripe["Stripe<br/>api.stripe.com"]
    fineract["Fineract<br/>FINERACT_API_BASE"]

    caller <-->|"PCM16 WS 24 kHz"| web
    web <-->|"realtime WS<br/>PCM16 24 kHz"| rsn
    rsn --> tts
    web --> tools
    tools --> stripe
    tools --> fineract
```

## What PolyAI owns

- **Turn-taking.** A voice-activity detector proposes each turn end and the
  model confirms it from the conversation. `create_response` is the only
  `turn_detection` field the service reads; silence duration and thresholds
  are ignored, so the benchmark's ~0.8 s end-of-turn standard is not pinned
  here.
- **Barge-in.** A confirmed interruption cancels the reply
  (`response.done` with status `cancelled`). The server paces its own audio,
  so it knows what the caller heard and trims history itself; the bridge sends
  no `conversation.item.truncate`.
- **Speech.** `output_modalities: ["audio"]`, a Cartesia voice id in
  `audio.output.voice`.

## Its own prompt

This is the one candidate that doesn't load the shared prompt. Dialog-RSN-1
treats any text it writes as the end of its turn (PolyAI's prompt guide: "One
output per turn"), so a line like "tell the caller, then transfer" hands the
floor back before the transfer runs, which cuts tool chains short.
[`agent_desc.txt`](rory_polyai_dialog_rsn_1/agent_desc.txt) is a copy of
[`rory_tools/agent_desc.txt`](../rory_tools/agent_desc.txt) with five changes:
one new rule (make the tool call first, speak once it returns) and four
"tell the caller, then transfer" lines reordered to call `transfer_to_human`
first. `transfer_to_human` already answers "Tell the caller they are being
transferred, then stop.", so the new order fits it. See the changes with:

```bash
diff rory_tools/agent_desc.txt polyai-dialog-rsn-1/rory_polyai_dialog_rsn_1/agent_desc.txt
```

The greeting and the tools are still the shared ones. A change to the shared
prompt has to be copied here by hand.

## Silent tool turns

Dialog-RSN-1 outputs either text or one tool call, never both, and a turn
runs tool, tool, ..., text. So **the caller hears nothing while a chain of
tools runs**, until the final reply. PolyAI's fix is `poly_spoken`: a required
string argument on each slow tool that the model fills with a hold line the
service speaks while the tool runs. It is not used here, because it would
give every tool an argument the other candidates' tools don't have, and the
parity run would fail. Expect longer silences on lookups than candidates that
can speak and call a tool in the same response.

## Configuration

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `POLY_API_KEY` | yes | | A workspace API key from the **API Keys** tab in Agent Studio. A personal access token is refused. |
| `POLYAI_RSN_URL` | no | `wss://api.studio.poly.ai/v1/realtime` | A key works only on the host matching the Studio that issued it. Enterprise keys from `studio.us.poly.ai` need `wss://api.us.poly.ai/v1/realtime`. |
| `POLYAI_RSN_VOICE` | no | `db6b0ed5-…` | A Cartesia voice id. The default is the voice the [`cartesia`](../cartesia/) candidate speaks with. An id Cartesia doesn't know fails every reply with `audio_synthesis_failed`. |
| `STRIPE_API_KEY` | yes | | Shared with every transport; the bench injects a sandbox fixture. |

The bench also injects `STRIPE_API_BASE` and `FINERACT_API_BASE`, which the
shared vendor clients honor.

## Running it

```
docker build --platform linux/amd64 -f polyai-dialog-rsn-1/Dockerfile -t rory-polyai-dialog-rsn-1 .
```

The image serves `/voice` and `/health` on `:8008` and is ready as soon as
uvicorn is up; nothing is provisioned at boot.

## Running the tests

```bash
uv run pytest polyai-dialog-rsn-1/tests
```

The parity run checks the tools sent in `session.update` against the shared
declaration. The lifecycle tests run a whole call against a local websocket
standing in for PolyAI (handshake, greeting, audio both ways, one tool
round), plus the refused-session path, cancelled tool calls, failure events
and the close codes. The model and the voice need a live session and a key.

## Known limits

- **Concurrency.** PolyAI caps open sessions per account: **2** on the
  sign-up credit, **10** once a card is on file. The next handshake past the
  cap is refused with `429`, which surfaces as a failed call, so keep bench
  parallelism at or below the account's cap.
- **One response at a time.** If the caller's turn opens a response while a
  tool round is still running, the bridge's follow-up `response.create` is
  refused with `conversation_already_has_active_response`, and the call ends
  with that error. PolyAI holds a request sent while the caller is still
  speaking, so this needs a turn that has already been committed.
- **English only**, for now.
- **Session length.** 60 minutes per session and a 5-minute idle timeout,
  both far beyond a bench call.
