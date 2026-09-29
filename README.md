# Rory Agent

Rory Agent is a growing collection of voice-agent implementations for the same
utility customer-service workflow. Each implementation gives Rory the same job,
tools, and synthetic Acme Energy data while using a different voice platform or
agent framework.

The repository is intended to make the implementations easy to study, run,
compare, and improve. Pull requests are welcome, including new provider
implementations.

It is the sibling of [Riley Agent](https://github.com/veris-ai/riley-agent),
which runs the same comparison over a card-support workflow at Acme Bank.
Where Riley's calls are about a single card, **Rory's are about money the
caller does not have** — so the workflow reaches arrears, payment
arrangements, shutoff protection, and explaining a bill the caller thinks is
wrong.

> [!IMPORTANT]
> Rory is a simulation and reference project, not a production utility service.
> The accounts, bills, meter readings, and payment arrangements it creates are
> synthetic, and the vendor credentials the bench injects are sandbox
> fixtures that are worthless anywhere else. Review authentication,
> authorization, privacy, and provider-cost controls before adapting any
> implementation for real callers or data.

## Implementations

| Directory | Voice stack |
| --- | --- |
| [`pipecat`](pipecat/) | Pipecat with Deepgram, OpenAI, and ElevenLabs — a cascaded STT → LLM → TTS pipeline |
| [`gemini-live`](gemini-live/) | Gemini Live API — a single speech-to-speech model, no STT or TTS of its own; `GEMINI_LIVE_MODEL` picks 2.5 native audio or 3.1 Flash Live |
| [`openai-realtime`](openai-realtime/) | OpenAI Realtime API — speech-to-speech over a websocket; `REALTIME_MODEL` picks the full or mini model |
| [`deepgram`](deepgram/) | Deepgram Voice Agent API — listen, think and speak in one vendor socket |
| [`elevenlabs`](elevenlabs/) | ElevenLabs Conversational AI — a platform-hosted agent with Rory's tools registered as client tools |
| [`livekit`](livekit/) | LiveKit Agents over a LiveKit server running inside the image — cascaded, with framework-native function calling |
| [`mistral`](mistral/) | Mistral Voxtral realtime STT and TTS around a Mistral chat model — a hand-rolled cascade |
| [`fluxions`](fluxions/) | Fluxions voice-agent API — ASR, a vLLM-served chat model and VUI TTS in one realtime websocket, with Rory's tools as bring-your-own functions |
| [`huggingface`](huggingface/) | Three Hugging Face Inference endpoints — Whisper, a router chat model, Kokoro — as a cascade |
| [`grok-voice`](grok-voice/) | xAI Grok speech-to-speech API (`grok-voice-think-fast-2.0`) — one realtime websocket per call, OpenAI-Realtime-compatible wire |
| [`vapi`](vapi/) | Vapi — a hosted orchestrator (Deepgram, gpt-4.1-mini, ElevenLabs) behind a per-call inline assistant, with Rory's tools as server tools reached through a quick tunnel |
| [`cartesia`](cartesia/) | Cartesia Managed Agents — Ink STT, a hosted LLM and Sonic TTS behind a per-call managed agent, audio over the agent WebSocket, Rory's tools as webhook tools created once per pod on the attempt's public endpoint |
| [`gradium`](gradium/) | gradbot — Gradium's open-source Rust voice engine running Gradium streaming STT and TTS around an OpenAI-compatible LLM (`gpt-4.1-mini`) in-process, turn-taking and barge-in owned by the engine, Rory's tools handed back to the pod through the shared dispatcher |
| [`polyai`](polyai/) | PolyAI Agent Studio — a hosted platform whose agent is project configuration pushed by the ADK at boot, reached over its WebRTC gateway with aiortc, with Rory's tools as Agent Studio functions that POST back through a quick tunnel |
| [`gpt-live`](gpt-live/) | OpenAI GPT-Live (`gpt-live-1`) — a full-duplex live model over one websocket per call that delegates tool use to a Responses backend model (`LIVE_BACKEND_MODEL`), Rory's tools answered by the pod as function-call outputs |

Each ships as its own image, built from one shared lockfile with
`uv sync --package`, so a candidate carries only its own voice SDK. Most have
a `<dir>/Dockerfile`; Pipecat and Gemini Live share `Dockerfile.bench`, where
`RORY_AGENT_APP` picks the transport that boots.

## Repository layout

Each implementation is **only a transport**. Everything an implementation could
otherwise get subtly wrong is shared, and lives above them:

```
rory_tools/    the 16 tools, their implementations, the vendor clients,
               the verification gate, the shared system prompt, and the
               parity checks every transport runs against itself
pipecat/       Pipecat transport
gemini-live/   Gemini Live transport
openai-realtime/, deepgram/, elevenlabs/, livekit/, mistral/, fluxions/, huggingface/,
grok-voice/, vapi/, cartesia/, gradium/, polyai/, gpt-live/   thirteen more transports, same contract
```

A transport is expected to use three things — `SCHEMAS`, `CallSession`, and
`dispatch` — and to adapt the first into its own framework's schema type. It
does not get to reword a tool description, re-implement the verification gate,
or carry its own prompt.

Parity is enforced per transport: each `<dir>/tests/test_parity_*.py` reads the
tool declarations back out of the objects it actually hands its framework and checks
them against the shared declaration in `rory_tools` — names, descriptions, complete
argument schemas, the shared prompt and greeting, and dispatcher identity
(`rory_tools/parity.py`). Transports therefore never need to import each other, which
is what lets each ship its own image. `tests/test_candidate_parity.py` keeps the
checks that have no transport in them at all.

Every transport executes tool batches sequentially because calls share account
and payment state. Their greeting mechanisms differ: Pipecat seeds an assistant
turn and speaks the fixed greeting through TTS; Gemini receives a user trigger
requesting that greeting and generates native audio, which can paraphrase it.
Lifecycle tests check the actual seeding and close behavior; they do not claim
that native audio is verbatim.

Model and tool messages are collected through OpenTelemetry instrumentation
injected by the bench; the agents carry no reporting code of their own.

## The workflow

Rory takes inbound calls on the billing line of Acme Energy, a regulated gas
and electric utility, covering:

- **Billing inquiries** — the account balance, retrieving a bill, explaining
  every charge on it, payment due dates, usage analysis, paperless enrolment,
  and who supplies the energy.
- **Payment assistance** — whether a payment arrangement is possible, what it
  costs, enrolling in one, extending a due date, reviewing payment history, and
  modifying an arrangement already running.

Both are shaped around the same constraint: the caller has to be verified
before anything is disclosed, and several of the right answers are refusals.

Payments currently settle a full remaining bill and require an explicit
caller-confirmed amount. Partial payments and arrangements requiring a down
payment go to a human: these tools cannot collect the down payment and will
not open the plan without it.

Due-date extensions currently fail against finalized invoices. The existing
implementation updates Stripe's `due_date`, which the
[Stripe API permits only on draft invoices](https://docs.stripe.com/api/invoices/update?lang=curl).
Supporting extensions requires a different billing mechanism; the twin's
refusal must not be bypassed.

## Why this workflow

A utility billing line is the highest-volume call type in the sector — a
mid-size utility takes millions of these a year — and almost every call is one
of two conversations: *what is this bill* and *I can't pay it*. Both are
unusually hard for a voice agent in ways that are easy to measure:

**Explaining a bill is an attribution problem, not a lookup.** A bill can only
move for three reasons — the customer used more energy, the rate changed, or
the billing period ran longer — and an agent that reasons from two totals will
routinely name the wrong one. Rory's world contains a customer whose bill went
**up** while her usage went **down**; getting her call right requires actually
decomposing the change rather than inferring it.

**Payment assistance is a policy problem with real stakes.** Some callers
qualify for an arrangement, some need money up front because they broke the
last one, some already have a plan and cannot open a second, and some are far
enough into arrears that the law requires a human. Getting these wrong is not
a bad customer experience — it is a regulatory exposure.

**Two failures matter more than the rest**, and both look like success in a
transcript: telling a caller a declined payment went through, and recording a
promise to pay as a payment. The environment makes both reachable, and both
detectable.

## What it runs against

Rory uses Stripe and Apache Fineract APIs. Bench connects the candidate to
[Veris](https://veris.ai) twins through the injected `STRIPE_API_BASE` and
`FINERACT_API_BASE` URLs:

| Vendor | Role |
|---|---|
| **Stripe** | The billing system — accounts, service agreements, bills, metered usage, payments, declines, credits |
| **Apache Fineract** | The payment-arrangement system — enrolment, instalment schedules, payments, default |

Neither vendor has heard of a utility, and that is the point: the agent sees a
utility CIS, assembled by the tool layer, while the behaviour underneath is a
real vendor's — including the refusals. Fineract will not accept a repayment
dated in the future, will not disburse a plan nobody approved, and matches an
account number exactly and case-sensitively. Stripe declines a card that has no
funds. None of that is mocked.

The split across two systems is deliberate and realistic: utilities do run
billing and collections separately, and keeping them consistent is precisely
the cross-system reasoning this agent exists to test.

## Run locally

```bash
uv sync                  # every transport plus the test dependencies
uv run pytest -q         # the offline suite; the live-twin integration tests skip
```

To start a transport, export its provider keys (listed in its README) and
`STRIPE_API_KEY`, then run its app on `:8008`:

```bash
uv run uvicorn rory_pipecat.web:app --host 0.0.0.0 --port 8008
```

It serves `GET /health` and `WS /voice`, the `voice_ws` protocol: raw PCM16,
24 kHz mono, in both directions. The tools need Stripe and Fineract backends
holding the Acme Energy world; point the clients at them with
`STRIPE_API_BASE` and `FINERACT_API_BASE`. LiveKit runs three processes
through `livekit/start.sh` and needs `livekit-server`; Vapi and PolyAI need
`cloudflared` or a `PUBLIC_BASE_URL`; Cartesia needs a `PUBLIC_BASE_URL`.

## Licence

MIT — see [LICENSE](LICENSE). Bundled third-party material keeps its own
licence; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
