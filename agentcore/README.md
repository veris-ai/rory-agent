# rory-agentcore

Rory on the Amazon Connect pattern: Amazon Lex transcribes the caller, a Lambda
initializes an agent in Amazon Bedrock AgentCore Runtime, the agent reasons on a
Bedrock model and calls Rory's tools over MCP through AgentCore Gateway, and the
reply streams back to be spoken by Polly, the engine behind Connect's voices.

The bench does not dial a phone, so the pod stands where Connect would: it takes
the Veris `voice_ws` socket, streams the caller's audio to Lex, takes each
transcript to the agent, and speaks the reply as it is generated.

```
caller ──voice_ws──▶ pod ──StartConversation (one stream per call)──▶ Lex V2
                     │ ◀────────────── transcript per turn ─────────────┘
                     │
                     ├─ on connect ──▶ init Lambda ──▶ DynamoDB session record
                     │                      └──▶ AgentCore Runtime (warm session, prime cache)
                     │
                     └─ per turn ──InvokeAgentRuntime (SSE)──▶ AgentCore Runtime
                          ◀── words as generated ──┘   (Bedrock ConverseStream, Sonnet 5.5)
                                                        │ MCP tools/call
                                                        ▼
                                AgentCore Gateway ──▶ tools Lambda ──▶ Stripe / Fineract twins
                                                         └── DynamoDB session record
```

## One call

1. **Connect.** The pod opens one Lex conversation stream for the call and
   invokes the init Lambda. The Lambda writes the DynamoDB session record (this
   call's twins, a clean verification state) and invokes the runtime once with
   `{"init": true}`, which boots the session's microVM, seeds the history and
   primes Bedrock's prompt cache. Meanwhile Rory speaks the shared greeting
   through Polly.
2. **Caller speaks.** Every audio frame goes to Lex at 8 kHz, the telephone
   rate. Lex decides when the turn is over (`LEX_END_TIMEOUT_MS` of silence) and
   sends its transcript on the stream. The bot only transcribes: no intent of
   its own matches, and its fallback intent has no fulfillment.
3. **Agent.** The pod sends the transcript to the runtime and reads its reply as
   server-sent events. The runtime holds the call's history and runs Bedrock
   ConverseStream with the shared prompt, the frozen-clock date and the shared
   tool declaration, up to five rounds. Each tool call goes out through the
   gateway, one at a time, with the session id added after the model chose the
   call. One voice note joins the prompt: say what you are doing before a tool
   call, so the caller hears something while it runs.
4. **Tools Lambda.** Rebuilds the `CallSession` from the session record, points
   the vendor clients at this call's twins, runs one tool through
   `rory_tools.dispatch`, and writes verification back.
5. **Reply.** The pod cuts the streamed words into sentences (the first piece
   may end at a clause), synthesizes each with Polly as soon as it is complete,
   and plays them in order, paced in 500 ms slices so a barge-in can cut them.
6. **Barge-in.** Silero VAD (shared with the other cascades) watches the
   caller's audio. Voice onset while Rory is speaking cuts the reply. Polly's
   word speech marks give the exact text the caller heard, and the next turn
   tells the runtime to keep only that.

Rory speaks first: the runtime seeds its history with the greeting behind a
`[call connected]` user turn, since Converse histories must open with the user.

## Latency

From the end of the caller's speech to Rory's first audio, measured live from a
laptop against the deployed stack, turns without tools: 1.7 to 2.6 s. Of that,
Lex takes 0.6 to 1.2 s to close the turn, the agent's first words take about
0.75 s, and Polly about 0.3 s. Tool turns open with the spoken preamble, so the
caller hears Rory in the same time while the tool runs.

## What stays shared

The model sees the shared prompt, greeting and 16 tool declarations
(`tools.CONVERSE_TOOLS`), plus the voice note, and every tool runs through the
shared dispatcher in the tools Lambda. The gateway target declares the same
tools plus `session_id`, without the JSON Schema keywords the gateway does not
accept (`minimum` on `make_payment.amount_cents`); the model never sees that
copy. The parity tests check all of this.

## What the bench sees

The bench reads an agent's model calls from OpenInference LLM spans exported in
the pod. The model runs in AgentCore, so the runtime ends each turn's stream with
the turn as the model saw it (messages, tool calls and results, timings, token
counts), and the pod exports one LLM span per model round (`trace.py`).

## Deploying the AWS side

```bash
uv run --package rory-agentcore python agentcore/deploy.py
```

Run from the repo root with AWS credentials. It creates or updates, in
`us-east-1`: the `rory-agentcore-sessions` table, the `rory-agentcore-tools` and
`rory-agentcore-init` Lambdas, the `rory-agentcore` gateway and its `rory-tools`
target, the `rory_agentcore` runtime (built for arm64 from `runtime.Dockerfile`
and pushed to ECR), the `rory-agentcore` Lex bot with a `live` alias, and the
`rory-agentcore-bench` IAM user, allowed only Lex streaming on that alias, Polly,
the init Lambda and the runtime. It prints the values the pod needs.

## The pod image

```
docker build --platform linux/amd64 -f agentcore/Dockerfile -t rory-agentcore .
```

| Variable | Default | Role |
|---|---|---|
| `RORY_AGENTCORE_AWS_ACCESS_KEY_ID` / `RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY` | — | Required. The bench user's key. Not the SDK's default names: bench candidate-env reaches every candidate, and the default names would hand the key to their AWS SDKs. |
| `AWS_REGION` | — | Required. |
| `LEX_BOT_ID` / `LEX_BOT_ALIAS_ID` | — | Required. From `deploy.py`. |
| `INIT_LAMBDA` / `AGENTCORE_RUNTIME_ARN` | — | Required. From `deploy.py`. |
| `STRIPE_API_BASE` / `FINERACT_API_BASE` | — | Required. Injected by the bench; passed on to the tools Lambda. |
| `STRIPE_API_KEY` | — | Required. Sandbox fixture; the billing twin's credential. |
| `RORY_REFERENCE_TIME` | wall clock | The bench's frozen clock, passed to the runtime and the tools. |
| `LEX_END_TIMEOUT_MS` | `400` | Silence that ends a caller's turn. Lower answers sooner and splits more pauses. |
| `LEX_LOCALE_ID` | `en_US` | Lex locale. |
| `POLLY_VOICE_ID` / `POLLY_ENGINE` | `Joanna` / `neural` | Polly voice. |

## Tests

```bash
cd .. && uv run pytest -q agentcore/tests
```

`test_parity_agentcore.py` runs the shared parity assertions against what the
runtime sends Bedrock and what the tools Lambda dispatches. `test_turn_agentcore.py`
covers each piece against fakes of its neighbours. `test_trace_agentcore.py`
checks the exported spans. `test_smoke_agentcore.py` boots the pod and runs a
call through the real tasks, including a barge-in reported on the next turn.
