# rory-agentcore

Rory on the Amazon Connect pattern: Amazon Lex transcribes, a Lambda hands the
turn to an agent in Amazon Bedrock AgentCore Runtime, the agent reasons on a
Bedrock model and calls Rory's tools over MCP through AgentCore Gateway, and the
reply is spoken by Polly, the engine behind Connect's voices.

The bench does not dial a phone, so the pod stands where Connect would: it takes
the Veris `voice_ws` socket, decides when the caller's turn is over, and speaks
the reply. Everything after Lex runs in AWS.

```
caller ──voice_ws──▶ pod (VAD, Polly) ──RecognizeUtterance──▶ Lex V2
                                                               │ fallback intent
                                                               ▼
                                              init Lambda (rory-agentcore-init)
                                                               │ InvokeAgentRuntime
                                                               ▼
                                     AgentCore Runtime (Bedrock Converse, Sonnet 5.5)
                                                               │ MCP tools/call
                                                               ▼
                                     AgentCore Gateway ──▶ tools Lambda ──▶ Stripe / Fineract twins
                                                                   │
                                                          DynamoDB session record
```

## One turn

1. **Pod.** Silero VAD (shared with the other cascades) opens a turn, keeps
   300 ms from before voice onset, and closes it after 0.8 s of silence. The
   audio goes to Lex at 16 kHz with session attributes: the twins the bench
   injected into the pod (`STRIPE_API_BASE`, `FINERACT_API_BASE`, the Stripe
   fixture key), the bench's frozen clock, and, after a barge-in, the reply as
   the caller heard it.
2. **Lex.** Transcribes. The bot has no intents of its own (one placeholder, so
   Lex will build), so every utterance falls to `FallbackIntent`, whose
   fulfillment is the init Lambda.
3. **Init Lambda.** On a call's first turn writes the DynamoDB session record
   with the twins and a clean verification state. Every turn invokes the
   AgentCore Runtime on the call's session and returns the reply to Lex.
4. **AgentCore Runtime.** Holds the call's history. Runs Bedrock Converse with
   the shared prompt, the frozen-clock date and the shared tool declaration, up
   to five rounds. Each tool call goes out through the gateway, one at a time,
   with the session id added after the model chose the call.
5. **Tools Lambda.** Rebuilds the `CallSession` from the session record, points
   the vendor clients at this call's twins, runs one tool through
   `rory_tools.dispatch`, and writes verification back.
6. **Pod.** Speaks Lex's reply with Polly, paced in 500 ms slices so a barge-in
   can cut it. Polly's word speech marks give the exact text the caller heard.

Rory speaks first: the pod says the shared greeting through Polly, and the
runtime seeds its history with it behind a `[call connected]` user turn, since
Converse histories must open with the user.

## What stays shared

The model sees exactly the shared prompt, greeting and 16 tool declarations
(`tools.CONVERSE_TOOLS`), and every tool runs through the shared dispatcher in
the tools Lambda. The gateway target declares the same tools plus `session_id`,
without the JSON Schema keywords the gateway does not accept (`minimum` on
`make_payment.amount_cents`); the model never sees that copy. The parity tests
check all of this.

## What the bench sees

The model runs in AgentCore, not in the pod, so the bench's in-process tracing
sees no model calls. This candidate is judged on the transcript and the twins'
state, as an image without a supported SDK is.

## Deploying the AWS side

```bash
uv run --package rory-agentcore python agentcore/deploy.py
```

Run from the repo root with AWS credentials. It creates or updates, in
`us-east-1`: the `rory-agentcore-sessions` table, the `rory-agentcore-tools` and
`rory-agentcore-init` Lambdas, the `rory-agentcore` gateway and its `rory-tools`
target, the `rory_agentcore` runtime (built for arm64 from `runtime.Dockerfile`
and pushed to ECR), the `rory-agentcore` Lex bot with a `live` alias, and the
`rory-agentcore-bench` IAM user, allowed only Lex on that alias and Polly. It
prints the bot and alias ids the pod needs.

## The pod image

```
docker build --platform linux/amd64 -f agentcore/Dockerfile -t rory-agentcore .
```

| Variable | Default | Role |
|---|---|---|
| `RORY_AGENTCORE_AWS_ACCESS_KEY_ID` / `RORY_AGENTCORE_AWS_SECRET_ACCESS_KEY` | — | Required. The bench user's key. Not the SDK's default names: bench candidate-env reaches every candidate, and the default names would hand the key to their AWS SDKs. |
| `AWS_REGION` | — | Required. |
| `LEX_BOT_ID` / `LEX_BOT_ALIAS_ID` | — | Required. From `deploy.py`. |
| `STRIPE_API_BASE` / `FINERACT_API_BASE` | — | Required. Injected by the bench; passed on to the tools Lambda. |
| `STRIPE_API_KEY` | — | Required. Sandbox fixture; the billing twin's credential. |
| `RORY_REFERENCE_TIME` | wall clock | The bench's frozen clock, passed to the runtime and the tools. |
| `LEX_LOCALE_ID` | `en_US` | Lex locale. |
| `POLLY_VOICE_ID` / `POLLY_ENGINE` | `Joanna` / `neural` | Polly voice. |

## Limits

- **Latency.** A turn is Lex, a Lambda, AgentCore and one or more Bedrock rounds
  before Polly starts. Measured live: 10 to 12 s from the end of the caller's
  turn to the reply text, with and without a tool call.

## Tests

```bash
cd .. && uv run pytest -q agentcore/tests
```

`test_parity_agentcore.py` runs the shared parity assertions against what the
runtime sends Bedrock and what the tools Lambda dispatches. `test_turn_agentcore.py`
covers each piece against fakes of its neighbours. `test_smoke_agentcore.py`
boots the pod and runs a call through the pump and worker, including a barge-in
reported on the next turn.
