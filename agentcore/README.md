# rory-agentcore

Rory on Amazon Connect's voice stack: Amazon Lex for speech-to-text, Amazon
Bedrock AgentCore Runtime for the reasoning loop, Amazon Polly for speech.
Transport only. The tools, vendor clients and prompt live in rory-core.

Amazon Connect is the telephony front end: a call comes in on Connect, and each
turn goes Lex → AgentCore → Polly. The bench does not dial a phone, so this
process stands where Connect would. It connects to the Veris `voice_ws` socket
and runs the same three legs as a cascade, owning turn-taking and barge-in
itself, as the other cascades do.

## The turn

1. **Caller speaks.** Silero VAD (shared with the other cascades) opens a turn
   and keeps 300 ms from before voice onset, so Lex does not lose the first word.
2. **Turn ends** after 0.8 s of silence. The buffered audio is resampled to
   16 kHz and sent to Lex as one utterance (`RecognizeUtterance`). Lex returns
   the transcript gzipped and base64-encoded in a header.
3. **Model round.** The transcript and history go to AgentCore Runtime
   (`InvokeAgentRuntime`), one request per model round. Up to five rounds per
   turn. Tool calls come back to this process and run through the shared
   dispatcher, one at a time, and their results go into the next round.
4. **Reply.** The final text goes to Polly as 16 kHz PCM, resampled to the
   actor's 24 kHz, and played in 500 ms slices, paced at most one second ahead
   of playback. Polly's word speech marks say which words the caller heard, so a
   barge-in keeps exactly that part of the reply in the history.

Rory speaks first. The greeting goes straight through Polly, like the Pipecat
cascade, and is not interruptible.

## The AgentCore Runtime contract

The AgentCore Runtime is deployed separately, not from this repository. It
answers one model round per request, so it owns the Bedrock call and nothing
else. Tool execution stays in this process, where verification and payment state
live.

```
request   {"messages": [...], "tools": [{"name", "description", "parameters"}]}
response  {"content": string | null, "tool_calls": [{"id", "name", "arguments"}]}
```

`messages` is the running history in the neutral shape this transport keeps:
`{role, content, tool_calls}` for assistant turns, `{role: "tool", tool_call_id,
name, content}` for results. `tools` is the shared declaration. The runtime
maps it onto its Bedrock Converse tool configuration. It must return both keys
on every response. A missing key is a failed turn, not a silent empty reply.

## The wire

Raw PCM16, 24 kHz mono, in both directions, as the other `voice_ws` candidates.
`GET /health` on `:8008`. `WS /voice` for the call.

## Running it

```
docker build --platform linux/amd64 -f agentcore/Dockerfile -t rory-agentcore .
```

Boot fails before the first call if a required variable is missing. The
container needs AWS credentials for the default provider chain, plus the
variables below.

## Environment

| Variable | Default | Role |
|---|---|---|
| `AWS_REGION` | — | Required. Region for Lex, AgentCore and Polly. |
| `LEX_BOT_ID` / `LEX_BOT_ALIAS_ID` | — | Required. The Lex V2 bot and alias that transcribe the caller. |
| `LEX_LOCALE_ID` | `en_US` | Lex locale. |
| `AGENTCORE_RUNTIME_ARN` | — | Required. The deployed AgentCore Runtime. |
| `POLLY_VOICE_ID` | `Joanna` | Polly voice. |
| `POLLY_ENGINE` | `neural` | Polly engine. |
| `STRIPE_API_KEY` | — | Required. Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Status

The offline suite covers parity with the shared declaration, the tool loop, the
barge-in trim, the Lex transcript decoding, resampling, and a full call through
the real pump and worker with AWS replaced. The image boots and serves `/health`.

**Not yet run against AWS.** The Lex request and transcript header, the
AgentCore response body, and Polly's speech marks are checked against the boto3
service models, not against a live account. The first live run needs the
variables above and a deployed AgentCore Runtime that implements the contract.

## Tests

```bash
cd .. && uv run pytest -q agentcore/tests
```

`test_parity_agentcore.py` runs the shared parity assertions. `test_turn_agentcore.py`
covers the turn logic with fake AWS clients. `test_smoke_agentcore.py` boots the app
and runs a call through the pump and worker.
