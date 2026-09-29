# rory-deepgram

Rory on the **Deepgram Voice Agent API** — one WebSocket to `agent.deepgram.com` that carries Deepgram STT (`nova-3`), a managed OpenAI LLM (`gpt-4.1-mini`), and Deepgram TTS (`aura-2-thalia-en`) as a single session. Transport only — the tools, vendor clients and prompt live in rory-core.

## What it does

`uvicorn rory_deepgram.web:app` on `:8008` exposes `WS /voice` (plus `/health`). Each `/voice` connection opens a dedicated Voice Agent WebSocket, sends one `Settings` message (the shared prompt plus today's date, the 16 functions from `rory_tools.SCHEMAS`, the STT/LLM/TTS providers, PCM16 at 24 kHz both ways, and the shared greeting), then runs two pumps: caller PCM16 → Deepgram, and Deepgram's message stream → caller audio + function dispatch through `rory_tools.dispatch`.

- **Audio is raw binary in both directions.** With `audio.input` and `audio.output` both `linear16` at 24 kHz — the actor's own format — and `container: "none"`, audio is a byte passthrough with no resampling. The socket is mixed: binary frames are audio, text frames are the JSON event stream.
- **A function is client-side exactly when it declares no `endpoint`.** None of the 16 declares one, so Deepgram sends a `FunctionCallRequest` and waits for this process to answer with a `FunctionCallResponse`. Calls run sequentially on a worker thread, because they share verification and payment state and the vendor clients block.
- **Turn-taking is Deepgram's built-in.** The `v1` listen provider exposes no endpointing control through `Settings`, so unlike the Pipecat and Gemini Live candidates — which pin ~0.8 s of end-of-turn silence — this agent runs whatever the Voice Agent decides.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `DEEPGRAM_API_KEY` | — | required; the only vendor credential (the LLM runs on Deepgram's managed OpenAI access) |
| `STRIPE_API_KEY` | — | required by rory-core |
| `DEEPGRAM_LISTEN_MODEL` | `nova-3` | STT model |
| `DEEPGRAM_VOICE` | `aura-2-thalia-en` | TTS voice |
| `LLM_MODEL` | `gpt-4.1-mini` | think model |
