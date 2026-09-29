# rory-huggingface

Rory on Hugging Face's hosted inference: the
[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
cascade — VAD → STT → LLM → TTS — with every model leg hosted on Hugging Face
and nothing running locally. Transport only — the tools, vendor clients and
prompt live in rory-core.

Two hosting configurations share the code, flipped per leg by env:

| Leg | Serverless default (no URL set) | Dedicated Inference Endpoint |
|---|---|---|
| STT | `HF_STT_MODEL` on hf-inference (default `openai/whisper-large-v3`) | `HF_STT_URL`, an ASR endpoint |
| LLM | `HF_LLM_MODEL` on the router (default `openai/gpt-oss-120b:groq`), through the `openai` SDK so OpenTelemetry instrumentation sees the model leg | `HF_LLM_URL`, a vLLM endpoint; `HF_LLM_MODEL` is then the bare served id |
| TTS | `HF_TTS_MODEL` via fal-ai (default `hexgrad/Kokoro-82M`, voice `HF_TTS_VOICE`) | `HF_TTS_URL`, a custom handler returning `{"audio_b64": WAV}` |

One `HF_TOKEN` bills all three legs. `STRIPE_API_KEY` and the World's base
URLs come from the bench.

- `rory_huggingface/web.py` — FastAPI app: `GET /health`, `WS /voice` (raw PCM16, 24 kHz mono). A missing credential fails the boot.
- Startup synthesizes the greeting once before `/health` answers: fal spins the Kokoro worker down when idle and the first synthesis after that takes 20–60 s, which the readiness window absorbs and a live call cannot (the actor hangs up after 30 s without audio).
- `rory_huggingface/agent.py` — per call: a Silero-VAD-gated audio pump that buffers each utterance and endpoints on 0.8 s of silence; a turn worker that transcribes it, drives the chat completion with Rory's tools, and paces the synthesized reply out so barge-in still works. A barge-in also trims the reply in the model's history to the words whose audio went out, marked `… [interrupted by the caller]`.
- `rory_huggingface/tools.py` — `ToolSchema` → OpenAI function-calling shape, plus `tool_surface()` for the parity check.

Utterances go up to Whisper at 16 kHz, base64 in a JSON body with `generate_kwargs` pinning `language=en` (left to detect the language, Whisper read accented callers into Arabic script and the name never verified); a request that times out drops that one turn rather than the call. The reply WAV is decoded and resampled to the actor's 24 kHz if the TTS leg returns anything else.
