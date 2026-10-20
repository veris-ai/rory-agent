# rory-gemini-live-xt-thinking

Rory, Acme Energy's voice support agent, on **Gemini 3.8 Live Extended
Thinking** (`gemini-3.8-live-extended-thinking`). This is a copy of the
[`gemini-live`](../gemini-live/) transport, which carries the bridge itself
(voice_ws audio, resampling, turn-taking, tool dispatch). Read that README for
how it works.

This model needs two changes that `gemini-live` can't take without changing
its other candidates:

1. **A thinking level.** The API refuses the session without one
   (`1007 Thinking level must be specified for this model`).
   `GEMINI_THINKING_LEVEL` sets it, and the server refuses to boot if it is
   unset. The level is `low`, `medium` or `high`; `minimal` isn't supported.
2. **Non-blocking tools.** The model accepts only `NON_BLOCKING` function
   declarations. While a tool runs, it keeps talking with fillers ("Let me
   check on that") and speaks the result when the response arrives. Every
   other `gemini-live` setting stays as it is: prompt, greeting, voice,
   800 ms end-of-turn silence, both-leg transcription.

## Build

```bash
docker build --platform linux/amd64 -f gemini-live-xt-thinking/Dockerfile -t rory-gemini-live-xt-thinking .
```

## Environment

| Variable | Default | Role |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. The Gemini Live credential. |
| `GEMINI_THINKING_LEVEL` | — | Required. `low`, `medium` or `high`. |
| `GEMINI_LIVE_MODEL` | `gemini-3.8-live-extended-thinking` | The Live model. |
| `GEMINI_VOICE` | `Puck` | Prebuilt voice name. |
| `STRIPE_API_KEY` | — | Sandbox fixture; the billing twin's credential. |
| `FINERACT_USER` / `FINERACT_PASSWORD` / `FINERACT_TENANT` | `mifos` / `password` / `default` | Fineract's public OSS defaults. |

## Tests

```bash
cd .. && uv run pytest -q gemini-live-xt-thinking/tests
```
