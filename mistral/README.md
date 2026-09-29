# rory-mistral

Rory on Mistral. Transport only — the tools, vendor clients and prompt live in rory-core.

Mistral ships the three legs of a voice agent as separate services, so this
process is the pipeline: Voxtral Realtime transcribes the caller (16 kHz, so
the actor's 24 kHz audio is downsampled on the way in), a Mistral chat
completion reasons over the transcript and calls Rory's tools, and Voxtral TTS
speaks the reply back at 24 kHz. Turn-taking and barge-in run here on the
caller's audio, not at the vendor. A barge-in cancels the TTS stream and trims the
reply in the model's history to the words whose audio went out, marked
`… [interrupted by the caller]`.

| Env | Default |
|---|---|
| `MISTRAL_API_KEY` | required — boot fails without it |
| `MISTRAL_LLM_MODEL` | `mistral-large-latest` |
| `MISTRAL_STT_MODEL` | `voxtral-mini-transcribe-realtime-2602` |
| `MISTRAL_TTS_MODEL` | `voxtral-mini-tts-2603` |
| `MISTRAL_VOICE` | `en_paul_neutral` (a Voxtral preset slug, resolved at boot) |
