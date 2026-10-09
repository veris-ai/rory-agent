# rory-grok-cascade

Rory on xAI's streaming speech-to-text and text-to-speech, cascaded around a
chat model. Transport only — the tools, vendor clients and prompt live in
rory-core.

The sibling [`grok-voice`](../grok-voice/) runs xAI's speech-to-speech model;
this one keeps the three legs apart so xAI's STT and TTS can be scored against
the cascades that run the same LLM (Pipecat, LiveKit, Gradium, Deepgram,
ElevenLabs and Vapi run `gpt-4.1-mini`):

```
actor 24 kHz ─► Grok STT (Smart Turn) ─► chat LLM + Rory's tools ─► tokens ─► Grok TTS 24 kHz ─► actor
        └─► Silero VAD ─► barge-in: stop audio + text.clear
```

| Env | Default |
|---|---|
| `XAI_API_KEY` | required — STT and TTS, and the LLM when it is Grok |
| `STRIPE_API_KEY` | required — sandbox fixture; the billing twin's credential |
| `GROK_CASCADE_LLM` | `gpt-4.1-mini` (needs `OPENAI_API_KEY`); `grok-4.3` runs Grok with `reasoning_effort: none` on xAI's OpenAI-compatible API |
| `GROK_STT_MODEL` | `grok-voice-transcribe-2.0` |
| `GROK_VOICE` | `carina` |

A missing `STRIPE_API_KEY`, `XAI_API_KEY` or key for the selected LLM, or an unknown
`GROK_CASCADE_LLM`, fails the boot.

## Pipeline

STT `speech_final` → streamed LLM completion → TTS websocket. Both xAI legs
are told the actor's format, 24 kHz PCM16, so no audio is converted here.

Each LLM round streams with `stream=True`, and every content token goes to the
TTS socket as a `text.delta` the moment it arrives — xAI does the chunking. The
audio is played back while the model is still writing, so speech starts on the
reply's first words. A round that says "let me check that" before calling a
tool is heard before the tool runs; a round with only tool calls opens no
utterance.

## Turn-taking

**End of turn is xAI's.** The STT socket runs with `smart_turn=0.7`, xAI's
setting for callers reading out numbers: at each pause xAI scores whether the
caller has finished, and only a confident score ends the turn with
`speech_final`. A pause it is not confident about — halfway through reading an
account number — leaves the utterance open. `smart_turn_timeout=3000` closes a
turn after 3 s of silence regardless.

Probed live on 2026-10-07 with a synthesized caller reading "four four seven
⟨1.2 s pause⟩ one two nine": Smart Turn at 0.5 and at 0.7 kept the number in
one turn (the pause scored 0.0); without Smart Turn, xAI's 400 ms endpointing
split it into two turns. Real turn ends scored 0.75–0.98.

`endpointing` stays on xAI's default (400 ms). Only `speech_final` text
reaches the LLM; it restates the whole utterance, so chunk finals are logged
and never accumulated.

**Barge-in is local.** Silero VAD (`rory_tools.vad`) runs on the caller's
audio, as in the Mistral and Hugging Face cascades, because it has to cut the
reply within a frame or two. A barge-in stops the audio to the actor at once.
The LLM round it lands in still streams to the end without sending the rest of
its text, and any tool call it emitted runs and is recorded; then the turn ends.
Once that stream is done, `text.clear` goes to the TTS socket and audio is
dropped until `audio.clear`, so the next reply starts clean on the same socket.

**Known issue.** Barge-in stays armed for the whole turn, not only while Rory is
speaking. A voice onset while a round's tools are running (background noise
included) therefore ends the turn once the tools return, and the answer round
after them never runs: the caller hears nothing until they speak again. The
fix is to arm barge-in only around the spoken part of each round; it is left
out here so this transport matches the version that was benchmarked.

**The greeting plays in full.** The bench's background noise starts with the
call, and a television under the caller triggered a barge-in 0.4 s into the
greeting, before any audio went out: the caller heard silence and hung up
after 30 s. Barge-in is live from the first reply on.

**The interrupted reply is trimmed.** A barge-in records the reply in the
model's history as the words whose audio went out, marked `… [interrupted by
the caller]` — estimated from audio sent at 305 ms per word, measured on
`carina`. xAI's `with_timestamps` would align characters exactly, but it
arrives a sentence late (each sentence's characters come after its audio), so
at the cut the sentence the caller interrupted has none yet. Measured
2026-10-07: it costs no time to first audio, so it is worth revisiting if xAI
sends alignment with the audio.

## Other settings

- STT: `language=en` and `format=true`, so spoken numbers come back as digits.
  Audio goes up in the actor's 20 ms frames at real-time pace.
- TTS: one socket per call, one utterance per LLM round. xAI streams audio
  ~5× faster than real time, so it goes to the actor in 0.5 s slices, a slice
  going out once playback is within 1 s of its start (at most 1.5 s ahead), as in
  the Hugging Face cascade — sent as it lands, the
  whole reply is in the actor's buffer before the caller can interrupt it.
  `optimize_streaming_latency` and `text_normalization` stay on xAI's defaults.
- xAI allows 50 concurrent TTS sockets per team. Each call holds one TTS socket
  and one STT socket.

## Files

- `rory_grok_cascade/web.py` — FastAPI app: `GET /health`, `WS /voice` (raw PCM16, 24 kHz mono).
- `rory_grok_cascade/agent.py` — per call: the STT and TTS sockets, the audio pump, the turn worker.
- `rory_grok_cascade/tools.py` — `ToolSchema` → OpenAI function-calling shape, plus `tool_surface()` for the parity check.
