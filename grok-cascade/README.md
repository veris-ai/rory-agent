# rory-grok-cascade

Rory on xAI's streaming speech-to-text and text-to-speech, cascaded around a
chat model. Transport only — the tools, vendor clients and prompt live in
rory-core.

The sibling [`grok-voice`](../grok-voice/) runs xAI's speech-to-speech model;
this one keeps the three legs apart so xAI's STT and TTS can be scored against
the other cascades on the same LLM:

```
actor 24 kHz ─► resample 16 kHz ─► Grok STT (Smart Turn) ─► chat LLM + Rory's tools ─► Grok TTS 24 kHz ─► actor
                     └─► Silero VAD ─► barge-in: cancel + text.clear
```

| Env | Default |
|---|---|
| `XAI_API_KEY` | required — STT and TTS, and the LLM when it is Grok |
| `GROK_CASCADE_LLM` | `gpt-4.1-mini` (the other cascades' model, needs `OPENAI_API_KEY`); `grok-4.3` runs Grok with `reasoning_effort: none` on xAI's OpenAI-compatible API |
| `GROK_STT_MODEL` | `grok-voice-transcribe-2.0` |
| `GROK_VOICE` | `eve`, the voice the `grok-voice` candidate speaks with |

A missing key for the selected LLM, or an unknown `GROK_CASCADE_LLM`, fails the boot.

## Turn-taking

**End of turn is xAI's.** The STT socket runs with `smart_turn=0.5` (xAI's
balanced setting): at each pause xAI scores whether the caller has finished,
and only a confident score ends the turn with `speech_final`. A pause it is
not confident about — halfway through reading an account number — leaves the
utterance open. `smart_turn_timeout=3000` closes a turn after 3 s of silence
regardless, so background speech cannot hold one open forever.

Probed live on 2026-10-07 with a synthesized caller reading "four four seven
⟨1.2 s pause⟩ one two nine": Smart Turn at 0.5 and at 0.7 kept the number in
one turn (the pause scored 0.0); without Smart Turn, xAI's 400 ms endpointing
split it into two turns. Real turn ends scored 0.75–0.95 and ended the turn
0.6–0.7 s after the speech. 0.7 would hold more, but a sentence end scoring
0.75 sits too close to it.

`endpointing` stays on xAI's default (400 ms). Only `speech_final` text
reaches the LLM; it restates the whole utterance, so chunk finals are logged
and never accumulated.

**Barge-in is local.** Silero VAD (`rory_tools.vad`) runs on the caller's
audio, as in the Mistral and Hugging Face cascades, because it has to cut the
reply within a frame or two. A barge-in cancels the reply, sends `text.clear`
on the TTS socket and drops audio until `audio.clear`, so the next reply
starts clean on the same socket.

**The interrupted reply is trimmed.** A barge-in rewrites the reply in the
model's history to the words whose audio went out, marked `… [interrupted by
the caller]` — estimated from audio sent at 330 ms per word, measured on
`eve`. xAI's `with_timestamps` would align characters exactly, but it arrives
a sentence late (each sentence's characters come after its audio), so at the
cut the sentence the caller interrupted has none yet. Measured 2026-10-07: it
costs no time to first audio, so it is worth revisiting if xAI sends
alignment with the audio.

## Other settings

- STT: `language=en` and `format=true`, so spoken numbers come back as digits.
  Audio goes up in the actor's 20 ms frames at real-time pace.
- TTS: one socket per call, the whole reply sent as one utterance (`text.delta`
  then `text.done`), PCM16 at 24 kHz. xAI streams ~5× faster than real time,
  so the reply goes to the actor in 0.5 s slices, at most 1 s ahead of
  playback, as in the Hugging Face cascade — sent as it lands, the whole reply
  is in the actor's buffer before the caller can interrupt it.
  `optimize_streaming_latency` and `text_normalization` stay on xAI's defaults.
- xAI allows 50 concurrent TTS sockets per team. Each call holds one TTS socket
  and one STT socket.

## Files

- `rory_grok_cascade/web.py` — FastAPI app: `GET /health`, `WS /voice` (raw PCM16, 24 kHz mono).
- `rory_grok_cascade/agent.py` — per call: the STT and TTS sockets, the audio pump, the turn worker.
- `rory_grok_cascade/tools.py` — `ToolSchema` → OpenAI function-calling shape, plus `tool_surface()` for the parity check.
