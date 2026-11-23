"""The pipeline's input audio arrives at the rate its VAD is told it is.

Silero accepts only 8 or 16 kHz and takes its rate from ``PipelineParams``, not
from the frames. The actor speaks 24 kHz, so the serializer has to resample to
the pipeline's input rate; when it did not, Silero read 24 kHz audio as 16 kHz
— slowed and pitched down, with every VAD timing 1.5x too short.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import OutputAudioRawFrame

from rory_pipecat import agent
from rory_pipecat.serializers import VERIS_SAMPLE_RATE_HZ, RawPCM16Serializer


def _worker_params(monkeypatch):
    for name in ("_build_stt", "_build_llm", "_build_tts"):
        monkeypatch.setattr(agent, name, lambda: None)
    monkeypatch.setattr(agent, "LLMContextAggregatorPair", lambda *a, **k: (None, None))
    monkeypatch.setattr(agent, "Pipeline", lambda processors: SimpleNamespace())
    monkeypatch.setattr(
        agent, "PipelineWorker", lambda pipeline, *, params, app_resources: params
    )
    return agent._build_pipeline_worker(SimpleNamespace(input=lambda: None, output=lambda: None))


def _deserialize_all(serializer: RawPCM16Serializer, chunks: list[bytes]):
    async def run():
        return [await serializer.deserialize(c) for c in chunks]

    return asyncio.run(run())


def test_pipeline_input_rate_is_one_silero_accepts(monkeypatch):
    params = _worker_params(monkeypatch)
    assert params.audio_in_sample_rate in (8000, 16000)
    assert params.audio_out_sample_rate == VERIS_SAMPLE_RATE_HZ


def test_services_take_their_rates_from_the_pipeline(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    assert agent._build_stt()._init_sample_rate is None
    assert agent._build_tts()._init_sample_rate is None


def test_serializer_resamples_wire_audio_to_the_pipeline_input_rate():
    rate = agent.PIPELINE_IN_SAMPLE_RATE_HZ
    serializer = RawPCM16Serializer()
    asyncio.run(serializer.setup(SimpleNamespace(audio_in_sample_rate=rate)))

    # One second of a 440 Hz tone at the wire rate, in the actor's 20 ms frames.
    t = np.arange(VERIS_SAMPLE_RATE_HZ) / VERIS_SAMPLE_RATE_HZ
    pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).tobytes()
    step = VERIS_SAMPLE_RATE_HZ // 50 * 2
    frames = [f for f in _deserialize_all(serializer, [pcm[i : i + step] for i in range(0, len(pcm), step)]) if f]

    assert frames and all(f.sample_rate == rate for f in frames)
    samples = sum(len(f.audio) for f in frames) // 2
    # A stream resampler holds back a few milliseconds; allow 20 ms of it.
    assert rate - rate // 50 <= samples <= rate


def test_serializer_passes_output_audio_through():
    serializer = RawPCM16Serializer()
    audio = b"\x01\x02" * 480
    frame = OutputAudioRawFrame(audio=audio, sample_rate=VERIS_SAMPLE_RATE_HZ, num_channels=1)
    assert asyncio.run(serializer.serialize(frame)) == audio


def test_serializer_ignores_text_frames():
    assert _deserialize_all(RawPCM16Serializer(), ["{}"]) == [None]
