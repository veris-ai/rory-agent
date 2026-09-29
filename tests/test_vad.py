"""Silero VAD over the actor's PCM16 stream.

An energy gate hears cafe and television noise under the caller as a speaking
caller. These pin the property that matters — noise at speech-like energy is
not speech — and the framing arithmetic around it. No speech fixture ships
with the repo, so "speech is detected" needs a live run, not this suite.
"""

from __future__ import annotations

import audioop

import numpy as np
import pytest

from rory_tools import vad
from rory_tools.vad import SileroVad

ACTOR_RATE_HZ = 24000
FRAME_MS = 20
FRAME_SAMPLES = ACTOR_RATE_HZ * FRAME_MS // 1000


def _frames(pcm: np.ndarray) -> list[bytes]:
    raw = pcm.astype("<i2").tobytes()
    step = FRAME_SAMPLES * 2
    return [raw[i : i + step] for i in range(0, len(raw) - step + 1, step)]


def test_silence_is_not_speech():
    det = SileroVad(ACTOR_RATE_HZ)
    verdicts = [det.push(f) for f in _frames(np.zeros(ACTOR_RATE_HZ * 2, dtype=np.int16))]
    assert not any(verdicts)
    assert 0.0 <= det.probability < vad.START_THRESHOLD


def test_noise_at_speech_energy_is_not_speech():
    """White noise well above an RMS-500 energy gate must stay below the speech threshold."""
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(ACTOR_RATE_HZ * 3) * 2000).clip(-32768, 32767).astype(np.int16)
    frames = _frames(noise)
    assert all(audioop.rms(f, 2) > 500 for f in frames)  # an energy gate would fire on every frame
    det = SileroVad(ACTOR_RATE_HZ)
    fired = sum(det.push(f) for f in frames)
    assert fired == 0


def test_verdict_is_carried_between_windows():
    """A 20 ms frame at 24 kHz is 320 samples at 16 kHz; a window is 512. Frames that
    complete no window must keep the previous verdict rather than reset it."""
    det = SileroVad(ACTOR_RATE_HZ)
    det._speaking = True  # pretend the last window said speech
    assert det.push(b"\x00\x00" * FRAME_SAMPLES) is True  # 320 samples pending, no window yet
    # the second frame completes a window of silence: verdict re-evaluated
    assert det.push(b"\x00\x00" * FRAME_SAMPLES) is False


def test_model_session_is_shared_and_state_is_per_call():
    a, b = SileroVad(ACTOR_RATE_HZ), SileroVad(ACTOR_RATE_HZ)
    assert vad._get_session() is vad._get_session()
    assert a._state is not b._state


@pytest.mark.parametrize("rate", [16000, 24000])
def test_any_input_rate_is_resampled_to_the_model_rate(rate):
    det = SileroVad(rate)
    det.push(np.zeros(rate * 40 // 1000, dtype=np.int16).tobytes())  # 40 ms
    # 40 ms at 16 kHz is 640 samples: one 512-sample window scored, 128 pending
    assert len(det._pending) == 128 * 2
