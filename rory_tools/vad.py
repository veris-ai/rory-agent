"""Silero voice activity detection over the actor's PCM16 stream.

Rory's cascaded transports (Mistral, Hugging Face) own their turn-taking: the
vendor legs are STT and TTS, and nothing upstream says when the caller starts
or stops. A frame-energy gate is not enough once the caller has background
noise: cafe babble or a television sits above any energy threshold for seconds
at a time, so the agent would hear a caller who never stopped talking, cut its
own replies as barge-ins, and endpoint turns full of background chatter.

Silero VAD is a speech-vs-not classifier, the same model Pipecat and LiveKit
run on the same audio. It scores 32 ms windows of 16 kHz audio; a frame at
the actor's 24 kHz is resampled and accumulated into those windows, and the
last window's verdict stands for every frame until the next one completes.

The ONNX weights ship with this package (``data/silero_vad.onnx``, MIT, see
``data/SILERO_VAD_LICENSE``) and run on CPU through onnxruntime; one session
is shared per process, the recurrent state is per call.
"""

from __future__ import annotations

import audioop
from pathlib import Path

import numpy as np
import onnxruntime

_MODEL_PATH = Path(__file__).parent / "data" / "silero_vad.onnx"
_MODEL_RATE_HZ = 16000
_WINDOW_SAMPLES = 512  # 32 ms at 16 kHz, the size the model is trained on
_CONTEXT_SAMPLES = 64  # the model wants the previous window's tail in front
_WINDOW_BYTES = _WINDOW_SAMPLES * 2

# Probability at which speech starts, and below which it ends. The gap keeps a
# breath or a soft syllable from flickering the verdict mid-sentence.
START_THRESHOLD = 0.5
STOP_THRESHOLD = 0.35

# Recurrent state is reset after this much continuous non-speech so a long
# silence (or a long stretch of background noise) cannot drift it.
_RESET_AFTER_S = 5.0

_session: onnxruntime.InferenceSession | None = None


def _get_session() -> onnxruntime.InferenceSession:
    global _session
    if _session is None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _session = onnxruntime.InferenceSession(
            str(_MODEL_PATH), providers=["CPUExecutionProvider"], sess_options=opts
        )
    return _session


class SileroVad:
    """Per-call detector: feed PCM16 mono frames, read back whether the caller is speaking."""

    def __init__(self, rate_hz: int) -> None:
        self._rate_hz = rate_hz
        self._resample_state = None
        self._pending = b""
        self._speaking = False
        self._quiet_s = 0.0
        self.probability = 0.0
        self._reset_states()

    def _reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def push(self, frame: bytes) -> bool:
        """Score ``frame`` (PCM16 mono at ``rate_hz``); return whether the caller is speaking now.

        Frames shorter than a model window carry the previous window's verdict
        forward, so callers can keep their per-frame accounting unchanged.
        """
        if self._rate_hz != _MODEL_RATE_HZ:
            frame, self._resample_state = audioop.ratecv(
                frame, 2, 1, self._rate_hz, _MODEL_RATE_HZ, self._resample_state
            )
        self._pending += frame
        while len(self._pending) >= _WINDOW_BYTES:
            window = np.frombuffer(self._pending[:_WINDOW_BYTES], dtype="<i2")
            self._pending = self._pending[_WINDOW_BYTES:]
            self._score(window.astype(np.float32) / 32768.0)
        return self._speaking

    def _score(self, window: np.ndarray) -> None:
        x = np.concatenate((self._context, window[np.newaxis, :]), axis=1)
        out, self._state = _get_session().run(
            None, {"input": x, "state": self._state, "sr": np.array(_MODEL_RATE_HZ, dtype=np.int64)}
        )
        self._context = x[:, -_CONTEXT_SAMPLES:]
        self.probability = float(out[0][0])
        if self._speaking:
            self._speaking = self.probability > STOP_THRESHOLD
        else:
            self._speaking = self.probability >= START_THRESHOLD
        if self._speaking:
            self._quiet_s = 0.0
        else:
            self._quiet_s += _WINDOW_SAMPLES / _MODEL_RATE_HZ
            if self._quiet_s >= _RESET_AFTER_S:
                self._reset_states()
                self._quiet_s = 0.0
