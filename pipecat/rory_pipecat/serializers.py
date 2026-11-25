"""Frame serializer for Veris's ``voice_ws`` channel.

Veris's actor speaks raw PCM16/24 kHz mono binary frames over a WebSocket
— no telephony envelope, no JSON wrapping. This serializer is the bytes ↔
frame mapping that lets ``FastAPIWebsocketTransport`` carry that audio into
and out of the Pipecat pipeline.

Incoming audio is resampled from the wire's 24 kHz to the pipeline's input
rate (16 kHz, see ``PipelineParams`` in ``agent.py``), which is the rate
Silero VAD and Smart Turn require. Outgoing audio is already 24 kHz — the TTS
and the output transport run at the wire rate — so it passes through.
"""

from __future__ import annotations

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer

VERIS_SAMPLE_RATE_HZ = 24000
VERIS_NUM_CHANNELS = 1


class RawPCM16Serializer(FrameSerializer):
    """Serializer for raw PCM16/24 kHz/mono WebSocket audio.

    Inbound (``deserialize``): bytes at 24 kHz → ``InputAudioRawFrame`` at the
    pipeline's input rate.
    Outbound (``serialize``): ``OutputAudioRawFrame`` → bytes, unchanged.
    All other frames are dropped (None) so the WS only carries audio.
    """

    def __init__(self) -> None:
        super().__init__(
            params=FrameSerializer.InputParams(audio_in_sample_rate=VERIS_SAMPLE_RATE_HZ)
        )

    async def serialize(self, frame: Frame) -> bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not isinstance(data, bytes):
            return None
        audio = await self._resample_audio_in(data)
        if not audio:
            return None
        return InputAudioRawFrame(
            audio=audio,
            sample_rate=self._sample_rate,
            num_channels=VERIS_NUM_CHANNELS,
        )
