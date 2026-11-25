"""Rory takes turns on Pipecat's detectors, gated on the LLM's verdict.

A user turn starts on voice activity or a transcript and Smart Turn v3 judges
when the caller has finished; ``FilterIncompleteUserTurnStrategies`` then has
the LLM mark the turn complete or incomplete, and only a complete one is
answered. An earlier override replaced the detectors: a fixed 0.6 s silence
timer to end the turn, and a start on final transcripts only, which delayed
barge-in by Deepgram's finalization.
"""

from __future__ import annotations

from types import SimpleNamespace

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    LLMTurnCompletionUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies

from rory_pipecat import agent


def _user_params(monkeypatch):
    captured = {}

    def pair(context, *, user_params):
        captured["user_params"] = user_params
        return None, None

    for name in ("_build_stt", "_build_llm", "_build_tts"):
        monkeypatch.setattr(agent, name, lambda: None)
    monkeypatch.setattr(agent, "LLMContextAggregatorPair", pair)
    monkeypatch.setattr(agent, "Pipeline", lambda processors: SimpleNamespace())
    monkeypatch.setattr(agent, "PipelineWorker", lambda *a, **k: None)
    agent._build_pipeline_worker(SimpleNamespace(input=lambda: None, output=lambda: None))
    return captured["user_params"]


def test_turns_end_on_smart_turn_then_the_llm_gate(monkeypatch):
    params = _user_params(monkeypatch)
    assert params.vad_analyzer is agent._SHARED_VAD
    strategies = params.user_turn_strategies
    assert isinstance(strategies, FilterIncompleteUserTurnStrategies)
    assert [type(s) for s in strategies.start] == [
        VADUserTurnStartStrategy,
        TranscriptionUserTurnStartStrategy,
    ]
    detector, gate = strategies.stop
    assert isinstance(detector, TurnAnalyzerUserTurnStopStrategy)
    assert isinstance(detector._turn_analyzer, LocalSmartTurnAnalyzerV3)
    assert isinstance(gate, LLMTurnCompletionUserTurnStopStrategy)


def test_empty_turns_that_cut_rory_off_are_recovered(monkeypatch):
    params = _user_params(monkeypatch)
    assert params.empty_user_turn is not None
    assert params.empty_user_turn.interrupted_prompt


def test_turn_strategies_build_with_the_real_pipeline(monkeypatch):
    from unittest.mock import MagicMock

    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from rory_pipecat.serializers import RawPCM16Serializer

    for key in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(key, "test")
    transport = FastAPIWebsocketTransport(
        websocket=MagicMock(),
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, serializer=RawPCM16Serializer()
        ),
    )
    assert agent._build_pipeline_worker(transport) is not None
