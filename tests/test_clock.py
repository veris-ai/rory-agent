from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rory_tools.clock import now, today


def test_reference_time_pins_all_agent_date_decisions(monkeypatch) -> None:
    monkeypatch.setenv("RORY_REFERENCE_TIME", "2026-09-01T22:15:30Z")
    assert now() == datetime(2026, 9, 1, 22, 15, 30, tzinfo=timezone.utc)
    assert today().isoformat() == "2026-09-01"


def test_reference_time_accepts_the_snapshot_epoch(monkeypatch) -> None:
    monkeypatch.setenv("RORY_REFERENCE_TIME", "1788300930")
    assert now().timestamp() == 1788300930


def test_malformed_or_naive_reference_time_fails_closed(monkeypatch) -> None:
    for value in ("not-a-time", "2026-09-01T22:15:30"):
        monkeypatch.setenv("RORY_REFERENCE_TIME", value)
        with pytest.raises(RuntimeError, match="RORY_REFERENCE_TIME"):
            now()
