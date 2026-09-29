"""The one clock Rory uses for customer-visible date decisions.

Ordinary deployments use UTC wall time.  A benchmark may set
``RORY_REFERENCE_TIME`` to the frozen instant carried by its immutable twin
snapshot.  That keeps the candidate and the twins on the same clock without
changing production behavior or teaching vendor clients about sandboxes.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone


def now() -> datetime:
    raw = os.environ.get("RORY_REFERENCE_TIME")
    if not raw:
        return datetime.now(timezone.utc)
    try:
        if raw.isdigit():
            value = datetime.fromtimestamp(int(raw), tz=timezone.utc)
        else:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (OSError, OverflowError, ValueError) as exc:
        raise RuntimeError("RORY_REFERENCE_TIME must be a UTC ISO timestamp or Unix epoch") from exc
    if value.tzinfo is None:
        raise RuntimeError("RORY_REFERENCE_TIME must include a timezone")
    return value.astimezone(timezone.utc)


def today() -> date:
    return now().date()
