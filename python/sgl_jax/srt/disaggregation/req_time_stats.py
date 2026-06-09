"""Per-request time_stats for PD disaggregation.

Records wall-clock marks at a request's lifecycle points and derives a
phase-by-phase latency breakdown. Each role (prefill / decode) records its
own marks in its own process; the breakdown is logged when
``--enable-request-time-stats-logging`` is set.

The structure is deliberately dependency-free (no jax / numpy) so the
hot-path cost is a dict insert and it is trivially unit-testable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Ordered (start_mark, end_mark, phase_label) per role. A phase is emitted
# only when both endpoint marks are present, so partial requests degrade
# gracefully instead of reporting bogus durations.
_PHASE_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "prefill": (
        ("queue_entry", "forward_start", "queue"),
        ("forward_start", "forward_done", "forward"),
        ("transfer_start", "transfer_done", "transfer"),
        ("queue_entry", "transfer_done", "total"),
    ),
    "decode": (
        ("bootstrap_start", "bootstrap_done", "bootstrap"),
        ("prealloc_entry", "transfer_entry", "prealloc_wait"),
        ("transfer_entry", "first_token", "kv_wait"),
        ("first_token", "completion", "decode"),
        ("bootstrap_start", "completion", "total"),
    ),
}


class TimeStats:
    """Lifecycle marks + derived phase durations for one request."""

    __slots__ = ("role", "marks", "_clock")

    def __init__(self, role: str, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.role = role
        self.marks: dict[str, float] = {}
        self._clock = clock

    def mark(self, name: str) -> None:
        """Record the current time for ``name`` (first write wins)."""
        if name not in self.marks:
            self.marks[name] = self._clock()

    def duration(self, start: str, end: str) -> float | None:
        a = self.marks.get(start)
        b = self.marks.get(end)
        if a is None or b is None:
            return None
        return b - a

    def phases(self) -> dict[str, float]:
        """Role-specific phase durations, skipping any with unset endpoints."""
        out: dict[str, float] = {}
        for start, end, label in _PHASE_SPECS.get(self.role, ()):
            d = self.duration(start, end)
            if d is not None:
                out[label] = d
        return out


def format_time_stats(ts: TimeStats, *, req_id: str) -> str:
    phases = ts.phases()
    if phases:
        body = " ".join(f"{label}={dur * 1000:.1f}ms" for label, dur in phases.items())
    else:
        # Unknown role or no derivable phases: fall back to raw marks so the
        # line is still informative.
        body = " ".join(sorted(ts.marks)) or "(no marks)"
    return f"PD-TIME-STATS role={ts.role} req_id={req_id} {body}"


def maybe_log_time_stats(ts: TimeStats | None, *, req_id: str, enabled: bool) -> None:
    """Log the phase breakdown when ``enabled`` and ``ts`` is present."""
    if not enabled or ts is None:
        return
    logger.info("%s", format_time_stats(ts, req_id=req_id))
