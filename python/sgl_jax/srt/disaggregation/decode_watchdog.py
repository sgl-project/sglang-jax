"""Diagnostic watchdog for the decode-side scheduler event loop.

The decode event loop is single-threaded: it intakes requests (with a
synchronous bootstrap HTTP lookup), polls KV receivers, and runs the
decode forward pass in the same thread. If any of those phases blocks,
token generation for *all* running requests freezes and the orphan
reaper's FAILED verdicts are never consumed (the consumer runs on the
same blocked thread).

This watchdog runs on a separate daemon thread. The loop calls
:meth:`beat` at each phase boundary (cheap: three field writes). When
the most recent beat is older than ``stall_threshold_s``, the watchdog
logs the stuck phase, a backlog snapshot, and the main thread's
traceback so a stress run pinpoints *which* phase / line is blocking.

It is pure observability: opt-in via ``disaggregation_decode_watchdog_seconds``
and off by default. It does not abort, retry, or otherwise alter loop
behavior.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress

logger = logging.getLogger(__name__)


class EventLoopWatchdog:
    """Detects a stalled event loop and dumps diagnostics once per stall."""

    def __init__(
        self,
        *,
        stall_threshold_s: float,
        check_interval_s: float = 1.0,
        snapshot_provider: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        traceback_dumper: Callable[[], None] | None = None,
    ) -> None:
        self._stall_threshold_s = stall_threshold_s
        self._check_interval_s = check_interval_s
        self._snapshot_provider = snapshot_provider
        self._clock = clock
        self._traceback_dumper = traceback_dumper or self._default_traceback_dumper
        self._phase = "init"
        self._beat_ts = clock()
        self._tick = 0
        # ``_tick`` is frozen while the loop is stuck, so reporting only
        # when it differs from the last reported tick yields exactly one
        # report per distinct stall and re-arms once the loop advances.
        self._last_reported_tick = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._stall_threshold_s > 0

    def beat(self, phase: str) -> None:
        """Mark the loop as alive and entering ``phase``. Hot-path cheap."""

        self._phase = phase
        self._tick += 1
        self._beat_ts = self._clock()

    def check_once(self, now: float | None = None) -> bool:
        """Single stall check. Returns True iff a stall was reported."""

        now = self._clock() if now is None else now
        age = now - self._beat_ts
        if age < self._stall_threshold_s:
            return False
        if self._tick == self._last_reported_tick:
            return False
        self._report(phase=self._phase, age=age, tick=self._tick)
        self._last_reported_tick = self._tick
        return True

    def _report(self, *, phase: str, age: float, tick: int) -> None:
        snapshot = ""
        if self._snapshot_provider is not None:
            with suppress(Exception):
                snapshot = self._snapshot_provider()
        logger.warning(
            "PD-DECODE-WATCHDOG stall detected: phase=%s age=%.1fs "
            "tick=%d backlog=[%s]; dumping main-thread traceback",
            phase,
            age,
            tick,
            snapshot,
        )
        with suppress(Exception):
            self._traceback_dumper()

    @staticmethod
    def _default_traceback_dumper() -> None:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._beat_ts = self._clock()
        self._thread = threading.Thread(
            target=self._loop,
            name="PD-DecodeEventLoopWatchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self._check_interval_s + 1.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with suppress(Exception):
                self.check_once()
            self._stop.wait(self._check_interval_s)
