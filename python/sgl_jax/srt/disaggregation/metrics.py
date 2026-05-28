"""PD Prometheus metrics.

All metric names follow the RFC schema. If ``prometheus_client`` is
not installed the module installs no-op stubs so production code can
emit metrics unconditionally without breaking tests / CPU-only
environments.

Emit points (see callers):
  * ``pd_state_transition_total`` — incremented from
    :py:meth:`StateHolder._transition_to`.
  * ``pd_transfer_bytes_total`` — incremented from
    :py:meth:`JaxTransferWrapper.register_pull` (direction="net"
    register-side; the d2h/h2d directions are populated from
    :py:meth:`QueueHostKVPool.copy_from_device` and from D's
    receiver-finalization).
  * ``pd_transfer_duration_seconds`` — observed at three phases:
    bootstrap lookup (in the decode Mixin), pull (in the receiver),
    ack (in the sender's ``_on_ack``).
  * ``pd_transfer_inflight`` — incremented on
    :py:meth:`JaxTransferKVManager.create_sender` /
    ``create_receiver``, decremented when the entry is pruned.
  * ``pd_host_pool_used_buffers`` — set by
    :py:meth:`QueueHostKVPool.alloc` / ``free``.
  * ``pd_transfer_failures_total`` — incremented on terminal FAILED
    transitions (orphan timeout / pull timeout / peer crash).
  * ``pd_bootstrap_registry_size`` — set by ``_Registry`` on
    register / unregister / evict.
"""

from __future__ import annotations

import logging
import threading
from contextlib import suppress

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # noqa: BLE001
    _PROM_AVAILABLE = False

    class _Noop:
        """Stand-in for missing prometheus_client primitives."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> _Noop:
            return self

        def inc(self, amount: float = 1.0) -> None:
            pass

        def dec(self, amount: float = 1.0) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    Counter = _Noop  # type: ignore[assignment]
    Gauge = _Noop  # type: ignore[assignment]
    Histogram = _Noop  # type: ignore[assignment]


# Module-level instances. Names follow Prometheus conventions
# (``pd_<thing>_<unit>``); labels follow the RFC schema.
PD_STATE_TRANSITION_TOTAL = Counter(
    "pd_state_transition_total",
    "PD KVPoll state transitions",
    labelnames=("from_state", "to_state", "role"),
)
PD_TRANSFER_BYTES_TOTAL = Counter(
    "pd_transfer_bytes_total",
    "Bytes transferred across PD KV path",
    labelnames=("direction", "role"),
)
PD_TRANSFER_DURATION_SECONDS = Histogram(
    "pd_transfer_duration_seconds",
    "Latency of each PD transfer phase",
    labelnames=("phase", "role"),
)
PD_TRANSFER_INFLIGHT = Gauge(
    "pd_transfer_inflight",
    "Currently in-flight PD transfers",
    labelnames=("role",),
)
PD_HOST_POOL_USED_BUFFERS = Gauge(
    "pd_host_pool_used_buffers",
    "QueueHostKVPool buffers currently allocated",
    labelnames=("pool_name",),
)
PD_TRANSFER_FAILURES_TOTAL = Counter(
    "pd_transfer_failures_total",
    "PD transfer failures by reason",
    labelnames=("reason", "role"),
)
PD_BOOTSTRAP_REGISTRY_SIZE = Gauge(
    "pd_bootstrap_registry_size",
    "Number of P registrations in the bootstrap registry",
)


def is_prometheus_available() -> bool:
    """True iff the real ``prometheus_client`` lib is installed."""

    return _PROM_AVAILABLE


# ---- helpers ----------------------------------------------------------------


class _DurationTimer:
    """Context manager / explicit timer that observes wall-clock
    elapsed time on a labelled histogram. Safe with the no-op
    histogram stub."""

    __slots__ = ("_metric", "_start")

    def __init__(self, metric) -> None:
        self._metric = metric
        self._start: float | None = None

    def __enter__(self) -> _DurationTimer:
        import time as _time

        self._start = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is None:
            return
        import time as _time

        elapsed = _time.perf_counter() - self._start
        try:
            self._metric.observe(elapsed)
        except Exception:  # noqa: BLE001
            logger.debug("metric observe failed", exc_info=True)


def time_phase(phase: str, role: str) -> _DurationTimer:
    """Return a context manager that records a phase duration into
    :data:`PD_TRANSFER_DURATION_SECONDS` with the given labels."""

    return _DurationTimer(PD_TRANSFER_DURATION_SECONDS.labels(phase=phase, role=role))


# Process-wide counter of (pool name -> currently allocated) so we
# only need ``alloc`` and ``free`` to send a delta and the gauge
# reflects truth even under races. The QueueHostKVPool already takes
# its own lock; this is a cheap separate counter used solely by the
# gauge.
_pool_in_use_lock = threading.Lock()
_pool_in_use: dict = {}


def host_pool_alloc(pool_name: str, count: int = 1) -> None:
    with _pool_in_use_lock:
        new = _pool_in_use.get(pool_name, 0) + count
        _pool_in_use[pool_name] = new
    with suppress(Exception):
        PD_HOST_POOL_USED_BUFFERS.labels(pool_name=pool_name).set(new)


def host_pool_free(pool_name: str, count: int = 1) -> None:
    with _pool_in_use_lock:
        new = max(0, _pool_in_use.get(pool_name, 0) - count)
        _pool_in_use[pool_name] = new
    with suppress(Exception):
        PD_HOST_POOL_USED_BUFFERS.labels(pool_name=pool_name).set(new)
