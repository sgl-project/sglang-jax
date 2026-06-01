"""Connection ABC for PD disaggregation backends.

Defines the 4-tuple contract every transfer backend must implement:
``KVManager`` factory + ``KVSender`` / ``KVReceiver`` per-request handles +
the ``KVPoll`` lifecycle enum. The legal-transition table is shared across
backends and validated by ``_StateHolder._transition_to``; illegal
transitions raise ``ValueError`` so contract violations surface
immediately rather than corrupting state.
"""

from __future__ import annotations

import abc
import enum
import logging

logger = logging.getLogger(__name__)


class KVPoll(enum.Enum):
    """Lifecycle of a single KV transfer request."""

    BOOTSTRAPPING = "bootstrapping"
    WAITING_FOR_INPUT = "waiting_for_input"
    TRANSFERRING = "transferring"
    SUCCESS = "success"
    FAILED = "failed"


TERMINAL_STATES: frozenset[KVPoll] = frozenset({KVPoll.SUCCESS, KVPoll.FAILED})


LEGAL_TRANSITIONS: frozenset[tuple[KVPoll, KVPoll]] = frozenset(
    {
        (KVPoll.BOOTSTRAPPING, KVPoll.WAITING_FOR_INPUT),
        (KVPoll.WAITING_FOR_INPUT, KVPoll.TRANSFERRING),
        (KVPoll.TRANSFERRING, KVPoll.SUCCESS),
        (KVPoll.BOOTSTRAPPING, KVPoll.FAILED),
        (KVPoll.WAITING_FOR_INPUT, KVPoll.FAILED),
        (KVPoll.TRANSFERRING, KVPoll.FAILED),
    }
)


def is_legal_transition(current: KVPoll, next_state: KVPoll) -> bool:
    return (current, next_state) in LEGAL_TRANSITIONS


class StateHolder:
    """Shared lifecycle bookkeeping for ``KVSender`` / ``KVReceiver``.

    Backends compose this in addition to inheriting from the ABC; it
    centralises the transition table check so every backend gets identical
    enforcement of the ``KVPoll`` state machine. Public on purpose — every
    backend (current and future) is expected to use it directly.

    ``role`` is one of ``"prefill"`` / ``"decode"`` and feeds the
    metrics label used by sender/receiver state transitions. ``None``
    means the holder is used outside of a sender/receiver and no metric
    will be emitted.
    """

    def __init__(
        self,
        initial: KVPoll = KVPoll.BOOTSTRAPPING,
        *,
        role: str | None = None,
    ) -> None:
        self._state: KVPoll = initial
        self._role = role

    @property
    def state(self) -> KVPoll:
        return self._state

    @property
    def role(self) -> str | None:
        return self._role

    def _transition_to(self, next_state: KVPoll) -> None:
        if not is_legal_transition(self._state, next_state):
            raise ValueError(
                f"illegal KVPoll transition: " f"{self._state.value} -> {next_state.value}"
            )
        from_state = self._state
        self._state = next_state
        if self._role is not None:
            try:
                from sgl_jax.srt.disaggregation.common.metrics import PD_STATE_TRANSITION_TOTAL

                PD_STATE_TRANSITION_TOTAL.labels(
                    from_state=from_state.value,
                    to_state=next_state.value,
                    role=self._role,
                ).inc()
            except Exception:  # noqa: BLE001
                logger.debug("metrics emit failed", exc_info=True)


class KVManager(abc.ABC):
    """Per-process factory that produces sender/receiver handles."""

    @abc.abstractmethod
    def create_sender(self, req_id: str) -> KVSender: ...

    @abc.abstractmethod
    def create_receiver(self, req_id: str) -> KVReceiver: ...


class KVSender(abc.ABC):
    """Prefill-side per-request handle."""

    @abc.abstractmethod
    def init(self, kv_indices, transfer_id: str | None = None) -> None: ...

    @abc.abstractmethod
    def send(self) -> None: ...

    @abc.abstractmethod
    def poll(self) -> KVPoll: ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Drop backend-local retained terminal state."""

    @abc.abstractmethod
    def abort(self) -> None:
        """Abort the current transfer."""

    @abc.abstractmethod
    def failure_exception(self) -> None:
        """Raise the terminal transfer failure as an exception."""
        ...


class KVReceiver(abc.ABC):
    """Decode-side per-request handle."""

    @abc.abstractmethod
    def init(self, p_metadata) -> None: ...

    @abc.abstractmethod
    def poll(self) -> KVPoll: ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Drop backend-local retained terminal state."""

    @abc.abstractmethod
    def abort(self) -> None:
        """Abort the current transfer."""

    @abc.abstractmethod
    def failure_exception(self) -> None:
        """Raise the terminal transfer failure as an exception."""
        ...


