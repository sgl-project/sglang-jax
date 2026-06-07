import logging
import pickle
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)


class CommunicationBackend(ABC):
    """Abstract communication backend."""

    @abstractmethod
    def recv_requests(self) -> list[Any]:
        """Receive requests (non-blocking, return empty list if none)."""
        pass

    @abstractmethod
    def send_pyobj(self, result: Any) -> None:
        """Send result to other components."""
        pass


class QueueBackend(CommunicationBackend):
    """Queue-based communication for Stage mode."""

    def __init__(self, in_queue: Queue, out_queue: Queue):
        self._in_queue = in_queue
        self._out_queue = out_queue

    def recv_requests(self) -> list[Any]:
        reqs = []
        while True:
            try:
                req = self._in_queue.get_nowait()
                reqs.append(req)
            except Empty:
                break
        return reqs

    def wait_for_new_requests(self, timeout: float = 0.0):
        time.sleep(timeout)

    def send_pyobj(self, result: Any) -> None:
        self._out_queue.put(result)


class MultiHostQueueBackend(QueueBackend):
    """Queue backend for a Stage that runs SPMD across multiple hosts.

    A multi-host stage (e.g. MiMo-V2.5's 16-chip AR backbone over 4 v6e hosts) runs the
    SAME jitted forward on every rank; jax handles the cross-host collectives. But only
    rank0's GlobalScheduler receives requests and feeds the in-queue — non-rank0 queues
    stay empty, so without coordination rank0's forward would block on a collective that
    the other ranks never enter, deadlocking the cluster.

    This backend makes ``recv_requests`` lockstep across ranks via a ZMQ PUB/SUB channel
    (the same ``pub_sub_addr`` the standard Scheduler uses):
    - rank0 drains its local in-queue and PUBLISHES the batch (even when empty);
    - non-rank0 ignore their (empty) queue and SUBSCRIBE to rank0's batch.
    Every rank then drives its AR forward with identical inputs in the same loop turn.

    Only rank0 produces output (its GlobalScheduler owns detokenizer IO), so ``send_pyobj``
    is a no-op on non-rank0.  See prework appendix "multi-host (SPMD)" §C/§D.3.
    """

    def __init__(self, in_queue: Queue, out_queue: Queue, *, node_rank: int, pub_sub_addr: str):
        super().__init__(in_queue, out_queue)
        self.node_rank = node_rank
        self._pub_sub_addr = pub_sub_addr
        self._publisher = None
        self._subscriber = None
        self._init_sockets()

    def _init_sockets(self):
        import zmq

        ctx = zmq.Context.instance()
        if self.node_rank == 0:
            self._publisher = ctx.socket(zmq.PUB)
            self._publisher.bind(self._pub_sub_addr)
        else:
            self._subscriber = ctx.socket(zmq.SUB)
            self._subscriber.connect(self._pub_sub_addr)
            self._subscriber.setsockopt(zmq.SUBSCRIBE, b"")
            self._subscriber.setsockopt(zmq.RCVTIMEO, 60000)

    def recv_requests(self) -> list[Any]:
        if self.node_rank == 0:
            reqs = super().recv_requests()
            try:
                self._publisher.send(pickle.dumps(reqs))
            except Exception as exc:  # pragma: no cover - transport error path
                logger.error("[MultiHostQueue pub rank0] send failed: %s", exc)
            return reqs
        # non-rank0: batch comes only from rank0's broadcast
        try:
            data = self._subscriber.recv()
        except Exception:
            # timeout / transient: no batch this turn (rank0 idle)
            return []
        try:
            return pickle.loads(data)
        except Exception as exc:  # pragma: no cover
            logger.error("[MultiHostQueue sub rank%s] decode failed: %s", self.node_rank, exc)
            return []

    def send_pyobj(self, result: Any) -> None:
        # Only rank0 owns the GlobalScheduler output path; other ranks discard.
        if self.node_rank == 0:
            self._out_queue.put(result)
