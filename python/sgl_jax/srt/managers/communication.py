import logging
import select
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
        self._read_fd = -1
        try:
            self._read_fd = self._in_queue._reader.fileno()
        except Exception:
            logger.warning(
                "QueueBackend could not get fileno from in_queue; "
                "falling back to sleep-based waiting."
            )
            pass

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
        if self._read_fd != -1:
            try:
                # Wait until the queue's file descriptor is ready to be read.
                select.select([self._read_fd], [], [], timeout)
            except Exception:
                time.sleep(timeout)  # Fallback
        else:
            # Fallback for systems where fileno is not available
            time.sleep(timeout)

    def send_pyobj(self, result: Any) -> None:
        self._out_queue.put(result)
