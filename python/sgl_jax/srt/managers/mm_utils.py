"""NumPy adaptation of SGLang's mm_utils shared-memory feature transport."""

import copy
import logging
import os
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShmPointerMMData:
    shm_name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def create(cls, array: np.ndarray):
        """Copy an array into a new shared segment owned by the receiver."""
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        try:
            # Reserve Linux tmpfs pages before writing, so exhaustion raises
            # OSError instead of killing the tokenizer with SIGBUS.
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(shm._fd, 0, array.nbytes)
            view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            np.copyto(view, array)
            del view
            return cls(shm.name, array.shape, array.dtype)
        except BaseException:
            shm.unlink()
            raise
        finally:
            shm.close()

    def materialize(self):
        """Copy to owned memory, then release the shared segment."""
        shm = shared_memory.SharedMemory(name=self.shm_name)
        try:
            return np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf).copy()
        finally:
            try:
                shm.unlink()
            finally:
                shm.close()

    def close_and_unlink(self):
        """Release a segment after reception or a failed send."""
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
        except FileNotFoundError:
            return
        try:
            shm.unlink()
        finally:
            shm.close()


def _map_features(request, transform):
    """Copy request containers and transform only multimodal feature fields."""
    mm = getattr(request, "mm_inputs", None)
    if mm is None:
        return request
    request = copy.copy(request)
    mm = copy.copy(mm)
    items = mm.get("mm_items", []) if isinstance(mm, dict) else mm.mm_items
    mapped = []
    for item in items:
        item = copy.copy(item)
        value = item.get("feature") if isinstance(item, dict) else item.feature
        if value is not None:
            value = _map_value(value, transform)
            if isinstance(item, dict):
                item["feature"] = value
            else:
                item.feature = value
        mapped.append(item)
    if isinstance(mm, dict):
        mm["mm_items"] = mapped
    else:
        mm.mm_items = mapped
    request.mm_inputs = mm
    return request


def _map_value(value, transform):
    if isinstance(value, (list, tuple)):
        return type(value)(_map_value(item, transform) for item in value)
    return transform(value)


def wrap_shm_features(request):
    """Wrap CPU arrays in SHM references without changing the original request."""
    segments = []

    def wrap(value):
        # Device arrays remain on the existing path; never force a D2H copy.
        if isinstance(value, np.ndarray) and value.nbytes and not value.dtype.hasobject:
            segment = ShmPointerMMData.create(value)
            segments.append(segment)
            return segment
        return value

    try:
        return _map_features(request, wrap)
    except BaseException as error:
        for segment in segments:
            segment.close_and_unlink()
        if not isinstance(error, OSError):
            raise
        logger.warning("Shared memory unavailable; sending multimodal features inline")
        return request


def discard_shm_features(request):
    """Release shared segments for a failed or unconsumed request."""

    def discard(value):
        if isinstance(value, ShmPointerMMData):
            value.close_and_unlink()
        return value

    _map_features(request, discard)


def unwrap_shm_features(request):
    """Restore features before dispatch or broadcasting to another host."""
    try:
        return _map_features(
            request,
            lambda value: value.materialize() if isinstance(value, ShmPointerMMData) else value,
        )
    except BaseException:
        # Also release unvisited segments if materialization failed partway.
        discard_shm_features(request)
        raise


def send_mm_request(socket, request):
    """Send wrapped features and release them if the synchronous/async send fails."""
    wire_request = wrap_shm_features(request)
    try:
        result = socket.send_pyobj(wire_request)
    except BaseException:
        discard_shm_features(wire_request)
        raise

    if wire_request is not request and result is not None:

        def on_sent(future):
            if future.cancelled() or future.exception() is not None:
                discard_shm_features(wire_request)

        result.add_done_callback(on_sent)
    return result
