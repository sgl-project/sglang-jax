import contextlib
import threading

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_lora_context = threading.local()


class LoraBatchContext:
    @staticmethod
    @contextlib.contextmanager
    def set_batch(forward_batch: ForwardBatch):
        try:
            _lora_context.batch = forward_batch
            yield
        finally:
            _lora_context.batch = None

    @staticmethod
    def get_batch() -> ForwardBatch | None:
        return getattr(_lora_context, "batch", None)
