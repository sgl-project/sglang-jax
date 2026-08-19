import asyncio
import concurrent.futures
import copy
import threading
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class _WorkerState(threading.local):
    processor: Any = None


class MultimodalProcessorExecutor:
    def __init__(self, processor: Any, max_workers: int):
        self._processors = (
            [processor]
            if max_workers == 1
            else [copy.deepcopy(processor) for _ in range(max_workers)]
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sgl-jax-mm-processor",
        )
        self._worker_state = _WorkerState()
        self._processor_lock = threading.Lock()

    async def run(self, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run, function, args, kwargs)

    def _run(
        self,
        function: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        processor = self._worker_state.processor
        if processor is None:
            with self._processor_lock:
                processor = self._processors.pop()
            self._worker_state.processor = processor
        return function(*args, processor=processor, **kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
