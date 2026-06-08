from concurrent.futures import Future, ThreadPoolExecutor

from sgl_jax.srt.speculative.overlap_future import make_spec_decode_future_result


class SpecWorkerClient:
    def __init__(self, worker):
        self.worker = worker
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spec-worker")

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def forward_batch_speculative_generation(self, model_worker_batch) -> Future:
        return self.executor.submit(
            self.worker.forward_batch_speculative_generation,
            model_worker_batch,
        )

    def resolve_last_batch_result(self, future: Future):
        return make_spec_decode_future_result(future.result())

    def close(self):
        self.executor.shutdown(wait=True)
