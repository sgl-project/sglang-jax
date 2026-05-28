"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        # overlap mode set worker need_prepare_lora_batch to False
        self.worker.need_prepare_lora_batch = False

        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        # output_queue carries the async (jax.Array) handles from the forward
        # worker; the new resolve thread drains it, blocks on materialization
        # (np.asarray / _value), and pushes ready Python values to resolved_queue.
        # This keeps materialization off the scheduler's critical path: while
        # scheduler does build_batch / process other work, resolve thread is
        # waiting for TPU forward to finish in parallel.
        self.output_queue = Queue()
        self.resolved_queue = Queue()
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.resolve_thread = threading.Thread(
            target=self.resolve_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.resolve_thread.start()
        self.parent_process = psutil.Process().parent()
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        self.async_gather_fn = jax.jit(lambda x: x, out_shardings=replicated_sharding)

    def get_model_runner(self):
        return self.worker.get_model_runner()

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def get_max_padded_size(self):
        return self.worker.get_max_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                # Signal the resolve thread to shut down too.
                self.output_queue.put(None)
                break

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map, self.mesh
            )

            # Run forward
            with jax.profiler.TraceAnnotation(f"forward_batch_generation {model_worker_batch.bid}"):
                logits_output, next_token_ids, cache_miss_count = (
                    self.worker.forward_batch_generation(
                        model_worker_batch,
                        model_worker_batch.launch_done,
                        sampling_metadata=sampling_metadata,
                        forward_metadata=forward_metadata,
                    )
                )
            next_token_ids = self.async_gather_fn(next_token_ids)
            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map,
                future_token_ids_ct,
                next_token_ids,
                self.mesh,
            )

            # Kick off async D2H copies HERE in the worker thread, immediately
            # after forward, so the PCIe transfer overlaps with the rest of the
            # worker loop (set_future_token_ids dispatch + queue handoff) and
            # whatever the scheduler thread is doing before it reads from us.
            # The scheduler's resolve_last_batch_result then just materializes
            # the results without firing the copies itself, shrinking the
            # critical-path block on `np.asarray(jax.Array)._value`.
            async_next_logprobs = (
                jax.copy_to_host_async(logits_output.next_token_logprobs)
                if logits_output.next_token_logprobs is not None
                else None
            )
            async_input_logprobs = (
                jax.copy_to_host_async(logits_output.input_token_logprobs)
                if logits_output.input_token_logprobs is not None
                else None
            )
            async_hidden_states = (
                jax.copy_to_host_async(logits_output.hidden_states)
                if logits_output.hidden_states is not None
                else None
            )
            async_next_tokens = jax.copy_to_host_async(next_token_ids)

            self.output_queue.put(
                (
                    None,
                    logits_output,
                    cache_miss_count,
                    {
                        "next_token_ids": async_next_tokens,
                        "next_token_logprobs": async_next_logprobs,
                        "input_token_logprobs": async_input_logprobs,
                        "hidden_states": async_hidden_states,
                    },
                )
            )

    def resolve_thread_func(self):
        try:
            self.resolve_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ResolveThread hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def resolve_thread_func_(self):
        """Drain output_queue, materialize JAX async handles to host (block on
        TPU forward completion + D2H copy), push ready Python values to
        resolved_queue.

        This runs on its own thread so the materialization wait (np.asarray /
        _value) does NOT sit on the scheduler's critical path. By the time the
        scheduler calls resolve_last_batch_result, the data is typically already
        sitting in resolved_queue (materialized during scheduler's build_batch /
        other host work).
        """
        while True:
            entry = self.output_queue.get()
            if entry is None:
                # Forward thread signaled shutdown.
                self.resolved_queue.put(None)
                break
            _, logits_output, cache_miss_count, async_handles = entry

            # Materialize. This blocks on TPU forward completion + the queued
            # D2H copy (kicked off in forward_thread_func_). Per-array waits
            # serialize, but the underlying transfers ran in parallel.
            if async_handles["next_token_logprobs"] is not None:
                logits_output.next_token_logprobs = np.asarray(
                    async_handles["next_token_logprobs"]
                ).tolist()
            if async_handles["input_token_logprobs"] is not None:
                logits_output.input_token_logprobs = np.asarray(
                    async_handles["input_token_logprobs"]
                ).tolist()
            if async_handles["hidden_states"] is not None:
                logits_output.hidden_states = np.asarray(async_handles["hidden_states"])
            next_token_ids = np.asarray(async_handles["next_token_ids"]).tolist()

            self.resolved_queue.put((logits_output, next_token_ids, cache_miss_count))

    def resolve_last_batch_result(self, launch_done: threading.Event | None = None):
        """
        Pull the next materialized result from the resolve thread.

        The forward thread queues the async (jax.Array) handles, and the
        resolve thread materializes them in parallel with whatever the
        scheduler is doing. By the time this is called, the materialization
        has typically already completed, so the queue.get() returns quickly
        with no wait on _value / TPU sync.
        """
        result = self.resolved_queue.get()
        logits_output, next_token_ids, cache_miss_count = result

        if launch_done is not None:
            launch_done.wait()

        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata = None,
    ) -> tuple[None, jax.Array, int]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                0,
                self.mesh,
                self.worker.model_config.vocab_size,
            )

        forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch
        )

        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora:
            self.worker.prepare_lora_batch(model_worker_batch)

        model_worker_batch.forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.worker.get_model_runner()
        )

        # Push a new batch to the queue (JAX handles synchronization automatically)
        self.input_queue.put(
            (
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            )
        )

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)

        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit
        return None, future_next_token_ids, 0

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    @property
    def page_size(self) -> int:
        return self.worker.page_size

    @property
    def sliding_window_size(self) -> int | None:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    def __delete__(self):
        self.input_queue.put((None, None, None, None))
