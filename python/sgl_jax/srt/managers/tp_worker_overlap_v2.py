"""Single-threaded worker for normal overlap scheduling."""

import dataclasses
from functools import partial

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.utils.overlap_utils import (
    create_relay_buffers,
    resolve_decode_relay_inputs,
    resolve_relay_inputs,
    update_relay_buffers,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs


@dataclasses.dataclass
class ForwardContext:
    batch: ModelWorkerBatch
    logits_output: LogitsProcessorOutput
    sampling_metadata: SamplingMetadata
    cache_miss_count: int


class ModelWorkerOverlap(ModelWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        model_class=None,
        precompile_params: dict | None = None,
    ):
        super().__init__(
            server_args=server_args,
            mesh=mesh,
            model_class=model_class,
            precompile_params=precompile_params,
        )
        self.need_prepare_lora_batch = False
        self.cur_sampling_info = None
        self.relay_buffers = create_relay_buffers(
            mesh,
            self.model_runner.req_to_token_pool,
            dp_size=self.dp_size,
        )
        relay_sharding = NamedSharding(mesh, P("data", None))
        input_sharding = NamedSharding(mesh, P("data"))
        self._resolve_relay = jax.jit(
            partial(
                resolve_relay_inputs,
                dp_size=self.dp_size,
                relay_sharding=relay_sharding,
                output_sharding=input_sharding,
            ),
            out_shardings=input_sharding,
        )
        self._resolve_decode_relay = jax.jit(
            partial(
                resolve_decode_relay_inputs,
                dp_size=self.dp_size,
                relay_sharding=relay_sharding,
                output_sharding=input_sharding,
            ),
            out_shardings=input_sharding,
        )
        self._update_relay = jax.jit(
            partial(
                update_relay_buffers,
                dp_size=self.dp_size,
                output_sharding=relay_sharding,
            )
        )

    def launch_forward(
        self,
        batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> ForwardContext:
        batch.sampling_info.update_penalties()
        self.cur_sampling_info = batch.sampling_info

        if sampling_metadata is None:
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                batch,
                0,
                self.mesh,
                self.model_config.vocab_size,
            )

        forward_metadata = self.model_runner.attn_backend.get_forward_metadata(batch)
        if self.server_args.enable_lora:
            self.prepare_lora_batch(batch)
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        if batch.forward_mode.is_decode():
            forward_batch.input_ids = self._resolve_decode_relay(
                self.relay_buffers,
                forward_batch.req_pool_indices,
                forward_batch.input_ids,
            )
        elif batch.relay_input_indices is not None:
            input_sharding = forward_batch.input_ids.sharding
            indices = jax.device_put(batch.relay_input_indices, input_sharding)
            mask = jax.device_put(batch.relay_input_mask, input_sharding)
            forward_batch.input_ids = self._resolve_relay(
                self.relay_buffers,
                indices,
                mask,
                forward_batch.input_ids,
            )
        batch.forward_batch = forward_batch

        logits_output, _, cache_miss_count = super().forward_batch_generation(
            batch,
            batch.launch_done,
            skip_sample=True,
            sampling_metadata=sampling_metadata,
            forward_metadata=forward_metadata,
        )
        return ForwardContext(
            batch=batch,
            logits_output=logits_output,
            sampling_metadata=sampling_metadata,
            cache_miss_count=cache_miss_count,
        )

    def launch_sample(
        self,
        context: ForwardContext,
    ) -> tuple[LogitsProcessorOutput, jax.Array, int]:
        import jax._src.test_util as jtu

        batch = context.batch
        logits_output = context.logits_output
        sampling_metadata = context.sampling_metadata

        self._update_grammar_vocab_mask(batch, sampling_metadata)
        with jtu.count_pjit_cpp_cache_miss() as count:
            next_token_ids, token_logprobs, sampled_output = self.model_runner.sample(
                logits_output,
                sampling_metadata,
            )
            cache_miss_count = context.cache_miss_count + count()

        if batch.return_output_logprob_only:
            logprobs = self.model_runner.compute_logprobs(token_logprobs, next_token_ids)
            logits_output.next_token_logprobs = logprobs
        if sampled_output is not None:
            logits_output = sampled_output

        self.relay_buffers = self._update_relay(
            self.relay_buffers,
            batch.forward_batch.req_pool_indices,
            next_token_ids,
        )

        output_ids = next_token_ids
        if self.dp_size > 1:
            from jax.experimental.multihost_utils import process_allgather

            output_ids = process_allgather(output_ids, tiled=True)

        output_ids.copy_to_host_async()
        for value in (
            logits_output.next_token_logprobs,
            logits_output.input_token_logprobs,
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.input_top_logprobs_val,
            logits_output.input_top_logprobs_idx,
            logits_output.input_token_ids_logprobs_val,
            logits_output.hidden_states,
        ):
            if isinstance(value, jax.Array):
                value.copy_to_host_async()

        return logits_output, output_ids, cache_miss_count

    def resolve_last_batch_result(
        self,
        logits_output: LogitsProcessorOutput,
        next_token_ids: jax.Array,
        batch: ModelWorkerBatch,
        cache_miss_count: int,
        launch_done=None,
    ) -> tuple[LogitsProcessorOutput, list[int], int]:
        next_token_ids = jax.device_get(next_token_ids).tolist()
        if batch.return_logprob or batch.return_output_logprob_only:
            self._materialize_logprobs_to_host(
                logits_output,
                batch,
                batch.logits_indices_selector,
            )
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = np.asarray(
                    logits_output.next_token_logprobs
                ).tolist()
        if logits_output.input_token_logprobs is not None:
            logits_output.input_token_logprobs = np.asarray(
                jax.device_get(logits_output.input_token_logprobs)
            ).tolist()
        if isinstance(logits_output.hidden_states, jax.Array):
            logits_output.hidden_states = np.asarray(jax.device_get(logits_output.hidden_states))
        if launch_done is not None:
            launch_done.wait()
        return logits_output, next_token_ids, cache_miss_count
