import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_spec_worker import BaseDraftWorker
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.jax_utils import device_array


class EagleDraftWorker(ModelWorker, BaseDraftWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None

        ModelWorker.__init__(
            self,
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self._init_embed_and_head()
        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self._target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_model_runner(self):
        return self.model_runner

    def _init_embed_and_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
        else:
            if self.draft_model_runner.model.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                head.data = head.data[self.hot_token_ids]
            self.draft_model_runner.model.set_embed_and_head(embed, head)

    def generate_model_worker_batch(self, *args, **kwargs):
        return self.compilation_manager.generate_model_worker_batch(*args, **kwargs)

    def draft(self, model_worker_batch):
        raise NotImplementedError

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> EagleDraftInput:
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND
        # last_idx = np.cumsum(model_worker_batch.extend_seq_lens, axis=0) - 1

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        logits_output.next_token_logits = logits_output.next_token_logits[
            : model_worker_batch.real_bs, :
        ]
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]

        self.capture_for_decode(logits_output, forward_batch.spec_info)
        return forward_batch.spec_info

    def capture_for_decode(self, logits_output, draft_input: EagleDraftInput):
        topk_p, topk_index = _topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = logits_output.hidden_states

    def draft_extend_for_decode(self, model_worker_batch, batch_output):
        raise NotImplementedError


@functools.partial(jax.jit, static_argnames=["topk"])
def _topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
    topk_logits, topk_index = jax.lax.top_k(working_logits, topk)
    logsumexp = jax.nn.logsumexp(working_logits, axis=-1, keepdims=True)
    topk_probs = jnp.exp(topk_logits - logsumexp)

    if axis != -1:
        topk_probs = jnp.moveaxis(topk_probs, -1, axis)
        topk_index = jnp.moveaxis(topk_index, -1, axis)

    return topk_probs, topk_index
