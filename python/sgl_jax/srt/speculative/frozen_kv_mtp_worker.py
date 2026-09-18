"""Frozen-KV MTP speculative worker (Gemma4 assistant drafts).

The draft model's attention is Q-only: it reads K/V from the *target* verifier's
KV cache instead of owning one. The worker has its own Frozen-KV state and
target-cache setup, while reusing only the common EAGLE-shaped proposal
mechanics through ``EagleDraftWorkerBase``.

Mirrors upstream sgl-project/sglang, where ``SpeculativeAlgorithm.create_worker``
routes FROZEN_KV_MTP to a dedicated ``FrozenKVMTPWorkerV2`` checked before the
EAGLE branch, rather than folding it into the multi-layer (per-MTP-head) worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorkerBase
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs_device
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.frozen_kv_mtp_seed import verify_frozen_kv_mtp_chain_greedy
from sgl_jax.srt.speculative.relay_buffer import (
    SpecSeedRelayBuffers,
    create_spec_seed_relay_buffers,
    gather_spec_seed_relay_buffers,
    update_spec_seed_relay_buffers,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _frozen_kv_verify_and_publish(
    draft_tokens,
    target_logits,
    target_hidden,
    positions,
    seed_relay_buffers,
    relay_future_indices,
    relay_valid_mask,
    *,
    draft_token_num: int,
    dp_size: int,
):
    """Verify a linear chain and publish its next target seed on device."""
    predict, accept_lens, accept_index = verify_frozen_kv_mtp_chain_greedy(
        draft_tokens,
        target_logits,
        draft_token_num=draft_token_num,
    )

    padded_bs = accept_lens.shape[0]
    row_offset = jnp.clip(accept_lens - 1, 0, draft_token_num - 1)
    seed_rows = jnp.arange(padded_bs, dtype=jnp.int32) * draft_token_num + row_offset

    def gather_rows(values, indices, *, output_bs=None):
        sharding = jax.typeof(values).sharding
        if isinstance(sharding, NamedSharding):
            mesh = sharding.mesh
            result_bs = padded_bs if output_bs is None else output_bs
            if "data" in mesh.shape and result_bs % int(mesh.shape["data"]) == 0:
                out_spec = P("data", *([None] * (values.ndim - 1)))
            else:
                out_spec = P(*([None] * values.ndim))
            return values.at[indices].get(out_sharding=NamedSharding(mesh, out_spec))
        return jnp.take(values, indices, axis=0)

    seed_token = gather_rows(predict, seed_rows)
    seed_hidden = gather_rows(target_hidden, seed_rows)
    seed_valid = relay_valid_mask & (accept_lens > 0)
    updated_seed_relay_buffers = update_spec_seed_relay_buffers(
        seed_relay_buffers,
        relay_future_indices,
        seed_valid,
        seed_token,
        seed_token,
        seed_hidden,
        seed_valid,
        dp_size=dp_size,
    )

    accept_width = draft_token_num
    request_ids = jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    per_request_last = request_ids * draft_token_num + draft_token_num - 1
    accept_index_sharding = jax.typeof(accept_index).sharding
    if isinstance(accept_index_sharding, NamedSharding) and not accept_index_sharding.mesh.empty:
        per_request_last = jax.sharding.reshard(per_request_last, accept_index_sharding)
    safe_index = jnp.where(accept_index >= 0, accept_index, per_request_last)
    selected_logits = gather_rows(target_logits, safe_index, output_bs=accept_index.shape[0])
    selected_hidden = gather_rows(target_hidden, safe_index, output_bs=accept_index.shape[0])
    selected_positions = gather_rows(positions, safe_index, output_bs=accept_index.shape[0])
    return (
        selected_logits,
        selected_hidden,
        selected_positions,
        predict,
        accept_lens,
        updated_seed_relay_buffers,
    )


def _build_frozen_kv_fused_verify():
    """Build one target-forward/verify/seed-publication executable.

    The Frozen-KV seed is a target token/hidden-state pair selected by the
    verification result.  Keeping that selection and the request-indexed relay
    update in the same JIT as the target forward prevents eager gather,
    reshard, and scatter operations from becoming separate TPU dispatches.
    Scheduler-owned request ordering and bucket selection remain inputs.
    """

    @partial(
        jax.jit,
        donate_argnames=("target_memory_pools", "seed_relay_buffers"),
        static_argnames=("target_model_state_def", "draft_token_num", "dp_size"),
    )
    def fused_verify(
        target_model_def,
        target_model_state_def,
        target_leaves,
        target_forward_batch,
        target_memory_pools,
        target_logits_metadata,
        seed_relay_buffers,
        relay_future_indices,
        relay_valid_mask,
        *,
        draft_token_num: int,
        dp_size: int,
    ):
        target_state = jax.tree_util.tree_unflatten(target_model_state_def, target_leaves)
        target_model = nnx.merge(target_model_def, target_state)
        target_output, target_pool_updates, _, _ = target_model(
            target_forward_batch,
            target_memory_pools,
            target_logits_metadata,
        )

        (
            selected_logits,
            selected_hidden,
            selected_positions,
            predict,
            accept_lens,
            updated_seed_relay_buffers,
        ) = _frozen_kv_verify_and_publish(
            target_forward_batch.spec_info.draft_token,
            target_output.next_token_logits,
            target_output.hidden_states,
            target_forward_batch.positions,
            seed_relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            draft_token_num=draft_token_num,
            dp_size=dp_size,
        )

        return (
            target_pool_updates,
            selected_logits,
            selected_hidden,
            selected_positions,
            predict,
            accept_lens,
            updated_seed_relay_buffers,
        )

    return fused_verify


def _select_frozen_kv_proposal_start(
    relay_draft_token,
    relay_hidden,
    relay_seed_mask,
    seed_logits,
    seed_hidden,
):
    """Select the first proposal state for live target seeds and padding.

    Every live row consumes its target token/hidden pair in the assistant
    before proposing. False rows are inactive scheduler padding and retain
    zero relay state. Keeping this selection inside the proposal JIT avoids
    separately dispatched reshard/select operations.
    """
    seed_token = _frozen_kv_top1_token(seed_logits)
    # Model outputs can retain a replicated batch dimension even when relay
    # state is P("data").  Normalize both cases to the relay layouts before
    # select; explicit TPU sharding rejects semantically equivalent P(None)
    # and P("data") values even when the data mesh axis has size one.
    token_sharding = jax.typeof(relay_draft_token).sharding
    hidden_sharding = jax.typeof(relay_hidden).sharding
    relay_seed_mask = jnp.asarray(relay_seed_mask, dtype=bool)
    if isinstance(token_sharding, NamedSharding) and not token_sharding.mesh.empty:
        seed_token = jax.sharding.reshard(seed_token, token_sharding)
        relay_seed_mask = jax.sharding.reshard(relay_seed_mask, token_sharding)
    if isinstance(hidden_sharding, NamedSharding) and not hidden_sharding.mesh.empty:
        seed_hidden = jax.sharding.reshard(seed_hidden, hidden_sharding)
    return (
        jnp.where(relay_seed_mask, seed_token, relay_draft_token),
        jnp.where(relay_seed_mask[:, None], seed_hidden, relay_hidden),
    )


def _frozen_kv_top1_token(logits):
    """Compute the global top-1 token while retaining the active mesh contract."""
    sharding = jax.typeof(logits).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        # The vocabulary must be replicated before argmax, but the request
        # dimension must stay data-sharded.  Replicating the whole tensor here
        # gives the resulting token vector P(None), which cannot be selected
        # against the P("data") relay mask under explicit TPU sharding.
        logits = jax.sharding.reshard(logits, NamedSharding(sharding.mesh, P("data", None)))
        token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return jax.sharding.reshard(token, NamedSharding(sharding.mesh, P("data")))
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _select_frozen_kv_prefill_seed_hidden(hidden_states, logits_indices):
    """Select each request's final prompt hidden state with DP-local indices.

    Target prefill returns token-major hidden states, while ``logits_indices``
    contains indices local to each data-parallel token shard.  A plain global
    gather is therefore wrong for DP ranks after rank zero.  This is the same
    local-gather contract used by :class:`LogitsProcessor`; keeping it here
    lets the seed selection remain inside Frozen-KV's relay publication JIT.
    """
    hidden_sharding = jax.typeof(hidden_states).sharding
    if isinstance(hidden_sharding, NamedSharding) and not hidden_sharding.mesh.empty:
        mesh = hidden_sharding.mesh
        if "data" in mesh.shape:
            hidden_states = jax.sharding.reshard(
                hidden_states, NamedSharding(mesh, P("data", None))
            )
            logits_indices = jax.sharding.reshard(logits_indices, NamedSharding(mesh, P("data")))

            def select_local(local_states, local_indices):
                return local_states[local_indices]

            return jax.shard_map(
                select_local,
                mesh=mesh,
                in_specs=(P("data", None), P("data")),
                out_specs=P("data", None),
            )(hidden_states, logits_indices)
    return hidden_states[logits_indices]


def _build_frozen_kv_fused_draft_extend(num_steps: int):
    """Build one request-relay/seed/recurrent-proposal executable.

    Frozen-KV uses one assistant repeatedly, unlike generic multi-layer NEXTN
    where each MTP depth owns a distinct model runner.  This dedicated program
    therefore mirrors the recurrent shape used by EAGLE3 while preserving the
    Frozen contract: every assistant attention reads target KV and every pool
    update is the model's unchanged target-buffer pass-through.
    """
    assert num_steps > 0

    @partial(
        jax.jit,
        donate_argnames=("memory_pools",),
        static_argnames=("model_state_def", "num_steps", "dp_size"),
    )
    def draft_extend_fused(
        model_def,
        model_state_def,
        model_leaves,
        forward_batch,
        memory_pools,
        logits_metadata,
        metadata_per_step,
        seed_relay_buffers,
        relay_future_indices,
        relay_seed_mask,
        *,
        num_steps: int,
        dp_size: int,
    ):
        state = jax.tree_util.tree_unflatten(model_state_def, model_leaves)
        model = nnx.merge(model_def, state)
        (
            relay_verified_id,
            relay_draft_token,
            relay_hidden,
            _stored_seed_mask,
        ) = gather_spec_seed_relay_buffers(
            seed_relay_buffers,
            relay_future_indices,
            dp_size=dp_size,
        )

        # Both post-prefill and post-verify rows need Gemma's target-hidden ->
        # assistant-state transition. This call also runs for padded rows so
        # the executable shape is independent of the live request count.
        forward_batch.input_ids = relay_verified_id
        forward_batch.positions = forward_batch.seq_lens - 1
        forward_batch.spec_info.hidden_states = relay_hidden
        forward_batch.attn_backend.forward_metadata = metadata_per_step[0]
        output, pool_updates, _, _ = model(
            forward_batch,
            memory_pools,
            logits_metadata,
        )
        memory_pools.replace_all(pool_updates)
        token, hidden = _select_frozen_kv_proposal_start(
            relay_draft_token,
            relay_hidden,
            relay_seed_mask,
            output.next_token_logits,
            output.hidden_states,
        )
        proposal_tokens = [token]
        proposal_token_sharding = jax.typeof(token).sharding

        # The seed call produced proposal zero.  Each remaining call consumes
        # the previous assistant token/hidden state at the same target-KV view
        # used by the legacy Frozen recurrence for that speculative position.
        for step in range(num_steps - 1):
            forward_batch.input_ids = token
            forward_batch.positions = forward_batch.seq_lens + step
            forward_batch.spec_info.hidden_states = hidden
            forward_batch.attn_backend.forward_metadata = metadata_per_step[step]
            output, pool_updates, _, _ = model(
                forward_batch,
                memory_pools,
                logits_metadata,
            )
            memory_pools.replace_all(pool_updates)
            token = _frozen_kv_top1_token(output.next_token_logits)
            if (
                isinstance(proposal_token_sharding, NamedSharding)
                and not proposal_token_sharding.mesh.empty
            ):
                token = jax.sharding.reshard(token, proposal_token_sharding)
            hidden = output.hidden_states
            proposal_tokens.append(token)

        token_list = jnp.stack(proposal_tokens, axis=1)
        padded_bs = forward_batch.seq_lens.shape[0]
        packed = build_chain_verify_inputs_device(
            relay_verified_id,
            token_list,
            forward_batch.seq_lens - 1,
            num_steps + 1,
            padded_bs,
        )
        return (
            pool_updates,
            packed[0],
            packed[1],
            packed[2].reshape(padded_bs, num_steps + 1),
            packed[3].reshape(padded_bs, num_steps + 1),
            packed[4].reshape(padded_bs, num_steps + 1),
        )

    return draft_extend_fused


@register_pytree_node_class
@dataclass
class FrozenKvMtpDraftInput(EagleDraftInput):
    """Checked per-request state for Frozen-KV non-overlap batch merging.

    Target KV pages remain owned by each request's token-pool row.  This state
    holds the matching token/hidden/length handles, which must stay in the same
    order whenever a completed prefill joins an active decode batch.
    """

    _REQUIRED_FIELDS = ("topk_p", "topk_index", "hidden_states", "verified_id", "allocate_lens")
    _OPTIONAL_PER_REQUEST_FIELDS = (
        "accept_length",
        "accept_length_cpu",
        "new_seq_lens",
    )
    # Opaque relay descriptors deliberately keep model tensors on device.  This
    # host-side, request-aligned bit is the only Frozen-specific information
    # needed before the next static scheduler bucket is selected: true rows
    # contain valid target token/hidden seeds; false rows are padded slots.
    relay_seed_mask: np.ndarray | None = None
    # Upstream EAGLE no longer carries this legacy overlap relay handle. Frozen
    # keeps the explicit sentinel so its non-overlap state validation can reject
    # an accidental mixed lifecycle before a batch is merged or filtered.
    pending_draft_extend_result: object | None = None

    def tree_flatten(self):
        """Keep Frozen's scheduler-visible length state across JAX pytrees.

        ``EagleDraftInput`` omits allocation and logical sequence lengths
        because generic EAGLE recreates them around its legacy handoff. Frozen
        non-overlap merging consumes those values directly, so dropping them
        during a scatter/copy would break the request-to-page association.
        """
        children, aux_data = super().tree_flatten()
        return (
            children
            + (
                self.allocate_lens,
                self.new_seq_lens,
                self.relay_seed_mask,
            ),
            aux_data,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # ``tree_flatten`` appends three Frozen-only children after the base
        # EAGLE state.  Pass exactly the base children back to its unflattener.
        obj = EagleDraftInput.tree_unflatten.__func__(cls, aux_data, children[:-3])
        obj.allocate_lens = children[-3]
        obj.new_seq_lens = children[-2]
        obj.relay_seed_mask = children[-1]
        obj.pending_draft_extend_result = None
        return obj

    def _batch_size(self) -> int:
        if self.future_indices is not None:
            return int(np.asarray(self.future_indices).shape[0])
        if self.verified_id is None:
            raise ValueError("Frozen-KV MTP state is missing verified_id.")
        return int(np.asarray(self.verified_id).shape[0])

    def _validate_non_overlap_state(self) -> None:
        """Ensure every state array describes the same request/KV allocation."""
        if self.pending_draft_extend_result is not None:
            raise ValueError(
                "Frozen-KV MTP non-overlap state must not carry a pending draft-extend result."
            )

        batch_size = self._batch_size()
        if self.future_indices is not None:
            # A relay descriptor deliberately carries only scheduler-owned
            # request metadata. The variable-sized model tensors live in a
            # request-indexed device buffer and are restored only after the
            # scheduler has selected the next static bucket. This is the same
            # generic future_indices lifecycle used by existing relay users.
            unexpected = [
                field
                for field in (
                    "topk_p",
                    "topk_index",
                    "hidden_states",
                    "verified_id",
                    "accept_length",
                )
                if getattr(self, field, None) is not None
            ]
            if unexpected:
                raise ValueError(
                    "Frozen-KV MTP relay state must not retain model tensors: " f"{unexpected}."
                )
            for field in (
                "allocate_lens",
                "new_seq_lens",
                "accept_length_cpu",
                "relay_seed_mask",
            ):
                value = getattr(self, field, None)
                if value is not None and np.asarray(value).shape[0] != batch_size:
                    raise ValueError(
                        f"Frozen-KV MTP relay field {field!r} has "
                        f"{np.asarray(value).shape[0]} rows; expected {batch_size}."
                    )
            return

        for field in self._REQUIRED_FIELDS:
            value = getattr(self, field)
            if value is None:
                raise ValueError(f"Frozen-KV MTP state is missing required field {field!r}.")
            if np.asarray(value).shape[0] != batch_size:
                raise ValueError(
                    f"Frozen-KV MTP field {field!r} has {np.asarray(value).shape[0]} rows; "
                    f"expected {batch_size}."
                )
        for field in self._OPTIONAL_PER_REQUEST_FIELDS:
            value = getattr(self, field)
            if value is not None and np.asarray(value).shape[0] != batch_size:
                raise ValueError(
                    f"Frozen-KV MTP optional field {field!r} has "
                    f"{np.asarray(value).shape[0]} rows; expected {batch_size}."
                )

        allocate_lens = np.asarray(self.allocate_lens)
        if np.any(allocate_lens < 0):
            raise ValueError("Frozen-KV MTP allocation lengths must be non-negative.")
        if self.new_seq_lens is not None and np.any(allocate_lens < np.asarray(self.new_seq_lens)):
            raise ValueError(
                "Frozen-KV MTP allocation length is shorter than its committed sequence length."
            )

    @staticmethod
    def _select(value, indices):
        return None if value is None else np.asarray(value)[indices]

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        """Keep state rows in exact request-pool order after finish/retract."""
        if self.future_indices is not None:
            # Reuse the generic opaque-relay metadata operation. It does not
            # materialize model state because no model state is present here.
            if self.relay_seed_mask is not None:
                src = np.asarray(self.relay_seed_mask)
                idx = (
                    slice(0, len(new_indices))
                    if has_been_filtered and len(new_indices) == len(src)
                    else np.asarray(new_indices, dtype=np.int32)
                )
                self.relay_seed_mask = src[idx]
            super().filter_batch(new_indices, has_been_filtered)
            self._validate_non_overlap_state()
            return
        self._ensure_host()
        self._validate_non_overlap_state()
        new_indices = np.asarray(new_indices, dtype=np.int32)
        indices = (
            slice(0, len(new_indices))
            if has_been_filtered and len(new_indices) == self._batch_size()
            else new_indices
        )
        for field in self._REQUIRED_FIELDS + self._OPTIONAL_PER_REQUEST_FIELDS:
            setattr(self, field, self._select(getattr(self, field), indices))
        self._validate_non_overlap_state()

    def trim_to_length(self, n: int) -> None:
        """Trim opaque relay metadata with the same request rows as the base input."""
        n = int(n)
        if self.future_indices is not None:
            super().trim_to_length(n)
            if self.relay_seed_mask is not None:
                self.relay_seed_mask = np.asarray(self.relay_seed_mask, dtype=bool)[:n]
            self._validate_non_overlap_state()
            return

        super().trim_to_length(n)
        self._validate_non_overlap_state()

    def merge_batch(self, other: EagleDraftInput) -> None:
        """Append a completed Frozen-KV prefill without reinterpreting pages.

        ``allocate_lens`` stays paired with its request-pool row.  The
        attention backend derives physical pages from that row and applies the
        per-request page-compaction rule; this method only concatenates rows.
        """
        if not isinstance(other, FrozenKvMtpDraftInput):
            raise TypeError(
                "Frozen-KV MTP non-overlap merge requires FrozenKvMtpDraftInput on both "
                f"sides; got {type(other).__name__}."
            )
        if self.future_indices is not None or other.future_indices is not None:
            # Generic relay descriptors can be concatenated without touching
            # device-resident model state. Mixing descriptor and non-relay
            # state is intentionally rejected by the base contract: callers
            # must publish *both* a running decode and a new prefill before
            # admitting their merged batch.
            left_mask, right_mask = self.relay_seed_mask, other.relay_seed_mask
            if (left_mask is None) != (right_mask is None):
                raise ValueError(
                    "Frozen-KV MTP relay merge requires relay_seed_mask on both sides or neither."
                )
            super().merge_batch(other)
            if left_mask is not None:
                self.relay_seed_mask = np.concatenate(
                    [np.asarray(left_mask, dtype=bool), np.asarray(right_mask, dtype=bool)]
                )
            self._validate_non_overlap_state()
            return
        self._ensure_host()
        other._ensure_host()
        self._validate_non_overlap_state()
        other._validate_non_overlap_state()
        for field in self._REQUIRED_FIELDS:
            setattr(
                self,
                field,
                np.concatenate(
                    [np.asarray(getattr(self, field)), np.asarray(getattr(other, field))]
                ),
            )
        for field in self._OPTIONAL_PER_REQUEST_FIELDS:
            left, right = getattr(self, field), getattr(other, field)
            if left is None and right is None:
                continue
            if left is None or right is None:
                if field in ("accept_length", "accept_length_cpu"):
                    # A prefill has no prior verification outcome. This field
                    # is optional scheduler bookkeeping; a merged batch cannot
                    # truthfully carry a per-request accept length for every
                    # row until after its next target verification.
                    setattr(self, field, None)
                    continue
                raise ValueError(
                    f"Frozen-KV MTP merge requires optional field {field!r} on both sides or neither."
                )
            setattr(self, field, np.concatenate([np.asarray(left), np.asarray(right)]))
        self._validate_non_overlap_state()


class FrozenKvMtpDraftWorker(EagleDraftWorkerBase):
    """Dedicated Frozen-KV draft worker whose KV reads are redirected.

    The worker owns the Gemma-specific target-KV setup and the Frozen state
    contract. It reuses only the EAGLE-shaped proposal mechanics from the
    abstract ``EagleDraftWorkerBase``; the target worker and scheduler still
    own shared verification, admission, padding buckets, and merge orchestration.
    """

    draft_input_cls = FrozenKvMtpDraftInput

    def new_draft_input(self, **kwargs) -> FrozenKvMtpDraftInput:
        """Construct the persistent Frozen-KV state owned by this worker."""
        return self.draft_input_cls(**kwargs)

    @property
    def draft_model_runner(self):
        return self._worker.get_model_runner()

    @property
    def mesh(self):
        return self._worker.mesh

    @property
    def model_config(self):
        return self._worker.model_config

    @property
    def compilation_manager(self):
        return self._worker.compilation_manager

    @property
    def max_req_len(self):
        return self._worker.max_req_len

    def get_max_padded_size(self):
        return self._worker.get_max_padded_size()

    def __init__(self, server_args, target_worker: ModelWorker):
        from sgl_jax.srt.layers.kv_share import compute_mtp_kv_share_map
        from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
            _build_non_hybrid_memory_pools,
        )

        self.server_args = server_args
        self.target_worker_ref = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.hot_token_ids = None

        # NB: ModelWorker.get_memory_pool() returns (req_to_token_pool,
        # token_to_kv_pool_ALLOCATOR) -- every other caller discards the second
        # value with `_`. The KV cache itself only comes off the model runner.
        req_to_token_pool, _ = target_worker.get_memory_pool()
        target_token_to_kv_pool = target_worker.model_runner.token_to_kv_pool

        self._worker = ModelWorker(
            server_args,
            target_worker.mesh,
            req_to_token_pool=req_to_token_pool,
            is_draft_worker=True,
        )

        self._kv_share_map = compute_mtp_kv_share_map(
            self._worker.model_config.hf_config, target_worker.model_config.hf_config
        )
        logger.info("Frozen-KV MTP KV-share map: %s", self._kv_share_map)

        # Alias the target's KV pool in. Combined with the layer_id redirection
        # below, this makes every draft attention read land on the target's cache.
        draft_runner = self._worker.model_runner
        draft_runner.token_to_kv_pool = target_token_to_kv_pool
        draft_runner.memory_pools = _build_non_hybrid_memory_pools(target_token_to_kv_pool)

        self._redirect_layer_ids(draft_runner.model, target_worker.model_runner.model)

        self._share_embed_head(target_worker)

        # The target's SWA remap table must be visible to the draft's backend,
        # since the draft now indexes the target's (possibly hybrid) pool.
        target_allocator = target_worker.model_runner.token_to_kv_pool_allocator
        target_swa_mapping = getattr(target_allocator, "full_to_swa_index_mapping", None)
        if target_swa_mapping is not None:
            object.__setattr__(draft_runner.attn_backend, "swa_index_mapping", target_swa_mapping)

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        self._worker.model_runner.initialize_jit()

        # Allocated lazily by the outer Frozen worker only for the opt-in
        # seed-relay route.  Do not use BaseSpecWorker.spec_relay_buffers:
        # that buffer selects the generic overlap/fused execution path, which
        # Frozen-KV intentionally does not implement.
        self.seed_relay_buffers: SpecSeedRelayBuffers | None = None
        self._jit_publish_seed_relay = None
        self._jit_publish_prefill_seed_relay = None

        (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()

    def init_seed_relay_buffers(self) -> None:
        """Allocate the generic request-indexed top-1 relay once per server.

        This owns values, not scheduling: request IDs, batch merge, and bucket
        selection stay in ``ScheduleBatch``.  The buffer is separate from the
        generic overlap relay because Frozen-KV's normal path is non-overlap.
        """
        if getattr(self, "seed_relay_buffers", None) is not None:
            return
        hidden_dtype = jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32
        req_to_token_pool, _ = self.target_worker_ref.get_memory_pool()
        self.seed_relay_buffers = create_spec_seed_relay_buffers(
            self.mesh,
            req_to_token_pool,
            dp_size=self.server_args.dp_size,
            hidden_size=self.target_worker_ref.model_config.hidden_size,
            hidden_dtype=hidden_dtype,
        )

    def _init_jit_seed_relay_ops(self) -> None:
        """Create generic bucketed device programs for Frozen relay state.

        Request IDs and the choice of padded batch remain scheduler-owned host
        metadata. Once a scheduler bucket is chosen, compact precompile
        publication and token-major prefill seed selection must each remain a
        single device program. Calling the pure relay helpers eagerly creates
        tiny dispatches and host synchronization points. This mirrors
        DFlash's cached relay JITs; it does not introduce batch policy.
        """
        if (
            getattr(self, "_jit_publish_seed_relay", None) is not None
            and getattr(self, "_jit_publish_prefill_seed_relay", None) is not None
        ):
            return

        from functools import partial

        data_sharding = NamedSharding(self.mesh, P("data"))
        hidden_sharding = NamedSharding(self.mesh, P("data", None))
        replicated_sharding = NamedSharding(self.mesh, P())

        @partial(
            jax.jit,
            donate_argnames=("buffers",),
            static_argnames=("dp_size",),
        )
        def publish(
            buffers,
            future_indices,
            selector,
            verified_id,
            draft_token_ids,
            hidden_states,
            is_target_seed,
            *,
            dp_size: int,
        ):
            future_indices = jax.sharding.reshard(future_indices, data_sharding)
            # These are compact live-request rows, not the scheduler's padded
            # DP-attention bucket. A c1 prefill is valid when dp_size > 1, and
            # even a divisible live-row count need not be balanced by rank.
            # Replicate the compact values, then use the scheduler-provided
            # selector to scatter them into their real DP-padded slots below.
            selector = jax.sharding.reshard(selector, replicated_sharding)
            verified_id = jax.sharding.reshard(verified_id, replicated_sharding)
            draft_token_ids = jax.sharding.reshard(draft_token_ids, replicated_sharding)
            hidden_states = jax.sharding.reshard(hidden_states, replicated_sharding)
            is_target_seed = jax.sharding.reshard(is_target_seed, replicated_sharding)

            total_bs = future_indices.shape[0]

            def scatter_rows(value, *, output_sharding, fill_value=0):
                out = jax.sharding.reshard(
                    jnp.full((total_bs,) + value.shape[1:], fill_value, dtype=value.dtype),
                    output_sharding,
                )
                return out.at[selector].set(value, out_sharding=output_sharding)

            valid_mask = (
                jax.sharding.reshard(jnp.zeros((total_bs,), dtype=bool), data_sharding)
                .at[selector]
                .set(True, out_sharding=data_sharding)
            )
            return update_spec_seed_relay_buffers(
                buffers,
                future_indices,
                valid_mask,
                scatter_rows(verified_id, output_sharding=data_sharding),
                scatter_rows(draft_token_ids, output_sharding=data_sharding),
                scatter_rows(hidden_states, output_sharding=hidden_sharding),
                scatter_rows(is_target_seed, output_sharding=data_sharding, fill_value=False),
                dp_size=dp_size,
            )

        @partial(
            jax.jit,
            donate_argnames=("buffers",),
            static_argnames=("dp_size",),
        )
        def publish_prefill(
            buffers,
            future_indices,
            selector,
            verified_id,
            target_hidden,
            logits_indices,
            *,
            dp_size: int,
        ):
            """Select the target prefill seed and publish it in one dispatch."""
            future_indices = jax.sharding.reshard(future_indices, data_sharding)
            verified_id = jax.sharding.reshard(verified_id, data_sharding)
            logits_indices = jax.sharding.reshard(logits_indices, data_sharding)
            selector = jax.sharding.reshard(selector, replicated_sharding)
            seed_hidden = _select_frozen_kv_prefill_seed_hidden(
                target_hidden,
                logits_indices,
            )
            total_bs = future_indices.shape[0]
            valid_mask = (
                jax.sharding.reshard(jnp.zeros((total_bs,), dtype=bool), data_sharding)
                .at[selector]
                .set(True, out_sharding=data_sharding)
            )
            return update_spec_seed_relay_buffers(
                buffers,
                future_indices,
                valid_mask,
                verified_id,
                verified_id,
                seed_hidden,
                valid_mask,
                dp_size=dp_size,
            )

        self._jit_publish_seed_relay = publish
        self._jit_publish_prefill_seed_relay = publish_prefill

    def _publish_seed_relay(
        self,
        *,
        model_worker_batch: ModelWorkerBatch,
        verified_id,
        draft_token_ids,
        hidden_states,
        is_target_seed,
    ) -> None:
        """Publish compact Frozen state into the scheduler's current bucket.

        The verifier/prefill outputs are compact request rows, while a relay
        update must have a DP-divisible leading dimension.  Scatter those rows
        onto the already-selected scheduler slots on device; this is the last
        place where their dynamic size is visible.  Subsequent merge/filter
        touches only host metadata and the next draft gathers the selected
        static bucket from this request-indexed store.
        """
        if getattr(self, "seed_relay_buffers", None) is None:
            return
        self._init_jit_seed_relay_ops()
        selector = jnp.asarray(model_worker_batch.logits_indices_selector, dtype=jnp.int32)
        if selector.shape[0] != jnp.asarray(verified_id).shape[0]:
            raise ValueError(
                "Frozen-KV relay publication must receive one compact row per selector: "
                f"selector={selector.shape[0]}, rows={jnp.asarray(verified_id).shape[0]}."
            )

        with jax.set_mesh(self.mesh):
            self.seed_relay_buffers = self._jit_publish_seed_relay(
                self.seed_relay_buffers,
                model_worker_batch.req_pool_indices,
                selector,
                verified_id,
                draft_token_ids,
                hidden_states,
                is_target_seed,
                dp_size=self.server_args.dp_size,
            )

    def _publish_prefill_seed_relay(
        self,
        *,
        model_worker_batch: ModelWorkerBatch,
        target_hidden,
        verified_id,
    ) -> None:
        """Publish the seed produced by target prefill without a draft forward."""
        if getattr(self, "seed_relay_buffers", None) is None:
            raise RuntimeError("Frozen-KV prefill requires initialized seed relay buffers.")
        self._init_jit_seed_relay_ops()
        total_bs = int(model_worker_batch.req_pool_indices.shape[0])
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        if model_worker_batch.logits_indices.shape != (total_bs,):
            raise ValueError("Frozen-KV prefill logits indices must match the padded batch.")
        if jnp.asarray(verified_id).shape != (total_bs,):
            raise ValueError("Frozen-KV prefill tokens must match the padded batch.")

        with jax.set_mesh(self.mesh):
            self.seed_relay_buffers = self._jit_publish_prefill_seed_relay(
                self.seed_relay_buffers,
                model_worker_batch.req_pool_indices,
                jnp.asarray(selector, dtype=jnp.int32),
                verified_id,
                target_hidden,
                model_worker_batch.logits_indices,
                dp_size=self.server_args.dp_size,
            )

    @staticmethod
    def _compact_request_rows(value, selector: np.ndarray, expected: int):
        """Return compact live rows without changing their value dtype."""
        value = np.asarray(value)
        if value.shape[0] == expected:
            return value
        if value.shape[0] > int(selector.max(initial=-1)):
            return value[selector]
        raise ValueError(
            "Frozen-KV verify metadata does not match live requests or padded slots: "
            f"rows={value.shape[0]}, requests={expected}, selector={selector.tolist()}"
        )

    def draft_extend_for_prefill(self, model_worker_batch, hidden_states, next_token_ids) -> None:
        """Publish target prefill's token/hidden pair as the next draft seed.

        Gemma 4 Frozen-KV extension is seed selection, not an assistant model
        forward.  The first fused draft call consumes this pair and performs
        the assistant seed forward together with the recurrent proposal loop.
        """
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        self._publish_prefill_seed_relay(
            model_worker_batch=model_worker_batch,
            target_hidden=hidden_states,
            verified_id=next_token_ids,
        )
        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)[selector]
        model_worker_batch.spec_info_padded = self.new_draft_input(
            future_indices=np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)[
                selector
            ],
            allocate_lens=seq_lens.copy(),
            new_seq_lens=seq_lens.copy(),
            relay_seed_mask=np.ones((selector.size,), dtype=bool),
        )
        model_worker_batch.spec_info_padded._validate_non_overlap_state()
        model_worker_batch.return_hidden_states = False

    def draft_extend_for_decode(self, model_worker_batch, batch_output) -> None:
        """Install the descriptor already published by fused target verify."""
        next_state = batch_output.next_draft_input
        if not isinstance(next_state, FrozenKvMtpDraftInput):
            raise TypeError(
                "Frozen-KV verify must publish FrozenKvMtpDraftInput, got "
                f"{type(next_state).__name__}"
            )
        next_state._validate_non_overlap_state()
        model_worker_batch.spec_info_padded = next_state
        batch_output.accept_lens = np.asarray(batch_output.accept_lens, dtype=np.int32)

    @staticmethod
    def _pad_relay_rows_for_bucket(value, *, padded_bs: int, dp_size: int, fill_value=0):
        """Pad scheduler relay metadata with the generic DP bucket layout."""
        value = np.asarray(value)
        if value.shape[0] == padded_bs:
            return value
        if value.shape[0] > padded_bs or padded_bs % dp_size:
            raise ValueError(
                "Frozen-KV relay metadata cannot fit the selected draft bucket: "
                f"rows={value.shape[0]}, padded_bs={padded_bs}, dp_size={dp_size}."
            )
        if dp_size == 1 or value.shape[0] % dp_size:
            pad = [(0, padded_bs - value.shape[0])] + [(0, 0)] * (value.ndim - 1)
            return np.pad(value, pad, constant_values=fill_value)
        per_dp_real = value.shape[0] // dp_size
        per_dp_padded = padded_bs // dp_size
        reshaped = value.reshape((dp_size, per_dp_real) + value.shape[1:])
        pad = [(0, 0), (0, per_dp_padded - per_dp_real)] + [(0, 0)] * (value.ndim - 1)
        return np.pad(reshaped, pad, constant_values=fill_value).reshape(
            (padded_bs,) + value.shape[1:]
        )

    def _can_use_fused_draft(self, model_worker_batch: ModelWorkerBatch) -> bool:
        state = model_worker_batch.spec_info_padded
        runner = self.draft_model_runner
        return (
            self.topk == 1
            and self.speculative_num_draft_tokens == self.speculative_num_steps + 1
            and isinstance(state, FrozenKvMtpDraftInput)
            and state.future_indices is not None
            and state.relay_seed_mask is not None
            and self.seed_relay_buffers is not None
            and hasattr(runner, "_model_def")
            and hasattr(runner, "_model_state_def")
        )

    def _draft_fused_linear_chain(self, model_worker_batch: ModelWorkerBatch) -> None:
        """Produce and pack the Frozen top-1 chain in one TPU dispatch."""
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _make_forward_batch,
            _prepare_device_array,
            _prepare_logits_metadata,
        )
        from sgl_jax.srt.speculative.eagle_info import EagleVerifyInput

        state = model_worker_batch.spec_info_padded
        assert isinstance(state, FrozenKvMtpDraftInput)
        relay_future_indices = np.asarray(state.future_indices, dtype=np.int32)
        relay_seed_mask = np.asarray(state.relay_seed_mask, dtype=bool)
        if relay_future_indices.shape != relay_seed_mask.shape:
            raise ValueError(
                "Frozen-KV relay indices and seed mask must have identical shapes: "
                f"indices={relay_future_indices.shape}, mask={relay_seed_mask.shape}."
            )

        # padding_for_decode owns the generic scheduler bucket and target-page
        # metadata.  Install shape-only placeholders so it need not dispatch a
        # device gather before the fused program; actual values are restored
        # from the request-indexed relay inside that program.
        compact_bs = relay_future_indices.shape[0]
        hidden_size = self.target_worker_ref.model_config.hidden_size
        state.verified_id = np.empty((compact_bs,), dtype=np.int32)
        state.topk_index = np.empty((compact_bs, 1), dtype=np.int32)
        state.topk_p = np.empty((compact_bs, 1), dtype=np.float32)
        state.hidden_states = np.empty((compact_bs, hidden_size), dtype=np.float32)
        state.future_indices = None
        state.relay_seed_mask = None
        self.padding_for_decode(model_worker_batch)

        padded_bs = int(model_worker_batch.seq_lens.shape[0])
        dp_size = int(model_worker_batch.dp_size)
        relay_future_indices = self._pad_relay_rows_for_bucket(
            relay_future_indices,
            padded_bs=padded_bs,
            dp_size=dp_size,
        )
        relay_seed_mask = self._pad_relay_rows_for_bucket(
            relay_seed_mask,
            padded_bs=padded_bs,
            dp_size=dp_size,
            fill_value=False,
        )

        runner = self.draft_model_runner
        metadata_per_step = runner.attn_backend.get_eagle_multi_step_metadata(model_worker_batch)
        runner.attn_backend.forward_metadata = metadata_per_step[0]
        forward_batch = _make_forward_batch(model_worker_batch, runner)
        forward_batch.bid = model_worker_batch.bid
        logits_metadata = _prepare_logits_metadata(model_worker_batch, self.mesh)
        data_sharding = NamedSharding(self.mesh, P("data"))
        relay_future_indices = _prepare_device_array(
            relay_future_indices,
            data_sharding,
            "frozen_draft.relay_future_indices",
        )
        relay_seed_mask = _prepare_device_array(
            relay_seed_mask,
            data_sharding,
            "frozen_draft.relay_seed_mask",
        )
        if not hasattr(self, "_frozen_kv_fused_draft_jit_fn"):
            self._frozen_kv_fused_draft_jit_fn = _build_frozen_kv_fused_draft_extend(
                self.speculative_num_steps
            )

        with jax.set_mesh(self.mesh):
            (
                pool_updates,
                draft_tokens,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
            ) = self._frozen_kv_fused_draft_jit_fn(
                runner._model_def,
                runner._model_state_def,
                tuple(runner.model_state_leaves),
                forward_batch,
                runner.memory_pools,
                logits_metadata,
                tuple(metadata_per_step),
                self.seed_relay_buffers,
                relay_future_indices,
                relay_seed_mask,
                num_steps=self.speculative_num_steps,
                dp_size=dp_size,
            )

        runner.memory_pools.replace_all(pool_updates)
        model_worker_batch.spec_info_padded = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=None,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            spec_steps=self.speculative_num_steps,
            draft_token_num=self.speculative_num_draft_tokens,
        )

    def draft(self, model_worker_batch):
        """Run the only supported Frozen proposal path as one TPU dispatch."""
        if not self._can_use_fused_draft(model_worker_batch):
            raise RuntimeError(
                "Frozen-KV MTP requires its fused top-1 linear draft path; "
                "the relay descriptor, model runner, or launch configuration "
                "does not satisfy that contract."
            )
        return self._draft_fused_linear_chain(model_worker_batch)

    def _redirect_layer_ids(self, draft_model, target_model) -> None:
        """Point each draft layer at the target cache slot it shares K/V with.

        Both checks below guard silent corruption rather than crashes: a missing
        entry leaves a layer reading target layer 0, and a geometry mismatch
        reads the cache at the wrong stride. Neither raises on its own.
        """
        for draft_layer_idx, layer in enumerate(draft_model.layers):
            draft_name = f"draft_layer.{draft_layer_idx}"
            if draft_name not in self._kv_share_map:
                raise ValueError(
                    f"Frozen-KV MTP: no KV-share entry for {draft_name}. That layer "
                    f"would read layer 0 of the target cache, silently producing garbage."
                )
            target_name = self._kv_share_map[draft_name]
            target_layer_idx = int(target_name.split(".")[-1])

            target_attn = target_model.model.layers[target_layer_idx].self_attn.attn
            draft_attn = layer.self_attn.attn
            if (draft_attn.head_dim, draft_attn.kv_head_num) != (
                target_attn.head_dim,
                target_attn.kv_head_num,
            ):
                raise ValueError(
                    f"Frozen-KV MTP KV-share geometry mismatch: {draft_name} has "
                    f"(head_dim={draft_attn.head_dim}, kv_heads={draft_attn.kv_head_num}) "
                    f"but target {target_name} cache holds "
                    f"(head_dim={target_attn.head_dim}, kv_heads={target_attn.kv_head_num})."
                )
            if bool(draft_attn.sliding_window_size) != bool(target_attn.sliding_window_size):
                raise ValueError(
                    f"Frozen-KV MTP KV-share attention-type mismatch: {draft_name} "
                    f"(sliding={bool(draft_attn.sliding_window_size)}) mapped to "
                    f"{target_name} (sliding={bool(target_attn.sliding_window_size)})."
                )

            draft_attn.layer_id = target_layer_idx
            logger.debug(
                "Frozen-KV MTP: draft_layer.%d -> target layer.%d",
                draft_layer_idx,
                target_layer_idx,
            )

    def _share_embed_head(self, target_worker: ModelWorker) -> None:
        """Bind the target's embedding; the assistant keeps its own lm_head."""
        embed, head = target_worker.model_runner.model.get_embed_and_head()
        m = self._worker.model_runner.model
        if getattr(m, "load_lm_head_from_target", False):
            m.set_embed_and_head(embed, head)
        else:
            m.set_embed(embed)


class FrozenKvMtpWorker(EAGLEWorker):
    """EAGLE orchestration with a frozen-KV draft worker.

    Startup bucket selection and dummy-batch construction are inherited from
    ``EAGLEWorker`` just like ordinary multi-layer MTP. Target-model prefill is
    compiled by the normal model runner; Frozen-KV does not compile or execute
    a separate assistant prefill. Instead it publishes target prefill's final
    token/hidden pair for the first fused proposal round.
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        super().__init__(
            server_args,
            target_worker,
            draft_worker=FrozenKvMtpDraftWorker(server_args, target_worker),
        )
        # This is deliberately independent from ``spec_relay_buffers``:
        # setting the latter would select EAGLE's overlap/fused machinery
        # rather than Frozen-KV's non-overlap path.
        self.draft_worker.init_seed_relay_buffers()

    def supports_non_fused_spec_prefill_precompile(self) -> bool:
        """Warm target prefill plus seed publication, never an assistant EXTEND.

        The generic speculative startup driver owns the scheduler's batch and
        token buckets. Frozen-KV opts into its non-overlap prefill call so both
        the ordinary target prefill and the small final-hidden relay publisher
        compile before traffic. ``draft_extend_for_prefill`` is seed-only, so
        this capability does not restore the removed assistant prefill path.
        """
        return True

    def prepare_spec_decode_precompile_state(self, model_worker_batch, spec_info):
        """Seed the relay so startup exercises the real fused decode route."""
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        verified_id = np.asarray(spec_info.verified_id)[selector]
        draft_token_ids = np.asarray(spec_info.topk_index)[selector, 0]
        hidden_states = np.asarray(spec_info.hidden_states)[selector]
        self.draft_worker._publish_seed_relay(
            model_worker_batch=model_worker_batch,
            verified_id=verified_id,
            draft_token_ids=draft_token_ids,
            hidden_states=hidden_states,
            is_target_seed=np.ones(selector.shape, dtype=bool),
        )
        allocate_lens = np.asarray(spec_info.allocate_lens)[selector]
        return self.draft_worker.new_draft_input(
            future_indices=np.asarray(
                model_worker_batch.req_pool_indices,
                dtype=np.int32,
            )[selector],
            allocate_lens=allocate_lens,
            new_seq_lens=np.asarray(model_worker_batch.seq_lens, dtype=np.int32)[selector],
            relay_seed_mask=np.ones(selector.shape, dtype=bool),
        )

    def _verify_fused_linear_chain(
        self,
        model_worker_batch: ModelWorkerBatch,
        spec_info,
        cur_allocate_lens,
        forward_metadata,
    ):
        """Run the supported Frozen top-1 verify as one compiled dispatch."""
        from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _active_dp_slot_mask,
            _count_pjit_cpp_cache_miss,
            _make_forward_batch,
            _prepare_device_array,
            _prepare_logits_metadata,
        )

        target_mr = self.target_worker.model_runner
        target_mr.attn_backend.forward_metadata = forward_metadata
        target_forward_batch = _make_forward_batch(model_worker_batch, target_mr)
        target_forward_batch.bid = model_worker_batch.bid
        target_logits_metadata = _prepare_logits_metadata(model_worker_batch, self.mesh)

        total_bs = int(model_worker_batch.seq_lens.shape[0])
        data_sharding = NamedSharding(self.mesh, P("data"))
        relay_future_indices = _prepare_device_array(
            model_worker_batch.req_pool_indices,
            data_sharding,
            "frozen_verify.relay_future_indices",
        )
        relay_valid_mask = _prepare_device_array(
            _active_dp_slot_mask(model_worker_batch, total_bs),
            data_sharding,
            "frozen_verify.relay_valid_mask",
        )
        if not hasattr(self, "_frozen_kv_fused_verify_jit_fn"):
            self._frozen_kv_fused_verify_jit_fn = _build_frozen_kv_fused_verify()

        with jax.set_mesh(self.mesh), _count_pjit_cpp_cache_miss() as count:
            (
                target_pool_updates,
                selected_logits,
                selected_hidden,
                selected_positions,
                predict_device,
                accept_lengths_device,
                updated_seed_relay_buffers,
            ) = self._frozen_kv_fused_verify_jit_fn(
                target_mr._model_def,
                target_mr._model_state_def,
                tuple(target_mr.model_state_leaves),
                target_forward_batch,
                target_mr.memory_pools,
                target_logits_metadata,
                self.draft_worker.seed_relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                draft_token_num=spec_info.draft_token_num,
                dp_size=int(model_worker_batch.dp_size),
            )
            cache_miss_count = count()

        target_mr.memory_pools.replace_all(target_pool_updates)
        self.draft_worker.seed_relay_buffers = updated_seed_relay_buffers
        model_worker_batch.positions = selected_positions

        # The scheduler must inspect accepted lengths to commit output tokens;
        # this compact vector is the sole mandatory device-to-host boundary.
        if hasattr(accept_lengths_device, "copy_to_host_async"):
            accept_lengths_device.copy_to_host_async()
        accept_padded = np.asarray(accept_lengths_device, dtype=np.int32).reshape(-1)
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32).reshape(
            -1
        )
        accept_live = self.draft_worker._compact_request_rows(
            accept_padded, selector, selector.size
        )
        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32).reshape(-1)
        allocate_lens = self.draft_worker._compact_request_rows(
            cur_allocate_lens, selector, selector.size
        )
        req_indices = self.draft_worker._compact_request_rows(
            model_worker_batch.req_pool_indices, selector, selector.size
        )
        next_draft_input = self.draft_worker.new_draft_input(
            future_indices=req_indices,
            allocate_lens=allocate_lens,
            new_seq_lens=seq_lens[selector] + accept_live + 1,
            accept_length_cpu=accept_live.copy(),
            relay_seed_mask=np.ones((selector.size,), dtype=bool),
        )
        next_draft_input._validate_non_overlap_state()
        model_worker_batch.spec_info_padded = next_draft_input
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=selected_logits,
                hidden_states=selected_hidden,
            ),
            next_token_ids=predict_device,
            next_draft_input=next_draft_input,
            accept_lens=accept_padded,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens=None):
        """Dedicated Frozen-KV target verify and seed-relay handoff.

        Ordinary EAGLE's ``verify`` materializes predict, acceptance indices,
        and acceptance lengths immediately, then uses those CPU values to pick
        target-hidden rows.  Frozen-KV instead consumes the device acceptance
        result first: it selects each accepted target row and publishes the
        next assistant seed into its device relay.  Only the small acceptance
        length vector is then read by the scheduler-facing descriptor.

        Target forward, acceptance, accepted-row selection, and relay
        publication are one compiled executable. Server admission rejects
        sampling/tree contracts outside this greedy top-1 linear path.
        """
        if not self.server_args.disable_overlap_schedule:
            raise RuntimeError("Frozen-KV MTP does not support overlap scheduling.")

        from sgl_jax.srt.speculative.eagle_info import EagleVerifyInput

        if cur_allocate_lens is None:
            cur_allocate_lens = self._get_cur_allocate_lens(model_worker_batch)
        spec_info = model_worker_batch.spec_info_padded
        if not isinstance(spec_info, EagleVerifyInput):
            raise TypeError(
                "Frozen-KV target verify requires EagleVerifyInput, got "
                f"{type(spec_info).__name__}."
            )

        spec_info.allocate_lens = cur_allocate_lens
        spec_info.prepare_for_verify(model_worker_batch)
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        # Frozen-KV MTP's public contract is a top-k-one chain. Verify that
        # chain directly against the target's native output sharding, following
        # DFlash's dedicated greedy target-verify design.
        native_chain_verify = (
            model_worker_batch.sampling_info.is_all_greedy
            and spec_info.custom_mask is None
            and spec_info.draft_token_num == spec_info.spec_steps + 1
        )
        fused_verify_ready = (
            native_chain_verify
            and getattr(self.draft_worker, "seed_relay_buffers", None) is not None
            and hasattr(self.target_worker.model_runner, "_model_def")
            and hasattr(self.target_worker.model_runner, "_model_state_def")
        )
        if not fused_verify_ready:
            raise RuntimeError(
                "Frozen-KV MTP requires its fused greedy linear verify path; "
                "the relay buffer, model runner, or verify metadata is incompatible."
            )
        return self._verify_fused_linear_chain(
            model_worker_batch,
            spec_info,
            cur_allocate_lens,
            forward_metadata,
        )
