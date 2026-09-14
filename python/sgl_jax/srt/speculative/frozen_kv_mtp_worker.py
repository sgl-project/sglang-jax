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

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.speculative.base_worker import replicate_to_mesh
from sgl_jax.srt.speculative.eagle_draft_worker import (
    EagleDraftWorkerBase,
    topk_probs_from_logits,
)
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs_device
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.frozen_kv_mtp_seed import (
    FrozenKvMtpSeedState,
    select_after_verify,
    verify_frozen_kv_mtp_chain_greedy,
)
from sgl_jax.srt.speculative.relay_buffer import (
    SpecSeedRelayBuffers,
    create_spec_seed_relay_buffers,
    gather_spec_seed_relay_buffers,
    update_spec_seed_relay_buffers,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.jax_utils import device_array

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
    """Select the first proposal state for target-seed and prefill rows.

    A post-verify row must consume its accepted target token/hidden pair in the
    assistant before proposing.  A newly-prefilled row already contains that
    first assistant proposal in the relay.  Keeping this selection inside the
    proposal JIT avoids separately dispatched reshard/select operations when
    the scheduler merges both row kinds into one static bucket.
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

        # Post-verify rows first need Gemma's target-hidden -> assistant-state
        # transition.  This call also runs for prefill-origin padded rows so the
        # executable shape is independent of the scheduler's dynamic mixture;
        # their already-computed proposal state is selected below.
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
    # The seed is deliberately optional while prefill and legacy compatibility
    # states are being migrated. Once present, it is the sole source for the
    # next Frozen proposal's token/target-hidden pair.
    seed_state: FrozenKvMtpSeedState | None = None
    # Opaque relay descriptors deliberately keep model tensors on device.  This
    # host-side, request-aligned bit is the only Frozen-specific information
    # needed before the next static scheduler bucket is selected: rows marked
    # true need an assistant seed forward; false rows are fresh prefills and
    # already contain ordinary one-token proposal state in the relay buffer.
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
                self.seed_state,
                self.relay_seed_mask,
            ),
            aux_data,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # ``tree_flatten`` appends four Frozen-only children after the base
        # EAGLE state.  Pass exactly the base children back to its unflattener.
        obj = EagleDraftInput.tree_unflatten.__func__(cls, aux_data, children[:-4])
        obj.allocate_lens = children[-4]
        obj.new_seq_lens = children[-3]
        obj.seed_state = children[-2]
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
                    "seed_state",
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
        if self.seed_state is not None and self.seed_state.batch_size != batch_size:
            raise ValueError(
                "Frozen-KV seed state has a different request count from draft state: "
                f"seed={self.seed_state.batch_size}, state={batch_size}."
            )

    @staticmethod
    def _select(value, indices):
        return None if value is None else np.asarray(value)[indices]

    def _invalid_seed_state(self) -> FrozenKvMtpSeedState:
        """Represent prefill-origin rows that have no post-verify seed yet.

        A merged non-overlap batch can contain both an existing speculative
        request (which has a target token/hidden seed from verification) and a
        request that has just completed prefill.  The latter starts from its
        normal draft-input fields, not a synthetic target-verify row.  Carry a
        shape-compatible, invalid seed entry for it so one batched draft can
        retain both forms of input.
        """
        batch_size = self._batch_size()
        hidden = jnp.asarray(self.hidden_states)
        return FrozenKvMtpSeedState(
            bonus_token=jnp.asarray(self.verified_id, dtype=jnp.int32),
            target_hidden=hidden,
            committed_lens=jnp.asarray(
                (
                    self.new_seq_lens
                    if self.new_seq_lens is not None
                    else np.zeros(batch_size, dtype=np.int32)
                ),
                dtype=jnp.int32,
            ),
            allocate_lens=jnp.asarray(self.allocate_lens, dtype=jnp.int32),
            request_indices=jnp.arange(batch_size, dtype=jnp.int32),
            valid_mask=jnp.zeros((batch_size,), dtype=bool),
        )

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
        if self.seed_state is not None:
            self.seed_state = self.seed_state.filter(jnp.asarray(indices, dtype=jnp.int32))
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

        old_size = self._batch_size()
        super().trim_to_length(n)
        if self.seed_state is not None and n < old_size:
            self.seed_state = self.seed_state.filter(jnp.arange(n, dtype=jnp.int32))
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
        left_seed = self.seed_state or self._invalid_seed_state()
        right_seed = other.seed_state or other._invalid_seed_state()

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
        self.seed_state = left_seed.merge(right_seed)
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
        self._jit_gather_seed_relay = None

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
        metadata.  Once a scheduler bucket is chosen, however, compact-row
        scatter/update and request-indexed gather must execute as a single
        device program.  Calling the pure relay helpers eagerly creates many
        tiny dispatches and turns otherwise asynchronous JAX arrays into host
        synchronization points.  This mirrors DFlash's cached relay JITs;
        it does not introduce a Frozen-specific batch policy.
        """
        if getattr(self, "_jit_publish_seed_relay", None) is not None:
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

        @partial(jax.jit, static_argnames=("dp_size",))
        def gather(buffers, future_indices, relay_seed_mask, *, dp_size: int):
            future_indices = jax.sharding.reshard(future_indices, data_sharding)
            relay_seed_mask = jax.sharding.reshard(relay_seed_mask, data_sharding)
            verified_id, draft_token_ids, hidden_states, _is_target_seed = (
                gather_spec_seed_relay_buffers(buffers, future_indices, dp_size=dp_size)
            )
            # `relay_seed_mask` is scheduler-owned lifecycle metadata.  It is
            # authoritative for this round; `is_target_seed` remains stored as
            # part of the generic relay contract for diagnostics/future users.
            return (
                verified_id,
                draft_token_ids,
                hidden_states,
                future_indices,
                relay_seed_mask,
            )

        self._jit_publish_seed_relay = publish
        self._jit_gather_seed_relay = gather

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

    def publish_seed_after_verify_device(
        self,
        *,
        model_worker_batch: ModelWorkerBatch,
        verified_tokens: jax.Array,
        target_hidden: jax.Array,
        accept_lengths: jax.Array,
        allocate_lens,
    ) -> None:
        """Select and relay the next Frozen seed without host acceptance reads.

        ``EagleVerifyInput.sample_device`` returns a padded candidate layout.
        The scheduler only needs its small acceptance result on host, whereas
        Frozen's next proposal needs the selected target token/hidden pair on
        device.  Select and publish that pair before the caller waits for the
        scheduler result.  Request IDs and bucket selection remain scheduler
        metadata; no model-specific admission policy enters this path.
        """
        if getattr(self, "seed_relay_buffers", None) is None:
            return

        selector_host = np.asarray(
            getattr(model_worker_batch, "logits_indices_selector", ()), dtype=np.int32
        ).reshape(-1)
        if selector_host.size == 0:
            return
        selector = jnp.asarray(selector_host, dtype=jnp.int32)
        accept_padded = jnp.asarray(accept_lengths, dtype=jnp.int32).reshape(-1)
        if int(accept_padded.shape[0]) <= int(selector_host.max(initial=-1)):
            raise ValueError(
                "Frozen-KV device verify acceptance has fewer padded rows than its selector: "
                f"accept_rows={accept_padded.shape[0]}, selector={selector_host.tolist()}"
            )

        rows_per_request = self.speculative_num_steps + 1
        if int(verified_tokens.shape[0]) < int(accept_padded.shape[0]) * rows_per_request:
            raise ValueError(
                "Frozen-KV device verify tokens do not cover the padded candidate layout: "
                f"tokens={verified_tokens.shape[0]}, accept_rows={accept_padded.shape[0]}, "
                f"rows_per_request={rows_per_request}"
            )

        seq_lens = jnp.asarray(model_worker_batch.seq_lens, dtype=jnp.int32).reshape(-1)
        allocate = jnp.asarray(allocate_lens, dtype=jnp.int32).reshape(-1)
        req_indices = jnp.asarray(model_worker_batch.req_pool_indices, dtype=jnp.int32).reshape(-1)

        def compact_request_rows(value, field):
            """Normalize compact or padded scheduler metadata to live rows.

            ``_get_cur_allocate_lens`` already compacts ``allocate_lens`` on
            the host, whereas seq lengths, request-pool indices, and verifier
            acceptance retain the DP-padded scheduler layout.  Treat both
            representations as an explicit contract instead of indexing a
            compact vector with global padded slots.
            """
            if int(value.shape[0]) == int(selector.shape[0]):
                return value
            max_slot = int(selector_host.max(initial=-1))
            if int(value.shape[0]) <= max_slot:
                raise ValueError(
                    "Frozen-KV device verify metadata does not match its live selector: "
                    f"field={field}, rows={value.shape[0]}, "
                    f"requests={selector.shape[0]}, selector={selector_host.tolist()}"
                )

            # The padded verifier can be TP-sharded by candidate rows, while
            # the compact live request count may be c1/c2 and therefore cannot
            # use a data-partitioned result. Annotate only this small metadata
            # gather; never force its source into a replicated layout first.
            value_sharding = jax.typeof(value).sharding
            if isinstance(value_sharding, NamedSharding):
                # The compact result is ordered by live request, not by equal
                # DP-attention partitions. Even a divisible row count can
                # represent an unbalanced batch, so keep this small metadata
                # gather replicated. The relay publisher scatters these rows
                # into the scheduler's explicitly selected padded slots.
                output_spec = P(*([None] * value.ndim))
                return value.at[selector].get(
                    out_sharding=NamedSharding(value_sharding.mesh, output_spec)
                )
            return jnp.take(value, selector, axis=0)

        accept_live = compact_request_rows(accept_padded, "accept_lengths")
        seed_state = select_after_verify(
            jnp.asarray(verified_tokens, dtype=jnp.int32),
            jnp.asarray(target_hidden),
            selector,
            accept_live,
            compact_request_rows(seq_lens, "seq_lens") + accept_live + 1,
            compact_request_rows(allocate, "allocate_lens"),
            compact_request_rows(req_indices, "req_pool_indices"),
            rows_per_request=rows_per_request,
        )
        self._publish_seed_relay(
            model_worker_batch=model_worker_batch,
            verified_id=seed_state.bonus_token,
            draft_token_ids=seed_state.bonus_token,
            hidden_states=seed_state.target_hidden,
            is_target_seed=seed_state.valid_mask,
        )

    def build_next_draft_input_after_verify(
        self,
        *,
        verified_id,
        hidden_states,
        new_seq_lens,
        allocate_lens,
        accept_lens,
        accept_index,
        model_worker_batch,
    ) -> FrozenKvMtpDraftInput:
        """Publish one target-hidden/token seed per request after verify.

        The common verifier has already made ``verified_id`` and
        ``hidden_states`` request-row aligned using its safe acceptance index.
        Frozen keeps that pair as a device-resident ``FrozenKvMtpSeedState``;
        the next ``draft`` consumes it before entering the recurrent proposal
        loop. No generic EAGLE extension result is returned as the seed state.
        """
        accept_padded = np.asarray(accept_lens, dtype=np.int32).reshape(-1)
        if accept_padded.size == 0:
            return self.new_draft_input(
                verified_id=np.empty(0, dtype=np.int32),
                hidden_states=jnp.empty((0, 0), dtype=jnp.float32),
                topk_p=np.empty((0, 1), dtype=np.float32),
                topk_index=np.empty((0, 1), dtype=np.int32),
                accept_length=accept_padded,
                accept_length_cpu=accept_padded.copy(),
                allocate_lens=np.empty(0, dtype=np.int32),
                new_seq_lens=np.empty(0, dtype=np.int32),
                seed_state=None,
            )

        selector = np.asarray(
            getattr(model_worker_batch, "logits_indices_selector", np.arange(accept_padded.size)),
            dtype=np.int32,
        ).reshape(-1)
        # ``EagleVerifyInput.sample`` returns one accept length for every
        # *padded* verifier slot.  The Frozen seed state instead has one row
        # per live scheduler request, identified by ``selector``.  Earlier we
        # assumed those shapes were already compact, which crashed the first
        # real Seed-MTP request at c1: selector=(1,), accept_lens=(16,).
        accept = self._compact_request_rows(accept_padded, selector, selector.size)
        if selector.size != accept.size:
            raise ValueError(
                "Frozen-KV verify selector must have one slot per accepted request: "
                f"selector={selector.shape}, accept_lens={accept.shape}"
            )

        verified = jnp.asarray(verified_id)
        hidden = jnp.asarray(hidden_states)
        accept_index = np.asarray(accept_index).reshape(-1)
        if accept_index.size % accept_padded.size != 0:
            raise ValueError(
                "Frozen-KV accept_index must contain a fixed candidate width per request: "
                f"rows={accept_index.size}, padded_requests={accept_padded.size}"
            )
        rows_per_request = accept_index.size // accept_padded.size
        if verified.shape[0] < accept_padded.size * rows_per_request:
            raise ValueError(
                "Frozen-KV verified rows are smaller than the padded verifier layout: "
                f"rows={verified.shape[0]}, padded_requests={accept_padded.size}, "
                f"rows_per_request={rows_per_request}"
            )
        committed = self._compact_request_rows(new_seq_lens, selector, accept.size)
        allocated = self._compact_request_rows(allocate_lens, selector, accept.size)
        req_indices = np.asarray(
            getattr(model_worker_batch, "req_pool_indices", np.arange(accept.size)),
            dtype=np.int32,
        ).reshape(-1)
        req_indices = self._compact_request_rows(req_indices, selector, accept.size)
        seed_state = select_after_verify(
            verified,
            hidden,
            jnp.asarray(selector),
            jnp.asarray(accept),
            jnp.asarray(committed),
            jnp.asarray(allocated),
            jnp.asarray(req_indices),
            rows_per_request=rows_per_request,
        )
        if getattr(self, "seed_relay_buffers", None) is not None:
            # Publish the target-selected seed in request-pool storage, then
            # return only scheduler metadata.  In particular, do not return
            # ``seed_state`` here: split/merge/filter would otherwise
            # materialize its variable-size hidden tensor before the next
            # static decode bucket is known.
            self._publish_seed_relay(
                model_worker_batch=model_worker_batch,
                verified_id=seed_state.bonus_token,
                draft_token_ids=seed_state.bonus_token,
                hidden_states=seed_state.target_hidden,
                is_target_seed=seed_state.valid_mask,
            )
            return self.new_draft_input(
                future_indices=req_indices,
                allocate_lens=allocated,
                new_seq_lens=committed,
                accept_length_cpu=accept.copy(),
                relay_seed_mask=np.ones((accept.size,), dtype=bool),
            )
        # These fields remain the scheduler-compatible representation. The
        # dedicated draft consumes seed_state and replaces top-k values after
        # its seed forward; they are not interpreted as a generic EAGLE
        # DRAFT_EXTEND result.
        return self.new_draft_input(
            verified_id=seed_state.bonus_token,
            hidden_states=seed_state.target_hidden,
            topk_p=jnp.ones((accept.size, 1), dtype=jnp.float32),
            topk_index=seed_state.bonus_token[:, None],
            accept_length=jnp.asarray(accept),
            accept_length_cpu=accept.copy(),
            allocate_lens=allocated,
            new_seq_lens=committed,
            seed_state=seed_state,
        )

    def _consume_seed_state(self, seed_state: FrozenKvMtpSeedState) -> tuple[jax.Array, jax.Array]:
        """Consume and validate the next Frozen proposal's seed pair.

        The returned token/hidden tensors are the inputs to the dedicated seed
        forward. Keeping this tiny seam separate lets fake runners test the
        handoff without constructing a TPU model runner.
        """
        if not isinstance(seed_state, FrozenKvMtpSeedState):
            raise TypeError(f"expected FrozenKvMtpSeedState, got {type(seed_state).__name__}")
        if not bool(np.all(np.asarray(seed_state.valid_mask))):
            raise ValueError("Frozen-KV cannot start a proposal from an invalid seed row")
        return seed_state.bonus_token, seed_state.target_hidden

    def _prepare_seed_proposal(self, model_worker_batch: ModelWorkerBatch) -> jax.Array | None:
        """Expose the seed-row mask for the next assistant seed forward.

        The scheduler has already scattered ``seed_state`` into its DP-padded
        slots.  Earlier experimental code compacted those slots and installed
        target hidden states as if they were EAGLE hidden states.  That skips
        Gemma's required assistant seed forward and also destroys the DP slot
        layout.  Keep the padded layout intact: ``draft_forward`` will run one
        assistant forward for every bucket slot and use this mask to retain its
        result only for rows which came from target verification.
        """
        state = model_worker_batch.spec_info_padded
        if not isinstance(state, FrozenKvMtpDraftInput) or state.seed_state is None:
            return None
        seed_state = state.seed_state
        valid_mask = jnp.asarray(seed_state.valid_mask, dtype=bool)
        state_rows = int(state.verified_id.shape[0])
        if seed_state.batch_size != state_rows:
            raise ValueError(
                "Frozen-KV seed state and draft fields must have the same DP-padded slots: "
                f"seed_bs={seed_state.batch_size}, draft_bs={state_rows}."
            )

        # Keep the target pair in the ordinary per-slot input fields so it
        # survives merge/split/scatter implementations which do not know about
        # ``seed_state`` yet.  Do not touch top-k values: target hidden is the
        # *input* of the assistant seed forward, never generic EAGLE output.
        # Relay lookup is data-sharded, while inherited draft-input fields can
        # be replicated. `where` does not insert a collective implicitly, so
        # align its predicate with each consumer field explicitly.
        def _mask_for(reference):
            sharding = getattr(reference, "sharding", None)
            if not isinstance(sharding, NamedSharding):
                return valid_mask if sharding is None else jax.device_put(valid_mask, sharding)
            # A hidden-state reference is rank two/three, but the `where`
            # predicate remains rank one before broadcasting.  Preserve the
            # relevant leading partition axes without applying a rank-two
            # sharding annotation to a vector.
            mask_sharding = NamedSharding(
                sharding.mesh, P(*tuple(sharding.spec)[: valid_mask.ndim])
            )
            return jax.device_put(valid_mask, mask_sharding)

        token_mask = _mask_for(state.verified_id)
        hidden_mask = _mask_for(state.hidden_states)
        state.verified_id = jnp.where(
            token_mask,
            jnp.asarray(seed_state.bonus_token, dtype=jnp.int32),
            jnp.asarray(state.verified_id, dtype=jnp.int32),
        )
        state.hidden_states = jnp.where(
            hidden_mask[:, None],
            jnp.asarray(seed_state.target_hidden),
            jnp.asarray(state.hidden_states),
        )
        state.seed_state = None
        return token_mask

    def _restore_seed_relay(self, model_worker_batch: ModelWorkerBatch) -> None:
        """Restore opaque relay metadata into one scheduler-selected bucket.

        ``future_indices`` is intentionally consumed here, after
        ``ScheduleBatch`` has merged/filtered and DP-scattered request metadata.
        No dynamic model tensor participates in those operations.  The request
        pool is the stable key, so slot reordering cannot pair one request with
        another request's target hidden state.
        """
        state = model_worker_batch.spec_info_padded
        if not isinstance(state, FrozenKvMtpDraftInput) or state.future_indices is None:
            return
        if getattr(self, "seed_relay_buffers", None) is None:
            raise RuntimeError(
                "Frozen-KV received a relay descriptor without initialized seed relay buffers."
            )
        if state.relay_seed_mask is None:
            raise ValueError("Frozen-KV relay descriptor is missing relay_seed_mask.")

        future_indices = np.asarray(state.future_indices, dtype=np.int32)
        relay_seed_mask = np.asarray(state.relay_seed_mask, dtype=bool)
        if future_indices.shape != relay_seed_mask.shape:
            raise ValueError(
                "Frozen-KV relay future_indices and relay_seed_mask must have identical shapes: "
                f"indices={future_indices.shape}, mask={relay_seed_mask.shape}."
            )
        self._init_jit_seed_relay_ops()
        with jax.set_mesh(self.mesh):
            (
                verified_id,
                draft_token_ids,
                hidden_states,
                device_indices,
                device_relay_seed_mask,
            ) = self._jit_gather_seed_relay(
                self.seed_relay_buffers,
                future_indices,
                relay_seed_mask,
                dp_size=int(model_worker_batch.dp_size),
            )

        state.verified_id = verified_id
        state.topk_index = draft_token_ids[:, None]
        # A top-1 proposal's probability affects tree scores but not its only
        # candidate token.  The first recurrent model forward replaces it
        # immediately; use a device scalar rather than retaining a second
        # request-indexed float buffer solely for that degenerate tree shape.
        state.topk_p = jnp.ones_like(state.topk_index, dtype=jnp.float32)
        state.hidden_states = hidden_states
        state.future_indices = None
        state.relay_seed_mask = None

        # Always retain a device mask, including an all-false one.  Branching
        # on `np.any` would materialize the just-gathered device relay state on
        # the host; `_prepare_seed_proposal` naturally becomes a no-op when no
        # row needs a target-hidden assistant seed forward.
        state.seed_state = FrozenKvMtpSeedState(
            bonus_token=verified_id,
            target_hidden=hidden_states,
            committed_lens=jnp.asarray(state.new_seq_lens, dtype=jnp.int32),
            allocate_lens=jnp.asarray(state.allocate_lens, dtype=jnp.int32),
            request_indices=device_indices,
            valid_mask=device_relay_seed_mask,
        )
        # Do not call ``_validate_non_overlap_state`` here: its intentional
        # CPU-side structural checks would materialize the freshly gathered
        # device arrays. Descriptor validation happens before the scheduler
        # stores the state; shape checks above cover this reconstruction seam.

    @staticmethod
    def _pad_seed_valid_mask(valid_mask, *, padded_bs: int, dp_size: int) -> jax.Array:
        """Pad a scheduler-slot mask with the same DP layout as draft fields."""
        valid_mask = jnp.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] == padded_bs:
            return valid_mask
        if valid_mask.shape[0] > padded_bs or padded_bs % dp_size:
            raise ValueError(
                "Frozen-KV seed mask cannot be padded to the draft bucket: "
                f"mask={valid_mask.shape[0]}, padded_bs={padded_bs}, dp_size={dp_size}."
            )
        if dp_size == 1 or valid_mask.shape[0] % dp_size:
            return jnp.pad(valid_mask, (0, padded_bs - valid_mask.shape[0]))
        per_dp_real = valid_mask.shape[0] // dp_size
        per_dp_padded = padded_bs // dp_size
        return jnp.pad(
            valid_mask.reshape(dp_size, per_dp_real),
            ((0, 0), (0, per_dp_padded - per_dp_real)),
        ).reshape(-1)

    @staticmethod
    def _replace_seed_rows(
        *,
        valid_mask: jax.Array,
        old_topk_p: jax.Array,
        old_topk_index: jax.Array,
        old_hidden: jax.Array,
        seed_topk_p: jax.Array,
        seed_topk_index: jax.Array,
        seed_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Use seed-forward outputs without disturbing freshly-prefilled rows."""
        valid_mask = jnp.asarray(valid_mask, dtype=bool)

        def _reshard_like(value: jax.Array, reference: jax.Array) -> jax.Array:
            """Match the persistent draft state's layout before `where`.

            The seed model forward returns data-sharded output, whereas the
            existing generic draft state is intentionally replicated at this
            handoff.  JAX rejects `where` across those layouts (correctly: it
            would otherwise require an implicit collective).  The state layout
            is the consumer contract for the inherited recurrence, so make the
            one explicit reshard here.
            """
            sharding = jax.typeof(reference).sharding
            # CPU unit tests use a zero-axis NamedSharding, which is already
            # effectively replicated and is not a legal `reshard` target.
            if isinstance(sharding, NamedSharding) and sharding.mesh.axis_names:
                return jax.sharding.reshard(value, sharding)
            return value

        def _mask_like(reference: jax.Array) -> jax.Array:
            sharding = getattr(reference, "sharding", None)
            if not isinstance(sharding, NamedSharding):
                return valid_mask if sharding is None else jax.device_put(valid_mask, sharding)
            mask_sharding = NamedSharding(
                sharding.mesh, P(*tuple(sharding.spec)[: valid_mask.ndim])
            )
            return jax.device_put(valid_mask, mask_sharding)

        seed_topk_p = _reshard_like(seed_topk_p, old_topk_p)
        seed_topk_index = _reshard_like(seed_topk_index, old_topk_index)
        seed_hidden = _reshard_like(seed_hidden, old_hidden)
        return (
            jnp.where(_mask_like(old_topk_p)[:, None], seed_topk_p, old_topk_p),
            jnp.where(
                _mask_like(old_topk_index)[:, None],
                seed_topk_index,
                old_topk_index,
            ),
            jnp.where(_mask_like(old_hidden)[:, None], seed_hidden, old_hidden),
        )

    def _run_seed_forward(
        self,
        model_worker_batch: ModelWorkerBatch,
        *,
        target_hidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run Gemma's one-token target-hidden -> assistant-state transition.

        Target verification produced the final accepted token and its target
        hidden state.  Gemma's assistant must consume that pair before the
        ordinary EAGLE-shaped recurrence can select a first draft candidate.
        This is intentionally a DECODE-shaped assistant operation: it reads
        the target KV view and has no assistant-owned cache to extend.
        """
        if self.topk != 1:
            raise NotImplementedError(
                "Frozen-KV seed forward currently supports speculative_eagle_topk=1 only."
            )
        state = model_worker_batch.spec_info_padded
        assert isinstance(state, FrozenKvMtpDraftInput)
        bs = int(model_worker_batch.seq_lens.shape[0])
        if target_hidden.shape[0] != bs:
            raise ValueError(
                "Frozen-KV seed hidden state must match the padded draft batch: "
                f"hidden_bs={target_hidden.shape[0]}, draft_bs={bs}."
            )

        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch
        )
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            model_worker_batch, self.draft_model_runner.mesh
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.out_cache_loc = np.empty((1,))
        forward_batch.cache_loc = np.empty((1,))
        forward_batch.spec_info = EagleDraftInput(hidden_states=jnp.asarray(target_hidden))
        # ``device_array`` intentionally accepts host arrays and therefore
        # calls ``np.asarray`` internally. The accepted Frozen seed is already
        # device-resident; preserve it with an explicit reshard instead.
        input_sharding = NamedSharding(self.mesh, P())
        forward_batch.input_ids = jax.device_put(
            jnp.asarray(state.verified_id, dtype=jnp.int32), input_sharding
        )
        # The seed token is already committed in target KV. Its target hidden
        # therefore belongs to the final committed position, not the next
        # proposal position. The attention metadata still exposes the target
        # cache at the current committed sequence length.
        forward_batch.positions = device_array(
            np.asarray(model_worker_batch.seq_lens, dtype=np.int32) - 1,
            sharding=NamedSharding(self.mesh, P()),
        )
        forward_batch.bid = model_worker_batch.bid
        self.draft_model_runner.attn_backend.forward_metadata = metadata_per_step[0]
        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_metadata,
        )
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        if self.hot_token_ids is not None:
            topk_index = self._map_hot_token_ids(topk_index)
        return topk_p, topk_index, replicate_to_mesh(self.mesh, logits_output.hidden_states)

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        """Start Frozen proposals with an assistant seed forward when required."""
        valid_mask = getattr(model_worker_batch, "_frozen_kv_seed_valid_mask", None)
        if valid_mask is None:
            return super().draft_forward(model_worker_batch)

        state = model_worker_batch.spec_info_padded
        assert isinstance(state, FrozenKvMtpDraftInput)
        padded_mask = self._pad_seed_valid_mask(
            valid_mask,
            padded_bs=int(model_worker_batch.seq_lens.shape[0]),
            dp_size=int(model_worker_batch.dp_size),
        )
        seed_topk_p, seed_topk_index, seed_hidden = self._run_seed_forward(
            model_worker_batch,
            target_hidden=jnp.asarray(state.hidden_states),
        )
        state.topk_p, state.topk_index, state.hidden_states = self._replace_seed_rows(
            valid_mask=padded_mask,
            old_topk_p=jnp.asarray(state.topk_p),
            old_topk_index=jnp.asarray(state.topk_index),
            old_hidden=jnp.asarray(state.hidden_states),
            seed_topk_p=seed_topk_p,
            seed_topk_index=seed_topk_index,
            seed_hidden=seed_hidden,
        )
        return super().draft_forward(model_worker_batch)

    def draft_extend_for_prefill(self, model_worker_batch, hidden_states, next_token_ids) -> None:
        """Publish the prefill's committed length for a later non-overlap merge."""
        super().draft_extend_for_prefill(model_worker_batch, hidden_states, next_token_ids)
        draft_input = model_worker_batch.spec_info_padded
        assert isinstance(draft_input, FrozenKvMtpDraftInput)
        selector = np.asarray(model_worker_batch.logits_indices_selector)
        draft_input.new_seq_lens = np.asarray(model_worker_batch.seq_lens)[selector].copy()
        draft_input._validate_non_overlap_state()
        if getattr(self, "seed_relay_buffers", None) is not None:
            # The inherited prefill extension currently computes the assistant
            # proposal correctly, but leaves compact model tensors in
            # ``draft_input``. Publish those tensors immediately and keep only
            # scheduler metadata for non-overlap merge/filter. A later dedicated
            # prefill program can remove that inherited host capture as a
            # separate, measurable change; this slice removes the cross-round
            # materialization without changing prefill math.
            self._publish_seed_relay(
                model_worker_batch=model_worker_batch,
                verified_id=draft_input.verified_id,
                draft_token_ids=jnp.asarray(draft_input.topk_index)[:, 0],
                hidden_states=draft_input.hidden_states,
                is_target_seed=jnp.zeros((selector.size,), dtype=bool),
            )
            model_worker_batch.spec_info_padded = self.new_draft_input(
                future_indices=np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)[
                    selector
                ],
                allocate_lens=np.asarray(draft_input.allocate_lens),
                new_seq_lens=np.asarray(draft_input.new_seq_lens),
                relay_seed_mask=np.zeros((selector.size,), dtype=bool),
            )
            model_worker_batch.spec_info_padded._validate_non_overlap_state()

    def draft_extend_for_decode(self, model_worker_batch, batch_output) -> None:
        """Publish the already-selected seed; do not run generic DRAFT_EXTEND.

        ``BaseSpecWorker.verify`` has already called
        ``build_next_draft_input_after_verify``. Frozen's next draft consumes
        that state at the start of ``draft``; there is no assistant forward at
        this boundary and no host-visible generic extension result.
        """
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
        """Run one fused Frozen proposal dispatch when its contract is available."""
        if self._can_use_fused_draft(model_worker_batch):
            return self._draft_fused_linear_chain(model_worker_batch)

        # Fake runners, direct state tests, and future non-linear configurations
        # retain the exact legacy mechanics as an explicit correctness fallback.
        self._restore_seed_relay(model_worker_batch)
        valid_mask = self._prepare_seed_proposal(model_worker_batch)
        if valid_mask is None:
            return super().draft(model_worker_batch)

        # This marker is deliberately ephemeral. It describes the current
        # scheduler batch only; persisting it in ``FrozenKvMtpDraftInput``
        # would make a finished/retracted request's validity leak into a later
        # batch.
        model_worker_batch._frozen_kv_seed_valid_mask = valid_mask
        try:
            return super().draft(model_worker_batch)
        finally:
            delattr(model_worker_batch, "_frozen_kv_seed_valid_mask")

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
    ``EAGLEWorker`` just like ordinary multi-layer MTP.  Frozen-KV only opts
    into that shared driver's non-fused prefill branch because its query-only
    assistant cannot use the generic fused NEXTN prefill path.
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

        For the supported greedy linear chain, target forward, acceptance,
        accepted-row selection, and relay publication are one compiled
        executable. Other sampling/tree contracts retain the generic fallback
        rather than silently changing semantics.
        """
        if not self.server_args.disable_overlap_schedule:
            return super().verify(model_worker_batch, cur_allocate_lens)

        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
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
        # Frozen-KV MTP's public contract is a top-k-one chain.  Verify that
        # chain directly against the target's native output sharding, following
        # DFlash's target-verify design.  Generic EAGLE tree verification has
        # P()-replicated Pallas inputs, which used to force both target logits
        # and target hidden states through ``replicate_to_mesh`` here before
        # the Frozen-specific seed relay could consume them.
        #
        # Keep the generic route as a correctness fallback for a future
        # branching configuration: its retrieval graph is not a linear chain.
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
        if fused_verify_ready:
            return self._verify_fused_linear_chain(
                model_worker_batch,
                spec_info,
                cur_allocate_lens,
                forward_metadata,
            )

        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        if native_chain_verify:
            (
                predict_device,
                accept_lengths_device,
                accept_index_device,
            ) = verify_frozen_kv_mtp_chain_greedy(
                spec_info.draft_token,
                logits_output.next_token_logits,
                draft_token_num=spec_info.draft_token_num,
            )
            verified_tokens_device = predict_device
        else:
            logits_output.next_token_logits, logits_output.hidden_states = replicate_to_mesh(
                self.mesh, logits_output.next_token_logits, logits_output.hidden_states
            )
            (
                predict_device,
                verified_tokens_device,
                accept_lengths_device,
                accept_index_device,
            ) = spec_info.sample_device(
                model_worker_batch,
                logits_output,
                self.draft_worker.draft_model_runner.rngs,
                self.mesh,
            )

        # Select and store the next target token/hidden seed before the host
        # waits for acceptance.  This follows DFlash's target-verify ownership:
        # device state is advanced first, host scheduler metadata follows.
        self.draft_worker.publish_seed_after_verify_device(
            model_worker_batch=model_worker_batch,
            verified_tokens=verified_tokens_device,
            target_hidden=logits_output.hidden_states,
            accept_lengths=accept_lengths_device,
            allocate_lens=cur_allocate_lens,
        )

        accept_width = self.speculative_num_steps + 1
        draft_width = self.speculative_num_draft_tokens
        flat_accept_index = jnp.asarray(accept_index_device, dtype=jnp.int32).reshape(-1)
        request_ids = jnp.arange(flat_accept_index.shape[0], dtype=jnp.int32) // accept_width
        per_request_last = request_ids * draft_width + draft_width - 1
        safe_index = jnp.where(flat_accept_index >= 0, flat_accept_index, per_request_last)

        # Preserve the target program's data/tensor output layouts through the
        # scheduler-facing candidate-row gather. Plain advanced indexing cannot
        # infer a safe output sharding when rows are TP-partitioned. This is a
        # device gather, not a host materialization or a P() replication.
        def gather_rows(value, indices):
            value_sharding = jax.typeof(value).sharding
            if isinstance(value_sharding, NamedSharding):
                return value.at[indices].get(out_sharding=value_sharding)
            return value[indices]

        logits_output.next_token_logits = gather_rows(logits_output.next_token_logits, safe_index)
        logits_output.hidden_states = gather_rows(logits_output.hidden_states, safe_index)
        model_worker_batch.positions = gather_rows(model_worker_batch.positions, safe_index)

        # The scheduler needs the acceptance length, but not the candidate-row
        # acceptance index. Start that small transfer after the relay dispatch;
        # it can overlap later device work exactly as in DFlash.
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
        new_seq_lens = seq_lens[selector] + accept_live + 1
        next_draft_input = self.draft_worker.new_draft_input(
            future_indices=req_indices,
            allocate_lens=allocate_lens,
            new_seq_lens=new_seq_lens,
            accept_length_cpu=accept_live.copy(),
            relay_seed_mask=np.ones((selector.size,), dtype=bool),
        )
        next_draft_input._validate_non_overlap_state()
        model_worker_batch.spec_info_padded = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict_device,
            next_draft_input=next_draft_input,
            accept_lens=accept_padded,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )
