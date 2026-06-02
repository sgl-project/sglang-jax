"""Fused single-JIT for N-layer MTP draft extend decode.

Merges all N model forward calls into one @jax.jit, eliminating:
- Per-layer host dispatch overhead (~N jit calls → 1)
- Per-layer device_get for rotate_ids (device .at[].set for topk=1)
- Per-layer replicate_to_mesh boundary (with_sharding_constraint inside jit)

Constraints:
- topk=1 only (rotate_ids = .at[sel_pos].set)
- All MTP layers share same model class (same model_def / model_state_def)
"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)


class GreedyVerifyPostprocessOutput(NamedTuple):
    next_token_logits: jax.Array
    hidden_states: jax.Array
    positions: jax.Array
    new_seq_lens: jax.Array
    select_index: jax.Array
    verified_id: jax.Array
    accept_lens: jax.Array


class GreedyStep3DraftInputs(NamedTuple):
    hidden_states: jax.Array
    positions: jax.Array
    new_seq_lens: jax.Array
    select_index: jax.Array
    verified_id: jax.Array
    accept_lens: jax.Array
    sel_pos: jax.Array


def _greedy_step3_prepare_draft_inputs(
    hidden_states,
    positions,
    seq_lens,
    accept_index,
    accept_length,
    verified_id,
    *,
    speculative_num_steps,
    speculative_num_draft_tokens,
):
    accept_width = speculative_num_steps + 1
    req_ids = jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    select_index = (
        jnp.arange(accept_length.shape[0], dtype=jnp.int32) * accept_width + accept_length - 1
    )
    hidden_sharding = jax.typeof(hidden_states).sharding
    positions_sharding = jax.typeof(positions).sharding
    if isinstance(hidden_sharding, NamedSharding) and not hidden_sharding.mesh.empty:
        gathered_hidden = hidden_states.at[safe_index, :].get(out_sharding=hidden_sharding.spec)
    else:
        gathered_hidden = hidden_states[safe_index, :]
    if isinstance(positions_sharding, NamedSharding) and not positions_sharding.mesh.empty:
        gathered_positions = positions.at[safe_index].get(out_sharding=positions_sharding.spec)
    else:
        gathered_positions = positions[safe_index]
    return GreedyStep3DraftInputs(
        hidden_states=gathered_hidden,
        positions=gathered_positions,
        new_seq_lens=seq_lens + accept_length,
        select_index=select_index,
        verified_id=verified_id,
        accept_lens=accept_length,
        sel_pos=jnp.clip(accept_length - 1, 0, None).astype(jnp.int32),
    )


@partial(
    jax.jit,
    static_argnames=["speculative_num_steps", "speculative_num_draft_tokens"],
)
def _greedy_verify_postprocess_jit(
    next_token_logits,
    hidden_states,
    positions,
    seq_lens,
    accept_index,
    accept_length,
    verified_id,
    *,
    speculative_num_steps,
    speculative_num_draft_tokens,
):
    accept_width = speculative_num_steps + 1
    req_ids = jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    gathered_logits = next_token_logits[safe_index, :]
    gathered_hidden = hidden_states[safe_index, :]
    gathered_positions = positions[safe_index]
    new_seq_lens = seq_lens + accept_length
    select_index = (
        jnp.arange(accept_length.shape[0], dtype=jnp.int32) * accept_width + accept_length - 1
    )
    return GreedyVerifyPostprocessOutput(
        next_token_logits=gathered_logits,
        hidden_states=gathered_hidden,
        positions=gathered_positions,
        new_seq_lens=new_seq_lens,
        select_index=select_index,
        verified_id=verified_id,
        accept_lens=accept_length,
    )


def _device_rotate_input_ids(input_ids, ext_lens, sel_pos, new_tokens):
    """Device-side per-req rotate for topk=1, exact mirror of the host
    ``MultiLayerDraftWorker._rotate_ids``:

      seg[:-1] = seg[1:]      # left-shift, KEEP last column (= prev last token)
      seg[sel_pos] = new_token
      padding reqs (ext_lens == 0) are left untouched

    input_ids is the flat (bs * tokens_per_req,) buffer; every req occupies a
    fixed ``tokens_per_req`` segment (real reqs have extend_seq_lens ==
    tokens_per_req, padding reqs have 0). The last column is the captured-logit
    position, so it must be copy-last (NOT jnp.roll, which wraps the first token
    into the last column and diverges on partial accept).
    """
    bs = ext_lens.shape[0]
    tokens_per_req = input_ids.shape[0] // bs
    ids_2d = input_ids.reshape(bs, tokens_per_req)
    shifted_2d = jnp.concatenate([ids_2d[:, 1:], ids_2d[:, -1:]], axis=1)
    shifted_2d = shifted_2d.at[jnp.arange(bs), sel_pos].set(
        new_tokens,
        out_sharding=jax.typeof(shifted_2d).sharding,
    )
    pad_mask = (ext_lens == 0)[:, None]
    shifted_2d = jnp.where(pad_mask, ids_2d, shifted_2d)
    return shifted_2d.reshape(-1)


def _replicate_for_host_output(value, replicated_sharding):
    """Move final small scheduler outputs to replicated sharding."""
    return jax.sharding.reshard(value, replicated_sharding)


def _build_fused_draft_extend_jit(num_layers: int, topk: int):
    """Build the fused JIT. Called once, result cached on draft_worker."""
    assert topk == 1, "Fused draft extend only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["all_memory_pools"],
        static_argnames=["model_state_def", "num_layers"],
    )
    def fused_draft_extend(
        model_def,
        model_state_def,
        all_leaves,  # tuple of N tuples-of-arrays
        forward_batch,  # single ForwardBatch (shared across layers)
        all_memory_pools,  # tuple of N MemoryPools pytrees
        logits_metadata,  # LogitsMetadata pytree
        target_hidden,  # (tokens, hidden_dim) replicated
        sel_pos,  # (bs,) device array — rotate position
        *,
        num_layers,
    ):
        all_topk_p = []
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None
        input_ids = forward_batch.input_ids

        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[i])
            model = nnx.merge(model_def, state)

            # shared forward_batch: only hidden_states and input_ids change per layer;
            # model() must not mutate other fields (positions, attn metadata, etc.)
            forward_batch.spec_info.hidden_states = target_hidden
            forward_batch.input_ids = input_ids

            output, pool_updates, _, _ = model(forward_batch, all_memory_pools[i], logits_metadata)
            all_pool_updates.append(pool_updates)

            # Replicate logits + hidden to P() (all-gather)
            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else None
            if mesh is not None:
                rep_logits = jax.sharding.reshard(
                    output.next_token_logits, NamedSharding(mesh, P())
                )
                rep_hidden = jax.sharding.reshard(output.hidden_states, NamedSharding(mesh, P()))
            else:
                rep_logits = output.next_token_logits
                rep_hidden = output.hidden_states

            if i == 0:
                layer0_hidden = rep_hidden

            # topk (inline, not separate jit)
            topk_logits, topk_idx = jax.lax.top_k(rep_logits, topk)
            logsumexp = jax.nn.logsumexp(rep_logits, axis=-1, keepdims=True)
            topk_p = jnp.exp(topk_logits - logsumexp).astype(rep_logits.dtype)

            all_topk_p.append(topk_p)
            all_topk_index.append(topk_idx)

            # Device-side rotate for topk=1 (shared with offline equivalence
            # test; exact mirror of host _rotate_ids).
            if i < num_layers - 1:
                ext_lens = forward_batch.extend_seq_lens
                input_ids = _device_rotate_input_ids(input_ids, ext_lens, sel_pos, topk_idx[:, 0])

        # Force P() replicated sharding on outputs that must be cross-process
        # consistent. Without this, donate_argnames may let XLA alias output
        # buffers with donated P("data")-sharded pool buffers, silently making
        # np.asarray() return per-process-different values.
        stacked_p = jnp.stack(all_topk_p, axis=1)
        stacked_idx = jnp.stack(all_topk_index, axis=1)
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            layer0_hidden = jax.lax.with_sharding_constraint(layer0_hidden, rep)
            stacked_p = jax.lax.with_sharding_constraint(stacked_p, rep)
            stacked_idx = jax.lax.with_sharding_constraint(stacked_idx, rep)

        return (
            layer0_hidden,
            stacked_p,
            stacked_idx,
            tuple(all_pool_updates),
        )

    return fused_draft_extend


def _build_fused_greedy_step3_draft_extend_jit(num_layers: int, topk: int):
    """Build Step3 fused JIT: greedy verify postprocess + N-layer draft extend."""
    assert topk == 1, "Fused greedy Step3 draft extend only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["all_memory_pools"],
        static_argnames=[
            "model_state_def",
            "num_layers",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
        ],
    )
    def fused_greedy_step3_draft_extend(
        model_def,
        model_state_def,
        all_leaves,
        forward_batch,
        all_memory_pools,
        logits_metadata,
        target_hidden,
        accept_index,
        accept_lens,
        verified_id,
        *,
        num_layers,
        speculative_num_steps,
        speculative_num_draft_tokens,
    ):
        prepared = _greedy_step3_prepare_draft_inputs(
            target_hidden,
            forward_batch.positions,
            forward_batch.seq_lens,
            accept_index,
            accept_lens,
            verified_id,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        all_topk_p = []
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        input_ids = prepared.verified_id
        mesh = None

        forward_batch.positions = prepared.positions
        forward_batch.spec_info.hidden_states = prepared.hidden_states
        forward_batch.spec_info.accept_length = prepared.accept_lens
        logits_metadata.accept_lens = prepared.accept_lens

        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[i])
            model = nnx.merge(model_def, state)

            forward_batch.spec_info.hidden_states = prepared.hidden_states
            forward_batch.input_ids = input_ids

            output, pool_updates, _, _ = model(forward_batch, all_memory_pools[i], logits_metadata)
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else None
            if mesh is not None:
                rep_logits = jax.sharding.reshard(
                    output.next_token_logits, NamedSharding(mesh, P())
                )
                rep_hidden = jax.sharding.reshard(output.hidden_states, NamedSharding(mesh, P()))
            else:
                rep_logits = output.next_token_logits
                rep_hidden = output.hidden_states

            if i == 0:
                layer0_hidden = rep_hidden

            topk_logits, topk_idx = jax.lax.top_k(rep_logits, topk)
            logsumexp = jax.nn.logsumexp(rep_logits, axis=-1, keepdims=True)
            topk_p = jnp.exp(topk_logits - logsumexp)

            all_topk_p.append(topk_p)
            all_topk_index.append(topk_idx)

            if i < num_layers - 1:
                input_ids = _device_rotate_input_ids(
                    input_ids,
                    forward_batch.extend_seq_lens,
                    prepared.sel_pos,
                    topk_idx[:, 0],
                )

        stacked_p = jnp.stack(all_topk_p, axis=1)
        stacked_idx = jnp.stack(all_topk_index, axis=1)
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            layer0_hidden = _replicate_for_host_output(layer0_hidden, rep)
            stacked_p = _replicate_for_host_output(stacked_p, rep)
            stacked_idx = _replicate_for_host_output(stacked_idx, rep)
            prepared_select_index = _replicate_for_host_output(prepared.select_index, rep)
            prepared_verified_id = _replicate_for_host_output(prepared.verified_id, rep)
            prepared_accept_lens = _replicate_for_host_output(prepared.accept_lens, rep)
            prepared_new_seq_lens = _replicate_for_host_output(prepared.new_seq_lens, rep)
        else:
            prepared_select_index = prepared.select_index
            prepared_verified_id = prepared.verified_id
            prepared_accept_lens = prepared.accept_lens
            prepared_new_seq_lens = prepared.new_seq_lens

        return (
            layer0_hidden,
            stacked_p,
            stacked_idx,
            tuple(all_pool_updates),
            prepared_select_index,
            prepared_verified_id,
            prepared_accept_lens,
            prepared_new_seq_lens,
        )

    return fused_greedy_step3_draft_extend


def _device_array_preserve_device(value, sharding):
    from sgl_jax.srt.utils.jax_utils import device_array

    if value is None:
        return None
    if isinstance(value, jax.Array):
        return jax.device_put(value, sharding)
    return device_array(value, sharding=sharding)


def _logits_metadata_from_model_worker_batch_preserve_device(batch, mesh):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    sharding = NamedSharding(mesh, P("data"))
    spec_info = batch.spec_info_padded
    accept_lens = (
        getattr(spec_info, "accept_length", None)
        if batch.forward_mode.is_draft_extend() and spec_info is not None
        else None
    )
    return LogitsMetadata(
        forward_mode=batch.forward_mode,
        capture_hidden_mode=batch.capture_hidden_mode,
        extend_return_logprob=False,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        extend_seq_lens=_device_array_preserve_device(batch.extend_seq_lens, sharding),
        logits_indices=_device_array_preserve_device(batch.logits_indices, sharding),
        accept_lens=_device_array_preserve_device(accept_lens, sharding),
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_logprob_pruned_lens_cpu=None,
        top_logprobs_nums=getattr(batch, "top_logprobs_nums", None),
        token_ids_logprobs=getattr(batch, "token_ids_logprobs", None),
        extend_input_logprob_token_ids_device=_device_array_preserve_device(
            getattr(batch, "extend_input_logprob_token_ids", None), sharding
        ),
    )


def _forward_batch_init_new_preserve_device(batch, model_runner):
    from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    data_sharding = NamedSharding(model_runner.mesh, P("data"))
    replicated_2d = NamedSharding(model_runner.mesh, P(None, None))

    input_embedding = _device_array_preserve_device(batch.input_embedding, replicated_2d)
    if input_embedding is not None:
        input_embedding = input_embedding.astype(jnp.bfloat16)

    deepstack_visual_embedding = None
    if getattr(batch, "apply_for_deepstack", False):
        deepstack_visual_embedding = _device_array_preserve_device(
            batch.deepstack_visual_embedding, replicated_2d
        )
        if deepstack_visual_embedding is not None:
            deepstack_visual_embedding = deepstack_visual_embedding.astype(jnp.bfloat16)

    if batch.lora_scalings is not None:
        lora_scalings = _device_array_preserve_device(batch.lora_scalings, data_sharding)
        lora_token_indices = _device_array_preserve_device(batch.lora_token_indices, data_sharding)
        lora_ranks = _device_array_preserve_device(batch.lora_ranks, data_sharding)
    else:
        lora_scalings = batch.lora_scalings
        lora_token_indices = batch.lora_token_indices
        lora_ranks = batch.lora_ranks

    return ForwardBatch(
        bid=batch.bid,
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=_device_array_preserve_device(batch.input_ids, data_sharding),
        seq_lens=_device_array_preserve_device(batch.seq_lens, data_sharding),
        out_cache_loc=_device_array_preserve_device(batch.out_cache_loc, data_sharding),
        positions=_device_array_preserve_device(batch.positions, data_sharding),
        mrope_positions=_device_array_preserve_device(batch.mrope_positions, replicated_2d),
        req_pool_indices=_device_array_preserve_device(batch.req_pool_indices, data_sharding),
        cache_loc=_device_array_preserve_device(batch.cache_loc, data_sharding),
        extend_prefix_lens=_device_array_preserve_device(batch.extend_prefix_lens, data_sharding),
        extend_seq_lens=_device_array_preserve_device(batch.extend_seq_lens, data_sharding),
        lora_ids=batch.lora_ids,
        lora_scalings=lora_scalings,
        lora_token_indices=lora_token_indices,
        lora_ranks=lora_ranks,
        attn_backend=model_runner.attn_backend,
        spec_info=batch.spec_info_padded,
        spec_algorithm=batch.spec_algorithm,
        capture_hidden_mode=batch.capture_hidden_mode,
        input_embedding=input_embedding,
        apply_for_deepstack=batch.apply_for_deepstack,
        deepstack_visual_embedding=deepstack_visual_embedding,
        expert_location_metadata=get_global_expert_location_metadata(),
        recurrent_indices=_device_array_preserve_device(batch.recurrent_indices, data_sharding),
    )


def draft_extend_for_decode_fused(draft_worker, model_worker_batch, batch_output):
    """Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.

    Fuses all N MTP layer forwards into a single jit call.
    """
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return
    target_hidden = batch_output.logits_output.hidden_states

    draft_input = EagleDraftInput(
        hidden_states=target_hidden, allocate_lens=batch_output.allocate_lens
    )
    mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_worker.draft_model_runner,
        batch_output,
        draft_worker.speculative_num_draft_tokens,
    )
    if mwb.input_ids.shape[0] <= 0:
        return

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    accept_host = np.asarray(jax.device_get(batch_output.accept_lens))
    assert (accept_host[sel] >= 1).all(), f"accept_length < 1: {accept_host[sel]}"
    sel_pos = np.clip(accept_host - 1, 0, None).astype(np.int32)

    # --- Host preparation (all done before single jit call) ---

    # Build ONE ForwardBatch and reuse for all layers. All workers share the
    # same mwb so forward_metadata is identical; only memory_pools/weights differ.
    mr0 = draft_worker._workers[0].model_runner
    mwb.spec_info_padded.hidden_states = target_hidden
    mr0.attn_backend.forward_metadata = mr0.attn_backend.get_eagle_forward_metadata(mwb)
    shared_fb = ForwardBatch.init_new(mwb, mr0)
    shared_fb.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    sel_pos_device = jax.device_put(sel_pos, NamedSharding(draft_worker.mesh, P("data")))

    # --- Build / get cached fused jit ---
    if not hasattr(draft_worker, "_fused_jit_fn"):
        draft_worker._fused_jit_fn = _build_fused_draft_extend_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    # --- Single jit call: all N layers fused ---
    # Model internals use P() with implicit mesh; need use_mesh context
    try:
        ctx = jax.sharding.use_mesh(draft_worker.mesh)
    except AttributeError:
        try:
            ctx = jax.set_mesh(draft_worker.mesh)
        except AttributeError:
            ctx = draft_worker.mesh
    with ctx:
        layer0_hidden, topk_p_stacked, topk_index_stacked, all_pool_updates = (
            draft_worker._fused_jit_fn(
                mr0._model_def,
                mr0._model_state_def,
                tuple(all_leaves),
                shared_fb,
                tuple(all_memory_pools),
                logits_metadata,
                target_hidden,
                sel_pos_device,
                num_layers=draft_worker.speculative_num_steps,
            )
        )

    # --- Host post-processing: replace memory pools + assemble outputs ---
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    select_index = sel * (draft_worker.speculative_num_steps + 1) + accept_host[sel] - 1
    verified_id_arr = batch_output.next_draft_input.verified_id
    if hasattr(verified_id_arr, "copy_to_host_async"):
        jax.copy_to_host_async(verified_id_arr)

    jax.copy_to_host_async(layer0_hidden)
    jax.copy_to_host_async(topk_p_stacked)
    jax.copy_to_host_async(topk_index_stacked)

    batch_output.next_draft_input.hidden_states = np.asarray(layer0_hidden)[select_index]
    batch_output.next_draft_input.topk_p = np.asarray(topk_p_stacked)[sel]
    batch_output.next_draft_input.topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.verified_id = np.asarray(verified_id_arr)[select_index]
    batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
    batch_output.accept_lens = accept_host


def _prepare_step3_model_worker_batch_for_draft_extend(
    draft_worker, model_worker_batch, batch_output
):
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    model_worker_batch.spec_info_padded = batch_output.next_draft_input
    sel = model_worker_batch.logits_indices_selector
    model_worker_batch.seq_lens[sel] = (
        model_worker_batch.seq_lens[sel] + draft_worker.speculative_num_draft_tokens - 1
    )

    bs = batch_output.accept_lens.shape[0]
    step_plus_1 = batch_output.next_draft_input.verified_id.shape[0] // bs
    model_worker_batch.extend_seq_lens = np.zeros((bs,), dtype=np.int32)
    model_worker_batch.extend_seq_lens[sel] = step_plus_1

    dp = model_worker_batch.dp_size
    per_dp = model_worker_batch.per_dp_bs_size if dp > 1 else bs
    model_worker_batch.logits_indices = (
        model_worker_batch.extend_seq_lens.reshape(dp, per_dp)
        .cumsum(axis=1, dtype=np.int32)
        .ravel()
        - 1
    )
    model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    model_worker_batch.spec_info_padded.capture_hidden_mode = CaptureHiddenMode.FULL
    model_worker_batch.forward_mode = ForwardMode.DRAFT_EXTEND
    model_worker_batch.spec_info_padded.hidden_states = batch_output.logits_output.hidden_states
    model_worker_batch.spec_info_padded.accept_length = batch_output.accept_lens
    model_worker_batch.input_ids = batch_output.next_draft_input.verified_id

    forward_metadata = draft_worker.draft_model_runner.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )
    draft_worker.draft_model_runner.attn_backend.forward_metadata = forward_metadata
    logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, draft_worker.mesh
    )
    return model_worker_batch, logits_metadata


def _draft_extend_for_decode_fused_step3_impl(draft_worker, model_worker_batch, batch_output):
    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return

    mwb, logits_metadata = _prepare_step3_model_worker_batch_for_draft_extend(
        draft_worker, model_worker_batch, batch_output
    )
    if mwb.input_ids.shape[0] <= 0:
        return

    mr0 = draft_worker._workers[0].model_runner
    mr0.attn_backend.forward_metadata = mr0.attn_backend.get_eagle_forward_metadata(mwb)
    shared_fb = _forward_batch_init_new_preserve_device(mwb, mr0)
    shared_fb.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_greedy_step3_jit_fn"):
        draft_worker._fused_greedy_step3_jit_fn = _build_fused_greedy_step3_draft_extend_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    try:
        ctx = jax.sharding.use_mesh(draft_worker.mesh)
    except AttributeError:
        try:
            ctx = jax.set_mesh(draft_worker.mesh)
        except AttributeError:
            ctx = draft_worker.mesh
    with ctx:
        (
            layer0_hidden,
            topk_p_stacked,
            topk_index_stacked,
            all_pool_updates,
            select_index_device,
            verified_id_device,
            accept_lens_device,
            new_seq_lens_device,
        ) = draft_worker._fused_greedy_step3_jit_fn(
            mr0._model_def,
            mr0._model_state_def,
            tuple(all_leaves),
            shared_fb,
            tuple(all_memory_pools),
            logits_metadata,
            batch_output.logits_output.hidden_states,
            batch_output.next_draft_input.accept_index,
            batch_output.accept_lens,
            batch_output.next_draft_input.verified_id,
            num_layers=draft_worker.speculative_num_steps,
            speculative_num_steps=draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
        )

    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    if hasattr(verified_id_device, "copy_to_host_async"):
        jax.copy_to_host_async(verified_id_device)
    jax.copy_to_host_async(layer0_hidden)
    jax.copy_to_host_async(topk_p_stacked)
    jax.copy_to_host_async(topk_index_stacked)
    jax.copy_to_host_async(select_index_device)
    jax.copy_to_host_async(accept_lens_device)
    jax.copy_to_host_async(new_seq_lens_device)

    select_index = np.asarray(select_index_device)[sel]
    batch_output.next_draft_input.hidden_states = np.asarray(layer0_hidden)[select_index]
    batch_output.next_draft_input.topk_p = np.asarray(topk_p_stacked)[sel]
    batch_output.next_draft_input.topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.verified_id = np.asarray(verified_id_device)[select_index]
    batch_output.next_draft_input.new_seq_lens = np.asarray(new_seq_lens_device)[sel]
    batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
    batch_output.accept_lens = np.asarray(accept_lens_device)


def draft_extend_for_decode_fused_step3(draft_worker, model_worker_batch, batch_output):
    """Greedy fixed-shape path for verify postprocess plus fused MTP extend.

    The target verify forward still runs outside this function. This path keeps
    greedy sample outputs on device and fuses the safe-index gather into the
    MTP draft-extend JIT.
    """
    return _draft_extend_for_decode_fused_step3_impl(draft_worker, model_worker_batch, batch_output)
