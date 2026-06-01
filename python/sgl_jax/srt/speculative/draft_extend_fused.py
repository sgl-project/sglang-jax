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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)


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
        all_forward_batches,  # tuple of N ForwardBatch pytrees (each with its own attn_backend)
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
        input_ids = all_forward_batches[0].input_ids

        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[i])
            model = nnx.merge(model_def, state)

            fb = all_forward_batches[i]
            fb.spec_info.hidden_states = target_hidden
            fb.input_ids = input_ids

            output, pool_updates, _, _ = model(fb, all_memory_pools[i], logits_metadata)
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
            topk_p = jnp.exp(topk_logits - logsumexp)

            all_topk_p.append(topk_p)
            all_topk_index.append(topk_idx)

            # Device-side rotate for topk=1 (shared with offline equivalence
            # test; exact mirror of host _rotate_ids).
            if i < num_layers - 1:
                ext_lens = all_forward_batches[0].extend_seq_lens
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
    sel_pos = np.clip(accept_host - 1, 0, None).astype(np.int64)
    mwb.input_ids = np.asarray(jax.device_get(mwb.input_ids)).copy()

    # --- Host preparation (all done before single jit call) ---

    # Pre-compute per-layer: forward_batch (with correct attn_backend), metadata, memory_pools, weights
    all_forward_batches = []
    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        mwb.spec_info_padded.hidden_states = target_hidden
        mr.attn_backend.forward_metadata = mr.attn_backend.get_eagle_forward_metadata(mwb)
        fb = ForwardBatch.init_new(mwb, mr)
        fb.bid = model_worker_batch.bid
        all_forward_batches.append(fb)
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    sel_pos_device = jax.device_put(sel_pos, NamedSharding(draft_worker.mesh, P("data")))

    # --- Build / get cached fused jit ---
    if not hasattr(draft_worker, "_fused_jit_fn"):
        draft_worker._fused_jit_fn = _build_fused_draft_extend_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    mr0 = draft_worker._workers[0].model_runner

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
                tuple(all_forward_batches),
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
