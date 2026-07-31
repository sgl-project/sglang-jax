"""Fused topk=1 EAGLE/EAGLE3/NEXTN verify and draft extend."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.speculative.kernel import top_k_renorm_prob, top_p_renorm_prob
from sgl_jax.srt.layers.attention.flashattention_metadata import (
    build_draft_extend_metadata,
    build_draft_forward_metadata,
    build_target_verify_metadata,
)
from sgl_jax.srt.sampling.sampling_params import TOP_K_ALL
from sgl_jax.srt.speculative.relay_buffer import (
    gather_spec_relay_buffers,
    make_dp_valid_mask,
    update_spec_relay_buffers,
)
from sgl_jax.srt.speculative.spec_utils import (
    SIMULATED_ACCEPTANCE_CONFIG,
    apply_simulated_acceptance,
    greedy_chain_verify,
)


class GreedyDraftInputs(NamedTuple):
    hidden_states: jax.Array
    positions: jax.Array
    new_seq_lens: jax.Array
    select_index: jax.Array
    verified_id: jax.Array
    accept_lens: jax.Array
    sel_pos: jax.Array


class GreedySampleAndPrepareOutput(NamedTuple):
    hidden_states: jax.Array
    positions: jax.Array
    new_seq_lens: jax.Array
    select_index: jax.Array
    safe_index: jax.Array
    verified_id: jax.Array
    accept_lens: jax.Array
    sel_pos: jax.Array
    predict: jax.Array


class FusedDraftExtendPendingResult(NamedTuple):
    batch_output: object
    selected_layer0_hidden: object
    topk_index_stacked: object
    next_verified_id: object
    accept_lens: object
    sel: np.ndarray
    updated_relay_buffers: object | None
    host_outputs_prefetched: bool = False


@contextmanager
def _count_pjit_cpp_cache_miss():
    try:
        import jax._src.test_util as jtu
    except (ImportError, ModuleNotFoundError):
        yield lambda: 0
        return
    with jtu.count_pjit_cpp_cache_miss() as count:
        yield count


def _active_dp_slot_mask(batch, total_bs: int) -> np.ndarray:
    mask = np.zeros(total_bs, dtype=bool)
    per_dp_bs = int(getattr(batch, "per_dp_bs_size", total_bs))
    real_bs_per_dp = getattr(batch, "real_bs_per_dp", None)
    if real_bs_per_dp is None:
        mask[: int(getattr(batch, "real_bs", total_bs))] = True
        return mask
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        start = dp_rank * per_dp_bs
        mask[start : start + int(real_bs)] = True
    return mask


def _prepare_rejection_sampling(sampling_info, batch, total_bs: int, vocab_size: int):
    temperatures = np.asarray(sampling_info.temperatures, dtype=np.float32).reshape(total_bs, 1)
    top_ks_src = getattr(sampling_info, "top_ks", None)
    top_ps_src = getattr(sampling_info, "top_ps", None)
    top_ks = (
        np.asarray(top_ks_src, dtype=np.int32).reshape(total_bs)
        if top_ks_src is not None
        else np.full(total_bs, TOP_K_ALL, dtype=np.int32)
    )
    top_ps = (
        np.asarray(top_ps_src, dtype=np.float32).reshape(total_bs)
        if top_ps_src is not None
        else np.ones(total_bs, dtype=np.float32)
    )

    active = _active_dp_slot_mask(batch, total_bs)
    temperatures = temperatures.copy()
    top_ks = top_ks.copy()
    top_ps = top_ps.copy()
    temperatures[~active] = 1.0
    top_ks[~active] = TOP_K_ALL
    top_ks[top_ks <= 0] = TOP_K_ALL
    top_ps[~active] = 1.0

    active_top_ks = top_ks[active]
    active_top_ps = top_ps[active]
    enable_top_k = bool(np.any((active_top_ks > 0) & (active_top_ks < vocab_size)))
    enable_top_p = bool(np.any(active_top_ps < 1.0))
    return temperatures, top_ks, top_ps, enable_top_k, enable_top_p


def _prepare_spec_prefill_output_token_ids(draft_worker, next_token_ids):
    if draft_worker.mesh is None:
        return next_token_ids
    if not hasattr(draft_worker, "_spec_prefill_output_gather_fn"):
        replicated_sharding = NamedSharding(draft_worker.mesh, P())
        draft_worker._spec_prefill_output_gather_fn = jax.jit(
            lambda x: x,
            out_shardings=replicated_sharding,
        )
    return draft_worker._spec_prefill_output_gather_fn(next_token_ids)


def _take_with_index_sharding(values, index):
    index_sharding = jax.typeof(index).sharding
    if isinstance(index_sharding, NamedSharding):
        return values.reshape(-1).at[index].get(out_sharding=index_sharding)
    return jnp.take(values.reshape(-1), index)


def _prepare_draft_inputs(
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
    req_ids = (
        jnp.zeros_like(accept_index)
        + jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    )
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    safe_accept_length = jnp.clip(accept_length, 1, None)
    select_index = (
        jnp.arange(accept_length.shape[0], dtype=jnp.int32) * accept_width + safe_accept_length - 1
    )
    hidden_sharding = jax.typeof(hidden_states).sharding
    positions_sharding = jax.typeof(positions).sharding
    if isinstance(hidden_sharding, NamedSharding):
        gathered_hidden = hidden_states.at[safe_index, :].get(out_sharding=hidden_sharding)
    else:
        gathered_hidden = hidden_states[safe_index, :]
    if isinstance(positions_sharding, NamedSharding):
        gathered_positions = positions.at[safe_index].get(out_sharding=positions_sharding)
    else:
        gathered_positions = positions[safe_index]
    new_seq_lens = seq_lens + accept_length + 1
    if SIMULATED_ACCEPTANCE_CONFIG.enabled:
        new_seq_lens = jnp.where(seq_lens > 0, new_seq_lens, 0)
    return GreedyDraftInputs(
        hidden_states=gathered_hidden,
        positions=gathered_positions,
        new_seq_lens=new_seq_lens,
        select_index=select_index,
        verified_id=verified_id,
        accept_lens=accept_length,
        sel_pos=jnp.clip(accept_length - 1, 0, None).astype(jnp.int32),
    )


def _verify_greedy(
    *,
    target_hidden,
    positions,
    seq_lens,
    draft_tokens,
    target_logits,
    speculative_num_steps,
    speculative_num_draft_tokens,
    simulation_rng=None,
):
    bs = seq_lens.shape[0]
    n = speculative_num_draft_tokens
    width = speculative_num_steps + 1
    if width != n:
        raise ValueError(
            "Greedy linear verify requires speculative_num_draft_tokens "
            f"({n}) == speculative_num_steps + 1 ({width})."
        )

    verify_result = greedy_chain_verify(
        draft_tokens,
        target_logits,
        draft_width=speculative_num_draft_tokens,
        valid_mask=seq_lens > 0,
    )
    is_padding = seq_lens == 0
    accepted_children = verify_result.accepted_children
    accept_length_raw = verify_result.accepted_draft_lens
    accept_length = verify_result.accept_lens

    row_ids = jnp.zeros_like(accept_length_raw) + jnp.arange(bs, dtype=jnp.int32)
    base = row_ids[:, None] * n
    child_offsets = jnp.arange(1, width, dtype=jnp.int32)[None, :]
    accept_index_children = jnp.where(accepted_children, base + child_offsets, -1)
    accept_index_2d = jnp.concatenate([base, accept_index_children], axis=1)
    accept_index_2d = jnp.where(is_padding[:, None], -1, accept_index_2d)
    draft_2d = draft_tokens.reshape(bs, n)
    target_predict_2d = verify_result.target_predict.reshape(bs, n)
    predict = verify_result.target_predict
    accept_index_2d, predict, accept_length = apply_simulated_acceptance(
        accept_index=accept_index_2d,
        predict=predict,
        accept_lens=accept_length,
        candidates=draft_2d,
        target_predict=target_predict_2d,
        valid_mask=~is_padding,
        spec_steps=speculative_num_steps,
        topk=1,
        rng=simulation_rng,
    )
    accept_index = accept_index_2d.reshape(-1)
    accept_width = speculative_num_steps + 1
    req_ids = (
        jnp.zeros_like(accept_index)
        + jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    )
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    safe_predict = _take_with_index_sharding(predict, safe_index)
    verified_id = jnp.where(accept_index >= 0, safe_predict, jnp.zeros_like(safe_predict))
    prepared = _prepare_draft_inputs(
        target_hidden,
        positions,
        seq_lens,
        accept_index,
        accept_length,
        verified_id,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
    )
    return GreedySampleAndPrepareOutput(
        hidden_states=prepared.hidden_states,
        positions=prepared.positions,
        new_seq_lens=prepared.new_seq_lens,
        select_index=prepared.select_index,
        safe_index=safe_index,
        verified_id=prepared.verified_id,
        accept_lens=prepared.accept_lens,
        sel_pos=prepared.sel_pos,
        predict=predict,
    )


def _verify_rejection_sampling(
    *,
    target_hidden,
    positions,
    seq_lens,
    draft_tokens,
    target_logits,
    temperatures,
    top_ks,
    top_ps,
    coins,
    coin_f,
    threshold_single,
    threshold_acc,
    enable_top_k,
    enable_top_p,
    speculative_num_steps,
    speculative_num_draft_tokens,
    simulation_rng=None,
):
    """Non-greedy counterpart of the greedy chain verify.

    Implements target-only typical acceptance for the pure topk=1 chain.
    Accepted slots emit the
    accepted draft token; the first rejected slot samples from the residual
    target distribution, while the all-accepted bonus slot samples from the
    full target distribution.
    """
    bs = seq_lens.shape[0]
    n = speculative_num_draft_tokens
    width = speculative_num_steps + 1
    vocab = target_logits.shape[-1]

    # v1: replicate the working set so explicit-sharding never has to resolve
    # gather/cumsum shardings. Correctness over speed for now.
    sh = jax.typeof(target_logits).sharding
    mesh = sh.mesh if isinstance(sh, NamedSharding) else None

    def _rep(x):
        return jax.sharding.reshard(x, NamedSharding(mesh, P())) if mesh is not None else x

    tl = _rep(target_logits.astype(jnp.float32))
    draft_2d = _rep(draft_tokens.reshape(bs, n).astype(jnp.int32))
    seq_lens_r = _rep(seq_lens.astype(jnp.int32))
    temp = _rep(temperatures.reshape(bs, 1).astype(jnp.float32))
    coins_r = _rep(coins.astype(jnp.float32))
    coin_f_r = _rep(coin_f.astype(jnp.float32))

    # target probs: temperature scale, then optional top_k/top_p renorm.
    # Everything is replicated here so the renorm kernels see consistent
    # full-vocabulary inputs.
    probs_3d = jax.nn.softmax(tl.reshape(bs, n, vocab) / temp[:, :, None], axis=-1)
    probs_2d = probs_3d.reshape(bs * n, vocab)
    if enable_top_k:
        tk = _rep(top_ks.astype(jnp.int32))
        tk_flat = jnp.broadcast_to(tk[:, None], (bs, n)).reshape(bs * n)
        probs_2d = top_k_renorm_prob(probs_2d, tk_flat)
    if enable_top_p:
        tp = _rep(top_ps.astype(jnp.float32))
        tp_flat = jnp.broadcast_to(tp[:, None], (bs, n)).reshape(bs * n)
        probs_2d = top_p_renorm_prob(probs_2d, tp_flat)
    probs_3d = probs_2d.reshape(bs, n, vocab)

    cand = draft_2d[:, 1:]  # (bs, n-1) candidate tokens d1..d_{n-1}
    p_cand = jnp.take_along_axis(probs_3d[:, : n - 1, :], cand[:, :, None], axis=-1)[:, :, 0]

    accept_mask = (coins_r <= p_cand / threshold_acc) | (p_cand >= threshold_single)

    is_padding = seq_lens_r == 0
    accepted_children = jnp.cumprod(accept_mask.astype(jnp.int32), axis=1).astype(jnp.bool_)
    accepted_children = jnp.where(is_padding[:, None], False, accepted_children)
    accept_length_raw = jnp.sum(accepted_children.astype(jnp.int32), axis=1)
    accept_length = jnp.where(is_padding, 0, accept_length_raw + 1)

    # residual / bonus sampling at emit position = accept_length_raw
    emit_pos = accept_length_raw.astype(jnp.int32)  # (bs,) in [0, n-1]
    p_emit = jnp.take_along_axis(probs_3d, emit_pos[:, None, None], axis=1)[:, 0, :]  # (bs, vocab)
    has_rejected_child = emit_pos < (n - 1)
    safe_reject_pos = jnp.minimum(emit_pos, n - 2)
    rejected_token = jnp.take_along_axis(cand, safe_reject_pos[:, None], axis=1)[:, 0]
    vocab_ids = jnp.arange(vocab, dtype=jnp.int32)[None, :]
    residual_probs = jnp.where(vocab_ids == rejected_token[:, None], 0.0, p_emit)
    final_probs = jnp.where(has_rejected_child[:, None], residual_probs, p_emit)
    cdf = jnp.cumsum(final_probs, axis=-1)
    u = coin_f_r * cdf[:, -1]
    sampled = jnp.sum((cdf <= u[:, None]).astype(jnp.int32), axis=-1).astype(jnp.int32)
    sampled = jnp.minimum(sampled, jnp.int32(vocab - 1))  # (bs,)

    # predict_2d[:, k] = cand[:, k] (=d_{k+1}); override emit_pos slot with sampled
    predict_2d = jnp.concatenate([cand, jnp.zeros((bs, 1), dtype=jnp.int32)], axis=1).astype(
        jnp.int32
    )
    predict_2d = predict_2d.at[jnp.arange(bs), emit_pos].set(sampled)
    predict = predict_2d.reshape(-1)

    # --- accept_index machinery (identical to greedy path) ---
    row_ids = jnp.arange(bs, dtype=jnp.int32)
    base = row_ids[:, None] * n
    child_offsets = jnp.arange(1, width, dtype=jnp.int32)[None, :]
    accept_index_children = jnp.where(accepted_children, base + child_offsets, -1)
    accept_index_2d = jnp.concatenate([base, accept_index_children], axis=1)
    accept_index_2d = jnp.where(is_padding[:, None], -1, accept_index_2d)
    target_predict_2d = jnp.argmax(tl, axis=-1).astype(jnp.int32).reshape(bs, n)
    accept_index_2d, predict, accept_length = apply_simulated_acceptance(
        accept_index=accept_index_2d,
        predict=predict,
        accept_lens=accept_length,
        candidates=draft_2d,
        target_predict=target_predict_2d,
        valid_mask=~is_padding,
        spec_steps=speculative_num_steps,
        topk=1,
        rng=simulation_rng,
    )
    accept_index = accept_index_2d.reshape(-1)

    accept_width = speculative_num_steps + 1
    req_ids = jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    safe_predict = _take_with_index_sharding(predict, safe_index)
    verified_id = jnp.where(accept_index >= 0, safe_predict, jnp.zeros_like(safe_predict))
    prepared = _prepare_draft_inputs(
        target_hidden,
        positions,
        seq_lens,
        accept_index,
        accept_length,
        verified_id,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
    )
    return GreedySampleAndPrepareOutput(
        hidden_states=prepared.hidden_states,
        positions=prepared.positions,
        new_seq_lens=prepared.new_seq_lens,
        select_index=prepared.select_index,
        safe_index=safe_index,
        verified_id=prepared.verified_id,
        accept_lens=prepared.accept_lens,
        sel_pos=prepared.sel_pos,
        predict=predict,
    )


def _build_chain_verify_arrays(
    *,
    verified_id,
    token_list,
    seq_lens,
    num_verify_tokens,
    batch_size,
):
    """Build the token and position arrays for fused topk=1 verification."""
    n = num_verify_tokens
    bs = batch_size
    tid_range = jnp.arange(n, dtype=jnp.int32)
    verified_column = verified_id.astype(jnp.int32)[:, None]
    token_chain = token_list[:, : n - 1].astype(jnp.int32)
    verified_sharding = jax.typeof(verified_column).sharding
    if (
        isinstance(verified_sharding, NamedSharding)
        and not verified_sharding.mesh.empty
        and jax.typeof(token_chain).sharding != verified_sharding
    ):
        token_chain = jax.sharding.reshard(token_chain, verified_sharding)
    draft_tokens = jnp.concatenate([verified_column, token_chain], axis=1).reshape(bs * n)
    positions = (seq_lens.astype(jnp.int32)[:, None] + tid_range[None, :]).reshape(bs * n)
    return draft_tokens, positions


def _rotate_mtp_decode_input_ids(input_ids, extend_seq_lens, selected_positions, new_tokens):
    """Shift fixed-width decode rows and append the previous MTP layer token."""
    batch_size = extend_seq_lens.shape[0]
    tokens_per_request = input_ids.shape[0] // batch_size
    input_rows = input_ids.reshape(batch_size, tokens_per_request)
    shifted_rows = jnp.concatenate([input_rows[:, 1:], input_rows[:, -1:]], axis=1)
    shifted_rows = shifted_rows.at[jnp.arange(batch_size), selected_positions].set(
        new_tokens,
        out_sharding=jax.typeof(shifted_rows).sharding,
    )
    shifted_rows = jnp.where(
        (extend_seq_lens == 0)[:, None],
        input_rows,
        shifted_rows,
    )
    return shifted_rows.reshape(input_ids.shape)


def _rotate_mtp_prefill_input_ids(
    input_ids,
    extend_seq_lens,
    new_tokens,
    dp_size: int,
    per_dp_bs: int,
):
    """Shift packed prefill segments and append the previous MTP layer token."""
    per_dp_tokens = input_ids.shape[0] // dp_size
    input_rows = input_ids.reshape(dp_size, per_dp_tokens)
    extend_rows = extend_seq_lens.reshape(dp_size, per_dp_bs)
    token_rows = new_tokens.reshape(dp_size, per_dp_bs)
    token_offsets = jnp.arange(per_dp_tokens, dtype=jnp.int32)

    def rotate_rank(input_row, extend_row, token_row):
        starts = jnp.cumsum(extend_row, axis=0) - extend_row
        ends = starts + extend_row
        in_request = (token_offsets[None, :] >= starts[:, None]) & (
            token_offsets[None, :] < ends[:, None]
        )
        has_request = jnp.any(in_request, axis=0)
        slot = jnp.argmax(in_request.astype(jnp.int32), axis=0)
        request_starts = starts.at[slot].get()
        request_lens = extend_row.at[slot].get()
        request_tokens = token_row.at[slot].get()
        shifted_index = jnp.minimum(token_offsets + 1, per_dp_tokens - 1)
        shifted = input_row.at[shifted_index].get()
        is_last = has_request & ((token_offsets - request_starts) == (request_lens - 1))
        rotated = jnp.where(is_last, request_tokens, shifted)
        return jnp.where(has_request, rotated, input_row)

    return jax.vmap(rotate_rank)(input_rows, extend_rows, token_rows).reshape(input_ids.shape)


def _topk1_index_from_logits(logits):
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _gather_rows_preserve_sharding(values, index):
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding):
        return values.at[index, :].get(out_sharding=sharding)
    return values[index, :]


def _reshard_values(sharding, *values):
    return tuple(jax.sharding.reshard(value, sharding) for value in values)


def _eagle3_raw_and_mapped_token_from_logits(logits, hot_token_ids):
    raw_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if hot_token_ids is None:
        return raw_token, raw_token
    return raw_token, _map_eagle3_token_ids(raw_token, hot_token_ids)


def _map_eagle3_token_ids(token_ids, hot_token_ids):
    """Map draft-vocabulary ids while preserving the token array sharding."""
    out_sharding = jax.typeof(token_ids).sharding
    if isinstance(out_sharding, NamedSharding):
        return hot_token_ids.at[token_ids].get(out_sharding=out_sharding)
    return hot_token_ids[token_ids]


def _build_mtp_prefill_draft_extend(num_layers: int):
    """Build one fused prefix draft pass across independent NEXTN layers."""

    @partial(
        jax.jit,
        donate_argnames=["all_memory_pools"],
        static_argnames=["model_state_def", "num_layers", "dp_size"],
    )
    def fused_mtp_prefill_draft_extend(
        model_def,
        model_state_def,
        all_leaves,
        forward_batch,
        all_memory_pools,
        logits_metadata,
        target_hidden,
        draft_logits_indices,
        allocated_lens,
        *,
        num_layers,
        dp_size,
    ):
        page_layout = forward_batch.attn_backend.forward_metadata
        forward_batch.attn_backend.forward_metadata = build_draft_extend_metadata(
            page_layout,
            forward_batch.seq_lens,
            allocated_lens,
            query_lens=forward_batch.extend_seq_lens,
            page_size=forward_batch.attn_backend.page_size,
            dp_size=dp_size,
        )

        input_ids = forward_batch.input_ids
        per_dp_bs = forward_batch.seq_lens.shape[0] // dp_size
        layer0_hidden = None
        token_chain = []
        all_pool_updates = []

        for layer_idx in range(num_layers):
            state = jax.tree_util.tree_unflatten(
                model_state_def,
                all_leaves[layer_idx],
            )
            model = nnx.merge(model_def, state)
            forward_batch.input_ids = input_ids
            forward_batch.spec_info.hidden_states = target_hidden
            output, pool_updates, _, _ = model(
                forward_batch,
                all_memory_pools[layer_idx],
                logits_metadata,
            )
            all_pool_updates.append(pool_updates)
            if layer_idx == 0:
                layer0_hidden = output.hidden_states

            next_token = _topk1_index_from_logits(output.next_token_logits)
            token_chain.append(next_token)
            if layer_idx < num_layers - 1:
                input_ids = _rotate_mtp_prefill_input_ids(
                    input_ids,
                    forward_batch.extend_seq_lens,
                    next_token,
                    dp_size,
                    per_dp_bs,
                )

        selected_indices = draft_logits_indices
        if dp_size > 1:
            per_dp_tokens = layer0_hidden.shape[0] // dp_size
            rank_ids = jnp.arange(
                selected_indices.shape[0],
                dtype=jnp.int32,
            ) // per_dp_bs
            selected_indices = selected_indices + rank_ids * per_dp_tokens
        selected_hidden = _gather_rows_preserve_sharding(
            layer0_hidden,
            selected_indices,
        )
        stacked_tokens = jnp.stack(token_chain, axis=1)

        sharding = jax.typeof(stacked_tokens).sharding
        if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
            replicated = NamedSharding(sharding.mesh, P())
            selected_hidden = jax.sharding.reshard(selected_hidden, replicated)
            stacked_tokens = jax.sharding.reshard(stacked_tokens, replicated)

        return (
            selected_hidden,
            stacked_tokens,
            tuple(all_pool_updates),
        )

    return fused_mtp_prefill_draft_extend


def _build_mtp_draft_extend(num_layers: int):
    """Build fused NEXTN draft-extend across independent prediction layers."""

    @partial(
        jax.jit,
        donate_argnames=["all_memory_pools"],
        static_argnames=["model_state_def", "num_layers", "update_relay", "dp_size"],
    )
    def fused_mtp_draft_extend(
        model_def,
        model_state_def,
        all_leaves,
        forward_batch,
        all_memory_pools,
        logits_metadata,
        target_hidden,
        selected_positions,
        draft_logits_indices,
        draft_verify_seq_lens,
        draft_allocate_lens,
        next_verified_id,
        next_new_seq_lens,
        relay_buffers,
        relay_future_indices,
        relay_valid_mask,
        *,
        num_layers,
        update_relay,
        dp_size,
    ):
        page_layout = forward_batch.attn_backend.forward_metadata
        valid_slots = draft_verify_seq_lens > 0
        forward_batch.seq_lens = jnp.where(
            valid_slots,
            draft_verify_seq_lens + num_layers,
            jnp.zeros_like(draft_verify_seq_lens),
        )
        forward_batch.attn_backend.forward_metadata = build_draft_extend_metadata(
            page_layout,
            forward_batch.seq_lens,
            draft_allocate_lens,
            query_lens=forward_batch.extend_seq_lens,
            page_size=forward_batch.attn_backend.page_size,
            dp_size=dp_size,
        )

        input_ids = forward_batch.input_ids
        layer0_hidden = None
        token_chain = []
        all_pool_updates = []

        for layer_idx in range(num_layers):
            state = jax.tree_util.tree_unflatten(
                model_state_def,
                all_leaves[layer_idx],
            )
            model = nnx.merge(model_def, state)
            forward_batch.input_ids = input_ids
            forward_batch.spec_info.hidden_states = target_hidden
            output, pool_updates, _, _ = model(
                forward_batch,
                all_memory_pools[layer_idx],
                logits_metadata,
            )
            all_pool_updates.append(pool_updates)
            if layer_idx == 0:
                layer0_hidden = output.hidden_states

            next_token = _topk1_index_from_logits(output.next_token_logits)
            token_chain.append(next_token)
            if layer_idx < num_layers - 1:
                input_ids = _rotate_mtp_decode_input_ids(
                    input_ids,
                    forward_batch.extend_seq_lens,
                    selected_positions,
                    next_token,
                )

        selected_indices = draft_logits_indices
        if logits_metadata.accept_lens is not None:
            selected_indices = selected_indices - (
                forward_batch.extend_seq_lens - logits_metadata.accept_lens
            )
            selected_indices = jnp.where(
                forward_batch.extend_seq_lens > 0,
                selected_indices,
                0,
            )
        if dp_size > 1:
            per_dp_tokens = layer0_hidden.shape[0] // dp_size
            per_dp_bs = selected_indices.shape[0] // dp_size
            rank_ids = jnp.arange(
                selected_indices.shape[0],
                dtype=jnp.int32,
            ) // per_dp_bs
            selected_indices = selected_indices + rank_ids * per_dp_tokens
        selected_hidden = _gather_rows_preserve_sharding(
            layer0_hidden,
            selected_indices,
        )
        stacked_tokens = jnp.stack(token_chain, axis=1)

        updated_relay_buffers = relay_buffers
        if update_relay:
            updated_relay_buffers = update_spec_relay_buffers(
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                stacked_tokens,
                selected_hidden,
                next_verified_id,
                next_new_seq_lens,
                dp_size=dp_size,
            )
        else:
            sharding = jax.typeof(stacked_tokens).sharding
            if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
                replicated = NamedSharding(sharding.mesh, P())
                selected_hidden = jax.sharding.reshard(selected_hidden, replicated)
                stacked_tokens = jax.sharding.reshard(stacked_tokens, replicated)

        return (
            selected_hidden,
            stacked_tokens,
            tuple(all_pool_updates),
            updated_relay_buffers,
        )

    return fused_mtp_draft_extend


def _build_eagle3_prefill_draft_extend():
    """Build the fused EAGLE3 prefix draft-extend JIT."""

    @partial(
        jax.jit,
        donate_argnames=["memory_pools"],
        static_argnames=["model_state_def", "dp_size"],
    )
    def fused_eagle3_prefill_draft_extend(
        model_def,
        model_state_def,
        model_leaves,
        forward_batch,
        memory_pools,
        logits_metadata,
        allocated_lens,
        *,
        dp_size,
    ):
        state = jax.tree_util.tree_unflatten(model_state_def, model_leaves)
        model = nnx.merge(model_def, state)
        forward_batch.attn_backend.forward_metadata = build_draft_extend_metadata(
            forward_batch.attn_backend.forward_metadata,
            forward_batch.seq_lens,
            allocated_lens,
            query_lens=forward_batch.extend_seq_lens,
            page_size=forward_batch.attn_backend.page_size,
            dp_size=dp_size,
        )
        output, pool_updates, _, _ = model(
            forward_batch,
            memory_pools,
            logits_metadata,
        )
        return output, pool_updates

    return fused_eagle3_prefill_draft_extend


def _build_eagle3_bootstrap(num_steps: int):
    """Build a fused recurrent JIT that expands one seed into a topk=1 chain."""
    assert num_steps > 1, "EAGLE3 fused bootstrap requires num_steps > 1"

    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    @partial(
        jax.jit,
        donate_argnames=["memory_pools"],
        static_argnames=["model_state_def", "num_steps", "dp_size"],
    )
    def fused_eagle3_bootstrap(
        model_def,
        model_state_def,
        model_leaves,
        forward_batch,
        memory_pools,
        logits_metadata,
        initial_hidden,
        initial_raw_token,
        allocated_lens,
        hot_token_ids,
        *,
        num_steps,
        dp_size,
    ):
        state = jax.tree_util.tree_unflatten(model_state_def, model_leaves)
        model = nnx.merge(model_def, state)
        page_layout = forward_batch.attn_backend.forward_metadata
        base_seq_lens = forward_batch.seq_lens
        valid = base_seq_lens > 0

        raw_token = initial_raw_token
        token = (
            raw_token
            if hot_token_ids is None
            else _map_eagle3_token_ids(raw_token, hot_token_ids)
        )
        hidden = initial_hidden
        raw_tokens = [raw_token]
        pool_updates = None

        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.extend_prefix_lens = None
        forward_batch.extend_seq_lens = None
        logits_metadata.forward_mode = ForwardMode.DECODE
        logits_metadata.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_metadata.extend_seq_lens = None
        logits_metadata.accept_lens = None
        logits_metadata.logits_indices = None

        # The prefix draft forward already produced the first raw token. Each
        # recurrent call consumes one chain token and produces the next one;
        # the final token is intentionally not written to draft KV.
        for step in range(num_steps - 1):
            decode_seq_lens = jnp.where(
                valid,
                base_seq_lens + step,
                jnp.zeros_like(base_seq_lens),
            )
            forward_batch.input_ids = token
            forward_batch.positions = decode_seq_lens
            forward_batch.seq_lens = decode_seq_lens
            forward_batch.spec_info.hidden_states = hidden
            forward_batch.attn_backend.forward_metadata = build_draft_forward_metadata(
                page_layout,
                decode_seq_lens,
                allocated_lens,
                page_size=forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )

            output, pool_updates, _, _ = model(
                forward_batch,
                memory_pools,
                logits_metadata,
            )
            memory_pools.replace_all(pool_updates)
            raw_token, token = _eagle3_raw_and_mapped_token_from_logits(
                output.next_token_logits,
                hot_token_ids,
            )
            hidden = output.hidden_states
            raw_tokens.append(raw_token)

        return jnp.stack(raw_tokens, axis=1), pool_updates

    return fused_eagle3_bootstrap


def _build_eagle3_recurrent_draft_extend(num_steps: int):
    """Build EAGLE3 draft-extend followed by recurrent one-token draft steps."""

    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    @partial(
        jax.jit,
        donate_argnames=["memory_pools"],
        static_argnames=["model_state_def", "num_steps", "update_relay", "dp_size"],
    )
    def fused_eagle3_draft_extend(
        model_def,
        model_state_def,
        model_leaves,
        forward_batch,
        memory_pools,
        logits_metadata,
        target_hidden,
        draft_logits_indices,
        draft_verify_seq_lens,
        draft_allocate_lens,
        next_verified_id,
        next_new_seq_lens,
        hot_token_ids,
        relay_buffers,
        relay_future_indices,
        relay_valid_mask,
        *,
        num_steps,
        update_relay,
        dp_size,
    ):
        state = jax.tree_util.tree_unflatten(model_state_def, model_leaves)
        model = nnx.merge(model_def, state)
        page_layout = forward_batch.attn_backend.forward_metadata

        valid_draft_slots = draft_verify_seq_lens > 0
        forward_batch.seq_lens = jnp.where(
            valid_draft_slots,
            draft_verify_seq_lens + num_steps,
            jnp.zeros_like(draft_verify_seq_lens),
        )
        forward_batch.attn_backend.forward_metadata = build_draft_extend_metadata(
            page_layout,
            forward_batch.seq_lens,
            draft_allocate_lens,
            query_lens=forward_batch.extend_seq_lens,
            page_size=forward_batch.attn_backend.page_size,
            dp_size=dp_size,
        )
        forward_batch.spec_info.hidden_states = target_hidden

        output, pool_updates, _, _ = model(forward_batch, memory_pools, logits_metadata)
        memory_pools.replace_all(pool_updates)

        last_idx = draft_logits_indices
        if logits_metadata.accept_lens is not None:
            last_idx = last_idx - (forward_batch.extend_seq_lens - logits_metadata.accept_lens)
            last_idx = jnp.where(forward_batch.extend_seq_lens > 0, last_idx, 0)
        if dp_size > 1:
            per_dp_tokens = output.hidden_states.shape[0] // dp_size
            per_dp_bs = last_idx.shape[0] // dp_size
            rank_ids = jnp.arange(last_idx.shape[0], dtype=jnp.int32) // per_dp_bs
            last_idx = last_idx + rank_ids * per_dp_tokens
        selected_stage0_hidden = _gather_rows_preserve_sharding(
            output.hidden_states,
            last_idx,
        )

        raw_token, token = _eagle3_raw_and_mapped_token_from_logits(
            output.next_token_logits,
            hot_token_ids,
        )
        raw_tokens = [raw_token]
        hidden = selected_stage0_hidden

        # The first call above extends the accepted target tokens. Remaining
        # calls are true recurrent EAGLE3 decode steps: each consumes the
        # previous draft token/hidden and the KV pool updated by the prior step.
        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch.extend_prefix_lens = None
        forward_batch.extend_seq_lens = None
        logits_metadata.forward_mode = ForwardMode.DECODE
        logits_metadata.capture_hidden_mode = CaptureHiddenMode.LAST
        logits_metadata.extend_seq_lens = None
        logits_metadata.accept_lens = None
        logits_metadata.logits_indices = None

        for step in range(1, num_steps):
            decode_seq_lens = jnp.where(
                valid_draft_slots,
                next_new_seq_lens + step - 1,
                jnp.zeros_like(next_new_seq_lens),
            )
            forward_batch.input_ids = token
            forward_batch.positions = decode_seq_lens
            forward_batch.seq_lens = decode_seq_lens
            forward_batch.spec_info.hidden_states = hidden
            forward_batch.attn_backend.forward_metadata = build_draft_forward_metadata(
                page_layout,
                decode_seq_lens,
                draft_allocate_lens,
                page_size=forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )

            output, pool_updates, _, _ = model(
                forward_batch,
                memory_pools,
                logits_metadata,
            )
            memory_pools.replace_all(pool_updates)
            raw_token, token = _eagle3_raw_and_mapped_token_from_logits(
                output.next_token_logits,
                hot_token_ids,
            )
            hidden = output.hidden_states
            raw_tokens.append(raw_token)

        # Persist draft-vocabulary ids. padding_for_decode applies d2t once
        # when the chain is consumed, which also makes a width-1 bootstrap
        # downgrade safe when a new prefill request joins the running batch.
        stacked_tokens = jnp.stack(raw_tokens, axis=1)
        relay_hidden = selected_stage0_hidden
        relay_topk_index = stacked_tokens
        updated_relay_buffers = relay_buffers
        if update_relay:
            updated_relay_buffers = update_spec_relay_buffers(
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                relay_topk_index,
                relay_hidden,
                next_verified_id,
                next_new_seq_lens,
                dp_size=dp_size,
            )

        sharding = jax.typeof(stacked_tokens).sharding
        mesh = sharding.mesh if isinstance(sharding, NamedSharding) else None
        if mesh is not None and not update_relay:
            rep = NamedSharding(mesh, P())
            selected_stage0_hidden = jax.sharding.reshard(selected_stage0_hidden, rep)
            stacked_tokens = jax.sharding.reshard(stacked_tokens, rep)

        return (
            selected_stage0_hidden,
            stacked_tokens,
            pool_updates,
            updated_relay_buffers,
        )

    return fused_eagle3_draft_extend


def _build_verify():
    """Build target forward + linear-chain verify JIT."""

    @partial(
        jax.jit,
        donate_argnames=["target_memory_pools"],
        static_argnames=[
            "target_model_state_def",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "return_target_logits",
            "use_relay_state",
            "dp_size",
            "is_greedy",
            "threshold_single",
            "threshold_acc",
            "enable_top_k",
            "enable_top_p",
            "rebuild_verify_metadata",
        ],
    )
    def fused_verify(
        target_model_def,
        target_model_state_def,
        target_leaves,
        target_forward_batch,
        target_memory_pools,
        target_logits_metadata,
        previous_verified_id,
        previous_token_list,
        draft_to_target_token_ids,
        relay_buffers,
        relay_future_indices,
        verify_allocate_lens,
        sampling_base_rng,
        sampling_step,
        temperatures,
        top_ks,
        top_ps,
        *,
        speculative_num_steps,
        speculative_num_draft_tokens,
        return_target_logits,
        use_relay_state,
        dp_size,
        is_greedy=True,
        threshold_single=1.0,
        threshold_acc=1.0,
        enable_top_k=False,
        enable_top_p=False,
        rebuild_verify_metadata=False,
    ):
        if use_relay_state:
            relay_topk_index, _, relay_verified_id, relay_new_seq_lens = gather_spec_relay_buffers(
                relay_buffers,
                relay_future_indices,
                dp_size=dp_size,
            )
            valid_seq_lens = target_forward_batch.seq_lens > 0
            target_forward_batch.seq_lens = jnp.where(
                valid_seq_lens,
                relay_new_seq_lens - 1,
                jnp.zeros_like(target_forward_batch.seq_lens),
            )
            previous_verified_id = relay_verified_id
            previous_token_list = relay_topk_index

        if use_relay_state or rebuild_verify_metadata:
            target_forward_batch.attn_backend.forward_metadata = (
                build_target_verify_metadata(
                    target_forward_batch.attn_backend.forward_metadata,
                    target_forward_batch.seq_lens,
                    verify_allocate_lens,
                    draft_width=speculative_num_draft_tokens,
                    page_size=target_forward_batch.attn_backend.page_size,
                    dp_size=dp_size,
                )
            )

        if draft_to_target_token_ids is not None:
            previous_token_list = _map_eagle3_token_ids(
                previous_token_list,
                draft_to_target_token_ids,
            )

        target_bs = target_forward_batch.seq_lens.shape[0]
        draft_tokens, positions = _build_chain_verify_arrays(
            verified_id=previous_verified_id,
            token_list=previous_token_list,
            seq_lens=target_forward_batch.seq_lens,
            num_verify_tokens=speculative_num_draft_tokens,
            batch_size=target_bs,
        )

        target_forward_batch.input_ids = draft_tokens
        target_forward_batch.positions = positions
        target_forward_batch.spec_info.draft_token = draft_tokens
        target_forward_batch.spec_info.positions = positions

        target_state = jax.tree_util.tree_unflatten(target_model_state_def, target_leaves)
        target_model = nnx.merge(target_model_def, target_state)
        target_output, target_pool_updates, _, _ = target_model(
            target_forward_batch,
            target_memory_pools,
            target_logits_metadata,
        )

        sh = jax.typeof(target_output.next_token_logits).sharding
        mesh = sh.mesh if isinstance(sh, NamedSharding) else None
        target_logits = target_output.next_token_logits
        target_hidden = target_output.hidden_states
        sampling_rng = jax.random.fold_in(sampling_base_rng, sampling_step)
        simulation_rng = jax.random.fold_in(sampling_rng, 1)
        if is_greedy:
            prepared = _verify_greedy(
                target_hidden=target_hidden,
                positions=target_forward_batch.positions,
                seq_lens=target_forward_batch.seq_lens,
                draft_tokens=draft_tokens,
                target_logits=target_logits,
                simulation_rng=simulation_rng,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )
        else:
            # Generate rejection-sampling coins inside the JIT: avoids building them
            # on host and copying (tbs, n-1)+(tbs,) arrays in every step, and keeps
            # the threefry/uniform ops fused into this module instead of becoming
            # standalone eager dispatches (the reason the earlier host-side jax.random
            # attempt was reverted).
            coins_key, coin_f_key = jax.random.split(sampling_rng)
            coins = jax.random.uniform(
                coins_key,
                (target_bs, speculative_num_draft_tokens - 1),
                dtype=jnp.float32,
            )
            coin_f = jax.random.uniform(coin_f_key, (target_bs,), dtype=jnp.float32)
            prepared = _verify_rejection_sampling(
                target_hidden=target_hidden,
                positions=target_forward_batch.positions,
                seq_lens=target_forward_batch.seq_lens,
                draft_tokens=draft_tokens,
                target_logits=target_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                coins=coins,
                coin_f=coin_f,
                simulation_rng=simulation_rng,
                threshold_single=threshold_single,
                threshold_acc=threshold_acc,
                enable_top_k=enable_top_k,
                enable_top_p=enable_top_p,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )

        target_logits_for_host = (
            _gather_rows_preserve_sharding(target_logits, prepared.safe_index)
            if return_target_logits
            else None
        )
        prepared_hidden = prepared.hidden_states
        prepared_verified_id = prepared.verified_id
        prepared_verified_id_data = prepared.verified_id
        prepared_next_verified_id = _take_with_index_sharding(
            prepared.verified_id, prepared.select_index
        )
        prepared_new_seq_lens = prepared.new_seq_lens
        prepared_accept_lens_host = prepared.accept_lens
        prepared_accept_lens_data = prepared.accept_lens
        prepared_extend_seq_lens = jnp.where(
            target_forward_batch.seq_lens > 0,
            jnp.full_like(target_forward_batch.seq_lens, speculative_num_draft_tokens),
            jnp.zeros_like(target_forward_batch.seq_lens),
        ).astype(jnp.int32)
        prepared_logits_indices = (
            jnp.cumsum(
                prepared_extend_seq_lens.reshape(dp_size, target_bs // dp_size),
                axis=1,
            ).reshape(-1)
            - 1
        ).astype(jnp.int32)
        prepared_sel_pos = prepared.sel_pos
        prepared_sel_pos_data = prepared.sel_pos
        prepared_predict = prepared.predict
        prepared_positions = prepared.positions
        prepared_positions_data = prepared.positions
        prepared_verify_seq_lens = target_forward_batch.seq_lens
        prepared_allocate_lens_data = verify_allocate_lens

        if mesh is not None:
            rep = NamedSharding(mesh, P())
            data = NamedSharding(mesh, P("data"))
            (
                prepared_hidden,
                prepared_verified_id,
                prepared_new_seq_lens,
                prepared_accept_lens_host,
                prepared_sel_pos,
                prepared_predict,
                prepared_positions,
            ) = _reshard_values(
                rep,
                prepared_hidden,
                prepared_verified_id,
                prepared_new_seq_lens,
                prepared_accept_lens_host,
                prepared_sel_pos,
                prepared_predict,
                prepared_positions,
            )
            (
                prepared_verified_id_data,
                prepared_next_verified_id,
                prepared_accept_lens_data,
                prepared_extend_seq_lens,
                prepared_logits_indices,
                prepared_sel_pos_data,
                prepared_positions_data,
                prepared_allocate_lens_data,
            ) = _reshard_values(
                data,
                prepared_verified_id_data,
                prepared_next_verified_id,
                prepared_accept_lens_data,
                prepared_extend_seq_lens,
                prepared_logits_indices,
                prepared_sel_pos_data,
                prepared_positions_data,
                prepared_allocate_lens_data,
            )
            if return_target_logits:
                target_logits_for_host = jax.sharding.reshard(target_logits_for_host, rep)

        return (
            target_pool_updates,
            prepared_hidden,
            prepared_verified_id,
            prepared_verified_id_data,
            prepared_next_verified_id,
            prepared_new_seq_lens,
            prepared_accept_lens_host,
            prepared_accept_lens_data,
            prepared_extend_seq_lens,
            prepared_logits_indices,
            prepared_sel_pos,
            prepared_sel_pos_data,
            prepared_predict,
            prepared_positions,
            prepared_positions_data,
            prepared_verify_seq_lens,
            prepared_allocate_lens_data,
            target_logits_for_host,
        )

    return fused_verify


def _prepare_verify(
    draft_worker,
    model_worker_batch,
    *,
    draft_padding_prepared: bool = False,
):
    """Prepare fixed-shape verify placeholders while keeping chain build inside JIT."""
    from sgl_jax.srt.speculative.eagle_info import EagleVerifyInput

    draft_input = model_worker_batch.spec_info_padded
    use_relay_state = (
        getattr(draft_input, "future_indices", None) is not None
        and getattr(draft_input, "topk_index", None) is None
    )
    if use_relay_state:
        bs = len(model_worker_batch.seq_lens)
        draft_input.verified_id = np.zeros((bs,), dtype=np.int32)
        draft_input.topk_index = np.zeros(
            (bs, draft_worker.speculative_num_steps),
            dtype=np.int32,
        )
        draft_input.hidden_states = np.zeros(
            (bs, draft_worker.model_config.hidden_size),
            dtype=np.float32,
        )

    if not draft_padding_prepared:
        # Relay buffers keep recurrent EAGLE3 ids in draft-vocabulary space;
        # fused_verify gathers and maps that chain itself.  The host-side
        # placeholders above are never consumed, so mapping them here launches
        # an eager gather (and its broadcast) on every overlap round.
        draft_worker.padding_for_decode(model_worker_batch)
    draft_input = model_worker_batch.spec_info_padded
    previous_verified_id = draft_input.verified_id
    if isinstance(previous_verified_id, np.ndarray):
        previous_verified_id = np.asarray(previous_verified_id, dtype=np.int32)
    topk_index = draft_input.topk_index
    if len(topk_index.shape) != 2:
        raise ValueError(
            "Fused speculative token state must have shape (batch, num_steps); "
            f"got {topk_index.shape}."
        )
    previous_token_list = topk_index
    if isinstance(previous_token_list, np.ndarray):
        previous_token_list = np.asarray(previous_token_list, dtype=np.int32)
    elif previous_token_list.dtype != jnp.int32:
        previous_token_list = previous_token_list.astype(jnp.int32)

    bs = model_worker_batch.seq_lens.shape[0]
    n = draft_worker.speculative_num_draft_tokens
    flat = bs * n
    placeholder_cache = getattr(draft_worker, "_fused_verify_placeholder_cache", None)
    if placeholder_cache is None:
        placeholder_cache = draft_worker._fused_verify_placeholder_cache = {}
    placeholder_key = (bs, n)
    verify_input = placeholder_cache.get(placeholder_key)
    if verify_input is None:
        data_sharding = NamedSharding(draft_worker.mesh, P("data"))
        verify_input = EagleVerifyInput(
            draft_token=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
            positions=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
        )
        placeholder_cache[placeholder_key] = verify_input
    model_worker_batch.spec_info_padded = verify_input
    return previous_verified_id, previous_token_list


def _prepare_device_array(value, sharding, name: str | None = None):
    from sgl_jax.srt.utils.jax_utils import device_array

    if value is None:
        return None
    if isinstance(value, jax.Array):
        if value.sharding == sharding:
            return value
        return jax.device_put(value, sharding)
    return device_array(value, sharding=sharding)


def _prepare_logits_metadata(batch, mesh, *, include_accept_lens: bool = True):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

    if batch.forward_mode.is_target_verify():
        return LogitsMetadata(
            forward_mode=batch.forward_mode,
            capture_hidden_mode=batch.capture_hidden_mode,
        )

    sharding = NamedSharding(mesh, P("data"))
    spec_info = batch.spec_info_padded
    accept_lens = (
        getattr(spec_info, "accept_length", None)
        if include_accept_lens and batch.forward_mode.is_draft_extend() and spec_info is not None
        else None
    )
    return LogitsMetadata(
        forward_mode=batch.forward_mode,
        capture_hidden_mode=batch.capture_hidden_mode,
        extend_return_logprob=False,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        extend_seq_lens=_prepare_device_array(
            batch.extend_seq_lens, sharding, "logits.extend_seq_lens"
        ),
        logits_indices=_prepare_device_array(
            batch.logits_indices, sharding, "logits.logits_indices"
        ),
        accept_lens=_prepare_device_array(accept_lens, sharding, "logits.accept_lens"),
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_logprob_pruned_lens_cpu=None,
        top_logprobs_nums=getattr(batch, "top_logprobs_nums", None),
        token_ids_logprobs=getattr(batch, "token_ids_logprobs", None),
        extend_input_logprob_token_ids_device=_prepare_device_array(
            getattr(batch, "extend_input_logprob_token_ids", None),
            sharding,
            "logits.extend_input_logprob_token_ids",
        ),
    )


def _make_forward_batch(batch, model_runner):
    from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    data_sharding = NamedSharding(model_runner.mesh, P("data"))
    replicated_2d = NamedSharding(model_runner.mesh, P(None, None))
    spec_info = getattr(batch, "spec_info_padded", None)
    input_ids = (
        getattr(spec_info, "verified_id_for_draft_extend", None) if spec_info is not None else None
    )
    if input_ids is None:
        input_ids = batch.input_ids
    positions = (
        getattr(spec_info, "positions_for_draft_extend", None) if spec_info is not None else None
    )
    if positions is None:
        positions = batch.positions
    extend_seq_lens = (
        getattr(spec_info, "extend_seq_lens_for_draft_extend", None)
        if spec_info is not None
        else None
    )
    if extend_seq_lens is None:
        extend_seq_lens = batch.extend_seq_lens

    input_embedding = _prepare_device_array(
        batch.input_embedding, replicated_2d, "forward.input_embedding"
    )
    if input_embedding is not None:
        input_embedding = input_embedding.astype(jnp.bfloat16)

    deepstack_visual_embedding = None
    if getattr(batch, "apply_for_deepstack", False):
        deepstack_visual_embedding = _prepare_device_array(
            batch.deepstack_visual_embedding,
            replicated_2d,
            "forward.deepstack_visual_embedding",
        )
        if deepstack_visual_embedding is not None:
            deepstack_visual_embedding = deepstack_visual_embedding.astype(jnp.bfloat16)

    if batch.lora_scalings is not None:
        lora_scalings = _prepare_device_array(
            batch.lora_scalings, data_sharding, "forward.lora_scalings"
        )
        lora_token_indices = _prepare_device_array(
            batch.lora_token_indices, data_sharding, "forward.lora_token_indices"
        )
        lora_ranks = _prepare_device_array(batch.lora_ranks, data_sharding, "forward.lora_ranks")
    else:
        lora_scalings = batch.lora_scalings
        lora_token_indices = batch.lora_token_indices
        lora_ranks = batch.lora_ranks

    return ForwardBatch(
        bid=batch.bid,
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=_prepare_device_array(input_ids, data_sharding, "forward.input_ids"),
        seq_lens=_prepare_device_array(batch.seq_lens, data_sharding, "forward.seq_lens"),
        out_cache_loc=_prepare_device_array(
            batch.out_cache_loc, data_sharding, "forward.out_cache_loc"
        ),
        positions=_prepare_device_array(positions, data_sharding, "forward.positions"),
        mrope_positions=_prepare_device_array(
            batch.mrope_positions, replicated_2d, "forward.mrope_positions"
        ),
        req_pool_indices=_prepare_device_array(
            batch.req_pool_indices, data_sharding, "forward.req_pool_indices"
        ),
        cache_loc=_prepare_device_array(batch.cache_loc, data_sharding, "forward.cache_loc"),
        extend_prefix_lens=_prepare_device_array(
            batch.extend_prefix_lens, data_sharding, "forward.extend_prefix_lens"
        ),
        extend_seq_lens=_prepare_device_array(
            extend_seq_lens, data_sharding, "forward.extend_seq_lens"
        ),
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
        recurrent_indices=_prepare_device_array(
            batch.recurrent_indices, data_sharding, "forward.recurrent_indices"
        ),
    )


def mtp_prefill_draft_extend(draft_worker, model_worker_batch, target_hidden):
    """Run all NEXTN prefix layers in one fused topk=1 draft JIT."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    runner0 = draft_worker.draft_model_runner
    runner0.attn_backend.forward_metadata = runner0.attn_backend.prepare_paged_kv_layout(
        model_worker_batch
    )
    forward_batch = _make_forward_batch(model_worker_batch, runner0)
    forward_batch.forward_mode = ForwardMode.EXTEND
    logits_metadata = _prepare_logits_metadata(model_worker_batch, draft_worker.mesh)
    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    draft_logits_indices = _prepare_device_array(
        model_worker_batch.logits_indices,
        data_sharding,
        "mtp_prefill.logits_indices",
    )
    allocated_lens = _prepare_device_array(
        np.asarray(model_worker_batch.seq_lens, dtype=np.int32),
        data_sharding,
        "mtp_prefill.allocate_lens",
    )

    all_memory_pools = tuple(
        worker.model_runner.memory_pools for worker in draft_worker._workers
    )
    all_leaves = tuple(
        tuple(worker.model_runner.model_state_leaves)
        for worker in draft_worker._workers
    )
    if not hasattr(draft_worker, "_fused_mtp_prefill_draft_extend_jit_fn"):
        draft_worker._fused_mtp_prefill_draft_extend_jit_fn = (
            _build_mtp_prefill_draft_extend(draft_worker.speculative_num_steps)
        )

    with jax.set_mesh(draft_worker.mesh):
        selected_hidden, token_chain, all_pool_updates = (
            draft_worker._fused_mtp_prefill_draft_extend_jit_fn(
                runner0._model_def,
                runner0._model_state_def,
                all_leaves,
                forward_batch,
                all_memory_pools,
                logits_metadata,
                target_hidden,
                draft_logits_indices,
                allocated_lens,
                num_layers=draft_worker.speculative_num_steps,
                dp_size=model_worker_batch.dp_size,
            )
        )

    for layer_idx, worker in enumerate(draft_worker._workers):
        worker.model_runner.memory_pools.replace_all(all_pool_updates[layer_idx])

    jax.copy_to_host_async(selected_hidden)
    jax.copy_to_host_async(token_chain)
    selector = np.asarray(model_worker_batch.logits_indices_selector)
    return np.asarray(selected_hidden)[selector], np.asarray(token_chain)[selector]


def eagle_prefill_draft_extend(draft_worker, model_worker_batch):
    """Run an EAGLE/EAGLE3 prefix draft forward with device-built metadata."""
    runner = draft_worker.draft_model_runner
    runner.attn_backend.forward_metadata = runner.attn_backend.prepare_paged_kv_layout(
        model_worker_batch
    )
    forward_batch = _make_forward_batch(model_worker_batch, runner)
    # Preserve the existing EAGLE3 model behavior: attention treats this as an
    # extend, while logits metadata retains the speculative draft-extend mode.
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

    forward_batch.forward_mode = ForwardMode.EXTEND
    logits_metadata = _prepare_logits_metadata(model_worker_batch, draft_worker.mesh)
    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    allocated_lens = _prepare_device_array(
        np.asarray(model_worker_batch.seq_lens, dtype=np.int32),
        data_sharding,
        "eagle3_prefill.allocate_lens",
    )

    if not hasattr(draft_worker, "_fused_eagle3_prefill_draft_extend_jit_fn"):
        draft_worker._fused_eagle3_prefill_draft_extend_jit_fn = (
            _build_eagle3_prefill_draft_extend()
        )

    with jax.set_mesh(draft_worker.mesh):
        output, pool_updates = draft_worker._fused_eagle3_prefill_draft_extend_jit_fn(
            runner._model_def,
            runner._model_state_def,
            tuple(runner.model_state_leaves),
            forward_batch,
            runner.memory_pools,
            logits_metadata,
            allocated_lens,
            dp_size=model_worker_batch.dp_size,
        )
    runner.memory_pools.replace_all(pool_updates)
    return output, forward_batch


def bootstrap_eagle_chain(draft_worker, model_worker_batch):
    """Expand the first EAGLE/EAGLE3 seed into a fused topk=1 chain."""
    runner = draft_worker.draft_model_runner
    spec_info = model_worker_batch.spec_info_padded
    runner.attn_backend.forward_metadata = runner.attn_backend.prepare_paged_kv_layout(
        model_worker_batch
    )
    forward_batch = _make_forward_batch(model_worker_batch, runner)

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    initial_raw_token = spec_info.topk_index
    if initial_raw_token.ndim == 2:
        initial_raw_token = initial_raw_token[:, 0]
    initial_raw_token = _prepare_device_array(
        initial_raw_token,
        data_sharding,
        "eagle3_bootstrap.initial_raw_token",
    )
    initial_hidden = _prepare_device_array(
        spec_info.hidden_states,
        data_sharding,
        "eagle3_bootstrap.initial_hidden",
    )
    allocated_lens = _prepare_device_array(
        spec_info.allocate_lens,
        data_sharding,
        "eagle3_bootstrap.allocate_lens",
    )
    logits_metadata = _prepare_logits_metadata(model_worker_batch, draft_worker.mesh)

    if not hasattr(draft_worker, "_fused_eagle3_bootstrap_jit_fn"):
        draft_worker._fused_eagle3_bootstrap_jit_fn = _build_eagle3_bootstrap(
            draft_worker.speculative_num_steps
        )

    with jax.set_mesh(draft_worker.mesh):
        token_chain, pool_updates = draft_worker._fused_eagle3_bootstrap_jit_fn(
            runner._model_def,
            runner._model_state_def,
            tuple(runner.model_state_leaves),
            forward_batch,
            runner.memory_pools,
            logits_metadata,
            initial_hidden,
            initial_raw_token,
            allocated_lens,
            draft_worker.hot_token_ids,
            num_steps=draft_worker.speculative_num_steps,
            dp_size=model_worker_batch.dp_size,
        )
    runner.memory_pools.replace_all(pool_updates)
    return token_chain


def restore_draft_extend_result(draft_worker, model_worker_batch, pending_result):
    if pending_result is None:
        return

    batch_output = pending_result.batch_output
    selected_layer0_hidden = pending_result.selected_layer0_hidden
    topk_index_stacked = pending_result.topk_index_stacked
    next_verified_id = pending_result.next_verified_id
    accept_host = np.asarray(jax.device_get(pending_result.accept_lens))
    sel = pending_result.sel

    jax.copy_to_host_async(selected_layer0_hidden)
    jax.copy_to_host_async(topk_index_stacked)
    if model_worker_batch.dp_size > 1:
        from jax.experimental.multihost_utils import process_allgather

        next_verified_id = process_allgather(next_verified_id, tiled=True)
    if not pending_result.host_outputs_prefetched:
        jax.copy_to_host_async(next_verified_id)

    batch_output.next_draft_input.hidden_states = np.asarray(selected_layer0_hidden)[sel]
    topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.topk_index = topk_index
    batch_output.next_draft_input.verified_id = np.asarray(next_verified_id)[sel]
    batch_output.next_draft_input.allocate_lens = batch_output.next_draft_input.allocate_lens[
        : model_worker_batch.real_bs
    ]
    batch_output.next_draft_input.accept_length = accept_host
    batch_output.next_draft_input.accept_length_cpu = accept_host
    batch_output.accept_lens = accept_host


def launch_mtp_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
    *,
    relay_buffers=None,
    relay_future_indices=None,
    relay_valid_mask=None,
):
    """Launch all independent NEXTN layers and optionally publish relay state."""
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None

    update_relay = relay_buffers is not None
    target_hidden = batch_output.logits_output.hidden_states
    draft_input = EagleDraftInput(
        hidden_states=target_hidden,
        allocate_lens=batch_output.next_draft_input.allocate_lens,
        accept_length=getattr(batch_output.next_draft_input, "accept_length", None),
    )
    for field in (
        "verified_id_for_draft_extend",
        "extend_seq_lens_for_draft_extend",
        "logits_indices_for_draft_extend",
        "positions_for_draft_extend",
        "allocate_lens_for_draft_extend",
    ):
        setattr(
            draft_input,
            field,
            getattr(batch_output.next_draft_input, field, None),
        )
    if getattr(batch_output.next_draft_input, "verify_seq_lens", None) is not None:
        draft_input.device_seq_lens_for_draft_extend = True

    draft_batch, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_worker.draft_model_runner,
        batch_output,
        draft_worker.speculative_num_draft_tokens,
    )
    if draft_batch.input_ids.shape[0] <= 0:
        return None

    selector = np.asarray(model_worker_batch.logits_indices_selector)
    runner0 = draft_worker.draft_model_runner
    draft_batch.spec_info_padded.hidden_states = target_hidden
    forward_batch = _make_forward_batch(draft_batch, runner0)
    forward_batch.bid = model_worker_batch.bid

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    selected_positions = getattr(
        batch_output.next_draft_input,
        "sel_pos_for_draft_extend",
        None,
    )
    if selected_positions is None:
        selected_positions = getattr(batch_output.next_draft_input, "sel_pos", None)
    if selected_positions is None:
        selected_positions = jnp.clip(batch_output.accept_lens - 1, 0, None).astype(
            jnp.int32
        )
    selected_positions = _prepare_device_array(
        selected_positions,
        data_sharding,
        "mtp_draft_extend.selected_positions",
    )
    draft_logits_indices_value = getattr(
        draft_batch.spec_info_padded,
        "logits_indices_for_draft_extend",
        None,
    )
    if draft_logits_indices_value is None:
        draft_logits_indices_value = draft_batch.logits_indices
    draft_logits_indices = _prepare_device_array(
        draft_logits_indices_value,
        data_sharding,
        "mtp_draft_extend.logits_indices",
    )
    draft_allocate_lens = getattr(
        batch_output.next_draft_input,
        "allocate_lens_for_draft_extend",
        None,
    )
    if draft_allocate_lens is None:
        draft_allocate_lens = np.zeros_like(
            model_worker_batch.seq_lens,
            dtype=np.int32,
        )
        draft_allocate_lens[selector] = np.asarray(
            batch_output.next_draft_input.allocate_lens
        )
    draft_allocate_lens = _prepare_device_array(
        draft_allocate_lens,
        data_sharding,
        "mtp_draft_extend.allocate_lens",
    )
    draft_verify_seq_lens = _prepare_device_array(
        batch_output.next_draft_input.verify_seq_lens,
        data_sharding,
        "mtp_draft_extend.verify_seq_lens",
    )
    next_verified_id = _prepare_device_array(
        batch_output.next_draft_input.next_verified_id,
        data_sharding,
        "mtp_draft_extend.next_verified_id",
    )
    next_new_seq_lens = _prepare_device_array(
        batch_output.next_draft_input.new_seq_lens,
        data_sharding,
        "mtp_draft_extend.new_seq_lens",
    )
    if update_relay:
        relay_future_indices = _prepare_device_array(
            relay_future_indices,
            data_sharding,
            "mtp_draft_extend.relay_future_indices",
        )
        relay_valid_mask = _prepare_device_array(
            relay_valid_mask,
            data_sharding,
            "mtp_draft_extend.relay_valid_mask",
        )

    all_memory_pools = tuple(
        worker.model_runner.memory_pools for worker in draft_worker._workers
    )
    all_leaves = tuple(
        tuple(worker.model_runner.model_state_leaves)
        for worker in draft_worker._workers
    )
    if not hasattr(draft_worker, "_fused_mtp_draft_extend_jit_fn"):
        draft_worker._fused_mtp_draft_extend_jit_fn = _build_mtp_draft_extend(
            draft_worker.speculative_num_steps
        )

    with jax.set_mesh(draft_worker.mesh):
        (
            selected_hidden,
            token_chain,
            all_pool_updates,
            updated_relay_buffers,
        ) = draft_worker._fused_mtp_draft_extend_jit_fn(
            runner0._model_def,
            runner0._model_state_def,
            all_leaves,
            forward_batch,
            all_memory_pools,
            logits_metadata,
            target_hidden,
            selected_positions,
            draft_logits_indices,
            draft_verify_seq_lens,
            draft_allocate_lens,
            next_verified_id,
            next_new_seq_lens,
            relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            num_layers=draft_worker.speculative_num_steps,
            update_relay=update_relay,
            dp_size=model_worker_batch.dp_size,
        )

    for layer_idx, worker in enumerate(draft_worker._workers):
        worker.model_runner.memory_pools.replace_all(all_pool_updates[layer_idx])

    return FusedDraftExtendPendingResult(
        batch_output=batch_output,
        selected_layer0_hidden=selected_hidden,
        topk_index_stacked=token_chain,
        next_verified_id=batch_output.next_draft_input.next_verified_id,
        accept_lens=batch_output.accept_lens,
        sel=selector,
        updated_relay_buffers=updated_relay_buffers,
        host_outputs_prefetched=not update_relay,
    )


def mtp_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
):
    """Run and restore fused NEXTN state for no-overlap decode."""
    pending_result = launch_mtp_draft_extend_for_decode(
        draft_worker,
        model_worker_batch,
        batch_output,
    )
    restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def launch_eagle_recurrent_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
    *,
    relay_buffers=None,
    relay_future_indices=None,
    relay_valid_mask=None,
):
    """Launch recurrent EAGLE/EAGLE3 draft stages and optionally publish relay state."""
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None

    update_relay = relay_buffers is not None

    target_hidden = batch_output.logits_output.hidden_states
    draft_input = EagleDraftInput(
        hidden_states=target_hidden,
        allocate_lens=batch_output.next_draft_input.allocate_lens,
        accept_length=getattr(batch_output.next_draft_input, "accept_length", None),
    )
    draft_input.verified_id_for_draft_extend = getattr(
        batch_output.next_draft_input,
        "verified_id_for_draft_extend",
        None,
    )
    draft_input.extend_seq_lens_for_draft_extend = getattr(
        batch_output.next_draft_input,
        "extend_seq_lens_for_draft_extend",
        None,
    )
    draft_input.logits_indices_for_draft_extend = getattr(
        batch_output.next_draft_input,
        "logits_indices_for_draft_extend",
        None,
    )
    draft_input.positions_for_draft_extend = getattr(
        batch_output.next_draft_input,
        "positions_for_draft_extend",
        None,
    )
    draft_input.allocate_lens_for_draft_extend = getattr(
        batch_output.next_draft_input,
        "allocate_lens_for_draft_extend",
        None,
    )
    if getattr(batch_output.next_draft_input, "verify_seq_lens", None) is not None:
        draft_input.device_seq_lens_for_draft_extend = True

    mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_worker.draft_model_runner,
        batch_output,
        draft_worker.speculative_num_draft_tokens,
    )
    if mwb.input_ids.shape[0] <= 0:
        return None

    runner = draft_worker.draft_model_runner
    mwb.spec_info_padded.hidden_states = target_hidden
    forward_batch = _make_forward_batch(mwb, runner)
    forward_batch.bid = model_worker_batch.bid

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    draft_logits_indices = _prepare_device_array(
        (
            getattr(mwb.spec_info_padded, "logits_indices_for_draft_extend", None)
            if getattr(mwb.spec_info_padded, "logits_indices_for_draft_extend", None) is not None
            else mwb.logits_indices
        ),
        data_sharding,
        "eagle3_draft_extend.logits_indices",
    )
    draft_allocate_lens = getattr(
        batch_output.next_draft_input,
        "allocate_lens_for_draft_extend",
        None,
    )
    if draft_allocate_lens is None:
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        draft_allocate_lens = np.zeros_like(model_worker_batch.seq_lens, dtype=np.int32)
        draft_allocate_lens[sel] = np.asarray(batch_output.next_draft_input.allocate_lens)
    draft_allocate_lens = _prepare_device_array(
        draft_allocate_lens,
        data_sharding,
        "eagle3_draft_extend.allocate_lens",
    )
    draft_verify_seq_lens = _prepare_device_array(
        batch_output.next_draft_input.verify_seq_lens,
        data_sharding,
        "eagle3_draft_extend.verify_seq_lens",
    )
    next_new_seq_lens = _prepare_device_array(
        batch_output.next_draft_input.new_seq_lens,
        data_sharding,
        "eagle3_draft_extend.new_seq_lens",
    )
    next_verified_id = _prepare_device_array(
        batch_output.next_draft_input.next_verified_id,
        data_sharding,
        "eagle3_draft_extend.next_verified_id",
    )
    if update_relay:
        relay_future_indices = _prepare_device_array(
            relay_future_indices,
            data_sharding,
            "eagle3_draft_extend.relay_future_indices",
        )
        relay_valid_mask = _prepare_device_array(
            relay_valid_mask,
            data_sharding,
            "eagle3_draft_extend.relay_valid_mask",
        )

    if not hasattr(draft_worker, "_fused_eagle3_recurrent_draft_extend_jit_fn"):
        draft_worker._fused_eagle3_recurrent_draft_extend_jit_fn = (
            _build_eagle3_recurrent_draft_extend(
                num_steps=draft_worker.speculative_num_steps,
            )
        )

    with jax.set_mesh(draft_worker.mesh):
        (
            selected_stage0_hidden,
            topk_index_stacked,
            pool_updates,
            updated_relay_buffers,
        ) = (
            draft_worker._fused_eagle3_recurrent_draft_extend_jit_fn(
                runner._model_def,
                runner._model_state_def,
                tuple(runner.model_state_leaves),
                forward_batch,
                runner.memory_pools,
                logits_metadata,
                target_hidden,
                draft_logits_indices,
                draft_verify_seq_lens,
                draft_allocate_lens,
                next_verified_id,
                next_new_seq_lens,
                draft_worker.hot_token_ids,
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                num_steps=draft_worker.speculative_num_steps,
                update_relay=update_relay,
                dp_size=model_worker_batch.dp_size,
            )
        )

    runner.memory_pools.replace_all(pool_updates)
    pending_result = FusedDraftExtendPendingResult(
        batch_output=batch_output,
        selected_layer0_hidden=selected_stage0_hidden,
        topk_index_stacked=topk_index_stacked,
        next_verified_id=batch_output.next_draft_input.next_verified_id,
        accept_lens=batch_output.accept_lens,
        sel=np.asarray(model_worker_batch.logits_indices_selector),
        updated_relay_buffers=updated_relay_buffers,
        host_outputs_prefetched=not update_relay,
    )
    return pending_result


def eagle_recurrent_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
):
    """Run and restore recurrent EAGLE/EAGLE3 state for no-overlap decode."""
    pending_result = launch_eagle_recurrent_draft_extend_for_decode(
        draft_worker,
        model_worker_batch,
        batch_output,
    )
    restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def spec_decode_verify(
    spec_worker,
    model_worker_batch,
    cur_allocate_lens,
    *,
    draft_to_target_token_ids=None,
    draft_padding_prepared: bool = False,
):
    """Run target verify as the first speculative decode JIT."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    draft_input = model_worker_batch.spec_info_padded
    use_relay_state = (
        getattr(draft_input, "future_indices", None) is not None
        and getattr(draft_input, "topk_index", None) is None
    )
    relay_future_indices = None
    if use_relay_state:
        relay_future_indices = np.asarray(draft_input.future_indices, dtype=np.int32)
        relay_future_indices = np.where(relay_future_indices >= 0, relay_future_indices, 0)
    previous_verified_id, previous_token_list = _prepare_verify(
        draft_worker,
        model_worker_batch,
        draft_padding_prepared=draft_padding_prepared,
    )
    spec_info = model_worker_batch.spec_info_padded
    return_target_logits = bool(
        getattr(model_worker_batch, "return_logprob", False)
        or getattr(model_worker_batch, "return_output_logprob_only", False)
    )

    spec_info.allocate_lens = cur_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch)
    rebuild_verify_metadata = draft_padding_prepared
    if not (rebuild_verify_metadata or use_relay_state):
        raise RuntimeError(
            "EAGLE/EAGLE3/NEXTN verify requires device-built fused metadata."
        )
    # Relay verify replaces seq_lens from the device relay buffer; first-round
    # verify uses the padded bootstrap lengths. Both rebuild complete metadata
    # inside fused_verify from this physical page layout.
    target_mr.attn_backend.forward_metadata = (
        target_mr.attn_backend.prepare_paged_kv_layout(model_worker_batch)
    )
    target_forward_batch = _make_forward_batch(model_worker_batch, target_mr)
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _prepare_logits_metadata(model_worker_batch, spec_worker.mesh)
    data_sharding = NamedSharding(spec_worker.mesh, P("data"))
    if relay_future_indices is None:
        constant_cache = getattr(draft_worker, "_fused_verify_constant_cache", None)
        if constant_cache is None:
            constant_cache = draft_worker._fused_verify_constant_cache = {}
        relay_key = ("relay_future_indices", target_forward_batch.seq_lens.shape[0])
        relay_future_indices = constant_cache.get(relay_key)
        if relay_future_indices is None:
            relay_future_indices = _prepare_device_array(
                np.zeros(model_worker_batch.seq_lens.shape, dtype=np.int32),
                data_sharding,
                "verify.relay_future_indices",
            )
            constant_cache[relay_key] = relay_future_indices
    else:
        relay_future_indices = _prepare_device_array(
            relay_future_indices, data_sharding, "verify.relay_future_indices"
        )
    verify_allocate_lens = np.zeros_like(model_worker_batch.seq_lens, dtype=np.int32)
    verify_allocate_lens[model_worker_batch.logits_indices_selector] = cur_allocate_lens
    verify_allocate_lens = _prepare_device_array(
        verify_allocate_lens, data_sharding, "verify.allocate_lens"
    )

    if not hasattr(draft_worker, "_fused_greedy_verify_jit_fn"):
        draft_worker._fused_greedy_verify_jit_fn = _build_verify()

    si = model_worker_batch.sampling_info
    _sv_is_greedy = bool(getattr(si, "is_all_greedy", True))
    _sv_tbs = target_forward_batch.seq_lens.shape[0]
    _sv_enable_top_k = False
    _sv_enable_top_p = False
    if _sv_is_greedy:
        constant_cache = getattr(draft_worker, "_fused_verify_constant_cache", None)
        if constant_cache is None:
            constant_cache = draft_worker._fused_verify_constant_cache = {}
        sampling_key = ("greedy_sampling", _sv_tbs)
        sampling_inputs = constant_cache.get(sampling_key)
        if sampling_inputs is None:
            sampling_inputs = (
                _prepare_device_array(np.ones((_sv_tbs, 1), np.float32), data_sharding),
                _prepare_device_array(
                    np.full((_sv_tbs,), TOP_K_ALL, np.int32), data_sharding
                ),
                _prepare_device_array(np.ones((_sv_tbs,), np.float32), data_sharding),
            )
            constant_cache[sampling_key] = sampling_inputs
        _sv_temps, _sv_topks, _sv_topps = sampling_inputs
    else:
        (
            _sv_temps_host,
            _sv_topks_host,
            _sv_topps_host,
            _sv_enable_top_k,
            _sv_enable_top_p,
        ) = _prepare_rejection_sampling(
            si,
            model_worker_batch,
            _sv_tbs,
            int(target_worker.model_config.vocab_size),
        )
        _sv_temps = _prepare_device_array(_sv_temps_host, data_sharding)
        _sv_topks = _prepare_device_array(_sv_topks_host, data_sharding)
        _sv_topps = _prepare_device_array(_sv_topps_host, data_sharding)
    _sv_thr_single = float(
        getattr(spec_worker.server_args, "speculative_accept_threshold_single", 1.0)
    )
    _sv_thr_acc = float(getattr(spec_worker.server_args, "speculative_accept_threshold_acc", 1.0))

    # Advance the per-step sampling RNG; coins are generated inside the verify JIT
    # from (base_rng, step), so only this small int crosses the host->device boundary.
    target_mr._sampler_step += 1

    with jax.set_mesh(draft_worker.mesh), _count_pjit_cpp_cache_miss() as count:
        (
            target_pool_updates,
            prepared_hidden,
            prepared_verified_id,
            prepared_verified_id_data,
            prepared_next_verified_id,
            prepared_new_seq_lens,
            prepared_accept_lens_host,
            prepared_accept_lens_data,
            prepared_extend_seq_lens,
            prepared_logits_indices,
            prepared_sel_pos,
            prepared_sel_pos_data,
            prepared_predict,
            prepared_positions,
            prepared_positions_data,
            prepared_verify_seq_lens,
            prepared_allocate_lens_data,
            target_logits,
        ) = draft_worker._fused_greedy_verify_jit_fn(
            target_mr._model_def,
            target_mr._model_state_def,
            tuple(target_mr.model_state_leaves),
            target_forward_batch,
            target_mr.memory_pools,
            target_logits_metadata,
            previous_verified_id,
            previous_token_list,
            draft_to_target_token_ids,
            getattr(spec_worker, "spec_relay_buffers", None),
            relay_future_indices,
            verify_allocate_lens,
            target_mr._sampler_base_rng,
            target_mr._sampler_step,
            _sv_temps,
            _sv_topks,
            _sv_topps,
            speculative_num_steps=draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
            return_target_logits=return_target_logits,
            use_relay_state=use_relay_state,
            dp_size=model_worker_batch.dp_size,
            is_greedy=_sv_is_greedy,
            threshold_single=_sv_thr_single,
            threshold_acc=_sv_thr_acc,
            enable_top_k=_sv_enable_top_k,
            enable_top_p=_sv_enable_top_p,
            rebuild_verify_metadata=rebuild_verify_metadata,
        )
        cache_miss_count = count()

    target_mr.memory_pools.replace_all(target_pool_updates)

    next_draft_input = EagleDraftInput(
        verified_id=prepared_verified_id,
        new_seq_lens=prepared_new_seq_lens,
        allocate_lens=cur_allocate_lens,
        hidden_states=prepared_hidden,
        accept_length=prepared_accept_lens_data,
    )
    next_draft_input.verified_id_for_draft_extend = prepared_verified_id_data
    next_draft_input.extend_seq_lens_for_draft_extend = prepared_extend_seq_lens
    next_draft_input.logits_indices_for_draft_extend = prepared_logits_indices
    next_draft_input.positions_for_draft_extend = prepared_positions_data
    next_draft_input.sel_pos_for_draft_extend = prepared_sel_pos_data
    next_draft_input.allocate_lens_for_draft_extend = prepared_allocate_lens_data
    next_draft_input.next_verified_id = prepared_next_verified_id
    next_draft_input.sel_pos = prepared_sel_pos
    next_draft_input.positions = prepared_positions
    next_draft_input.verify_seq_lens = prepared_verify_seq_lens
    if draft_padding_prepared:
        for value in (
            prepared_accept_lens_host,
            prepared_predict,
            prepared_next_verified_id,
        ):
            if hasattr(value, "copy_to_host_async"):
                value.copy_to_host_async()
    batch_output = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=target_logits,
            hidden_states=prepared_hidden,
        ),
        next_token_ids=prepared_predict,
        next_draft_input=next_draft_input,
        accept_lens=prepared_accept_lens_host,
        bid=model_worker_batch.bid,
        cache_miss_count=cache_miss_count,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )
    model_worker_batch.spec_info_padded = next_draft_input
    return batch_output


def spec_decode_eagle_overlap(spec_worker, model_worker_batch, cur_allocate_lens):
    """Launch fused EAGLE/EAGLE3 verify and publish the next relay state."""
    draft_worker = spec_worker.draft_worker
    draft_input = model_worker_batch.spec_info_padded
    use_relay_state = (
        getattr(draft_input, "future_indices", None) is not None
        and getattr(draft_input, "topk_index", None) is None
    )
    if use_relay_state:
        # Recurrent relay buffers retain raw draft-vocabulary ids. The target
        # mapping is consumed once inside fused verify after the device gather.
        draft_to_target_token_ids = draft_worker.hot_token_ids
        draft_padding_prepared = False
    else:
        # The first decode after prefill still carries the width-1 bootstrap
        # seed. Complete its recurrent chain before entering relay steady state.
        draft_to_target_token_ids = draft_worker.prepare_for_fused_verify(
            model_worker_batch
        )
        draft_padding_prepared = True

    batch_output = spec_decode_verify(
        spec_worker,
        model_worker_batch,
        cur_allocate_lens,
        draft_to_target_token_ids=draft_to_target_token_ids,
        draft_padding_prepared=draft_padding_prepared,
    )
    sel = np.asarray(model_worker_batch.logits_indices_selector)
    batch_output.next_draft_input.future_indices = np.asarray(
        model_worker_batch.req_pool_indices,
        dtype=np.int32,
    )[sel]

    from sgl_jax.srt.speculative.overlap_utils import publish_spec_decode_new_seq_lens

    published_new_seq_lens = publish_spec_decode_new_seq_lens(batch_output)
    valid_mask = make_dp_valid_mask(
        model_worker_batch.real_bs_per_dp,
        total_bs=model_worker_batch.req_pool_indices.shape[0],
        per_dp_bs=model_worker_batch.per_dp_bs_size,
    )
    safe_indices = np.where(
        valid_mask,
        np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
        0,
    )
    pending_result = launch_eagle_recurrent_draft_extend_for_decode(
        draft_worker,
        model_worker_batch,
        batch_output,
        relay_buffers=spec_worker.spec_relay_buffers,
        relay_future_indices=safe_indices,
        relay_valid_mask=valid_mask,
    )
    if pending_result is not None:
        spec_worker.spec_relay_buffers = pending_result.updated_relay_buffers
    batch_output.next_draft_input.new_seq_lens = None
    return batch_output, published_new_seq_lens


def spec_decode_mtp_overlap(spec_worker, model_worker_batch, cur_allocate_lens):
    """Launch fused NEXTN verify and publish the next multi-layer draft chain."""
    draft_worker = spec_worker.draft_worker
    draft_input = model_worker_batch.spec_info_padded
    use_relay_state = (
        getattr(draft_input, "future_indices", None) is not None
        and getattr(draft_input, "topk_index", None) is None
    )
    if use_relay_state:
        draft_padding_prepared = False
    else:
        draft_worker.prepare_for_fused_verify(model_worker_batch)
        draft_padding_prepared = True

    batch_output = spec_decode_verify(
        spec_worker,
        model_worker_batch,
        cur_allocate_lens,
        draft_padding_prepared=draft_padding_prepared,
    )
    selector = np.asarray(model_worker_batch.logits_indices_selector)
    batch_output.next_draft_input.future_indices = np.asarray(
        model_worker_batch.req_pool_indices,
        dtype=np.int32,
    )[selector]

    from sgl_jax.srt.speculative.overlap_utils import publish_spec_decode_new_seq_lens

    published_new_seq_lens = publish_spec_decode_new_seq_lens(batch_output)
    valid_mask = make_dp_valid_mask(
        model_worker_batch.real_bs_per_dp,
        total_bs=model_worker_batch.req_pool_indices.shape[0],
        per_dp_bs=model_worker_batch.per_dp_bs_size,
    )
    safe_indices = np.where(
        valid_mask,
        np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
        0,
    )
    pending_result = launch_mtp_draft_extend_for_decode(
        draft_worker,
        model_worker_batch,
        batch_output,
        relay_buffers=spec_worker.spec_relay_buffers,
        relay_future_indices=safe_indices,
        relay_valid_mask=valid_mask,
    )
    if pending_result is not None:
        spec_worker.spec_relay_buffers = pending_result.updated_relay_buffers
    batch_output.next_draft_input.new_seq_lens = None
    return batch_output, published_new_seq_lens
