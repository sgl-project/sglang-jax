"""Fused greedy speculative decode and MTP draft extend."""

from __future__ import annotations

import copy
import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

_SPEC_DECODE_MONOLITHIC = os.environ.get("SGL_JAX_SPEC_DECODE_MONOLITHIC") == "1"


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
    verified_id: jax.Array
    accept_lens: jax.Array
    sel_pos: jax.Array
    predict: jax.Array


class FusedGreedyDraftExtendState(NamedTuple):
    batch_output: object
    positions: jax.Array
    predispatched: object | None = None


class FusedGreedyVerifyPhaseAsync(NamedTuple):
    logits_output: object
    next_token_ids_prefetch: object
    accept_lens_prefetch: object
    accept_lens_device: jax.Array
    allocate_lens: np.ndarray
    scheduler_next_draft_input_allocate_lens: np.ndarray
    selector: np.ndarray
    seq_lens_host: np.ndarray
    draft_extend_state: FusedGreedyDraftExtendState
    bid: int
    cache_miss_count: int
    deferred_target_pool_updates: object | None = None


class PreparedFusedGreedyVerifyLaunch(NamedTuple):
    target_forward_batch: object
    target_logits_metadata: object
    previous_verified_id: object | None
    previous_token_list: object | None
    draft_verify_write_lens: object
    return_target_logits: bool
    return_target_hidden: bool


class FusedDraftExtendDispatch(NamedTuple):
    batch_output: object
    selector: np.ndarray
    selected_layer0_hidden: jax.Array
    topk_index_stacked: jax.Array
    previous_token_list: jax.Array
    selected_verified_id: jax.Array
    verified_id_arr: jax.Array
    accept_lens_device: jax.Array
    materialize_hidden: bool = True
    materialize_topk: bool = True
    accept_lens_host: np.ndarray | None = None
    verified_id_host: np.ndarray | None = None


class FusedVerifyZeroPlaceholders(NamedTuple):
    draft_token: jax.Array
    positions: jax.Array
    retrive_index: jax.Array
    retrive_next_token: jax.Array
    retrive_next_sibling: jax.Array


def _take_with_index_sharding(values, index):
    index_sharding = jax.typeof(index).sharding
    if isinstance(index_sharding, NamedSharding):
        return values.reshape(-1).at[index].get(out_sharding=index_sharding)
    return jnp.take(values.reshape(-1), index)


def _host_array_global(value):
    if isinstance(value, jax.Array):
        if not getattr(value, "is_fully_addressable", True):
            from jax.experimental.multihost_utils import process_allgather

            return np.asarray(process_allgather(value, tiled=True))
        jax.copy_to_host_async(value)
        return np.asarray(jax.device_get(value))
    return np.asarray(value)


def _greedy_prepare_draft_inputs(
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
    return GreedyDraftInputs(
        hidden_states=gathered_hidden,
        positions=gathered_positions,
        new_seq_lens=seq_lens + accept_length,
        select_index=select_index,
        verified_id=verified_id,
        accept_lens=accept_length,
        sel_pos=jnp.clip(accept_length - 1, 0, None).astype(jnp.int32),
    )


def _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
    *,
    target_hidden,
    positions,
    seq_lens,
    draft_tokens,
    target_predict,
    speculative_num_steps,
    speculative_num_draft_tokens,
):
    bs = seq_lens.shape[0]
    n = speculative_num_draft_tokens
    width = speculative_num_steps + 1
    draft_2d = draft_tokens.reshape(bs, n)
    target_predict_2d = target_predict.reshape(bs, n)

    child_matches = draft_2d[:, 1:] == target_predict_2d[:, :-1]
    is_padding = seq_lens == 0
    accepted_children = jnp.cumprod(child_matches.astype(jnp.int32), axis=1).astype(jnp.bool_)
    accepted_children = jnp.where(is_padding[:, None], False, accepted_children)
    accept_length_raw = jnp.sum(accepted_children.astype(jnp.int32), axis=1)
    accept_length = jnp.where(is_padding, 0, accept_length_raw + 1)

    row_ids = jnp.zeros_like(accept_length_raw) + jnp.arange(bs, dtype=jnp.int32)
    base = row_ids[:, None] * n
    child_offsets = jnp.arange(1, width, dtype=jnp.int32)[None, :]
    accept_index_children = jnp.where(accepted_children, base + child_offsets, -1)
    accept_index_2d = jnp.concatenate([base, accept_index_children], axis=1)
    accept_index_2d = jnp.where(is_padding[:, None], -1, accept_index_2d)
    accept_index = accept_index_2d.reshape(-1)

    predict = target_predict.astype(jnp.int32).reshape(-1)
    accept_width = speculative_num_steps + 1
    req_ids = (
        jnp.zeros_like(accept_index)
        + jnp.arange(accept_index.shape[0], dtype=jnp.int32) // accept_width
    )
    per_req_last = req_ids * speculative_num_draft_tokens + speculative_num_draft_tokens - 1
    safe_index = jnp.where(accept_index >= 0, accept_index, per_req_last)
    safe_predict = _take_with_index_sharding(predict, safe_index)
    verified_id = jnp.where(accept_index >= 0, safe_predict, jnp.zeros_like(safe_predict))
    prepared = _greedy_prepare_draft_inputs(
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
        verified_id=prepared.verified_id,
        accept_lens=prepared.accept_lens,
        sel_pos=prepared.sel_pos,
        predict=predict,
    )


def _build_topk1_chain_verify_inputs_device_tuple(
    *,
    verified_id,
    token_list,
    seq_lens,
    num_verify_tokens,
    batch_size,
):
    """Build topk=1 linear-chain verify inputs in-JIT without stacking shardings."""
    n = num_verify_tokens
    bs = batch_size
    tid_range = jnp.arange(n, dtype=jnp.int32)
    draft_tokens = jnp.concatenate(
        [verified_id.astype(jnp.int32)[:, None], token_list[:, : n - 1].astype(jnp.int32)],
        axis=1,
    ).reshape(bs * n)
    positions = (seq_lens.astype(jnp.int32)[:, None] + tid_range[None, :]).reshape(bs * n)
    retrive_index = jnp.arange(bs * n, dtype=jnp.int32)
    retrive_next_token = jnp.broadcast_to(
        jnp.concatenate([jnp.arange(1, n, dtype=jnp.int32), jnp.array([-1], dtype=jnp.int32)]),
        (bs, n),
    ).reshape(bs * n)
    retrive_next_sibling = jnp.full((bs * n,), -1, dtype=jnp.int32)
    return (
        draft_tokens,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
    )


def _per_dp_cumsum_device(lens: jax.Array, dp_size: int, per_dp_bs: int) -> jax.Array:
    lens_2d = lens.astype(jnp.int32).reshape(dp_size, per_dp_bs)
    zeros = jnp.zeros((dp_size, 1), dtype=jnp.int32)
    return jnp.concatenate([zeros, jnp.cumsum(lens_2d, axis=1)], axis=1).reshape(-1)


def _refresh_target_verify_dynamic_metadata_in_jit(target_forward_batch):
    """Fold device-lens target-verify metadata math into the verify JIT."""
    metadata = target_forward_batch.attn_backend.forward_metadata
    if metadata.custom_mask is not None:
        return
    page_size = int(target_forward_batch.attn_backend.page_size)
    dp_size = int(metadata.distribution.shape[0] // 3)
    per_dp_bs = int(target_forward_batch.seq_lens.shape[0] // dp_size)
    draft_token_num = int(target_forward_batch.spec_info.draft_token_num)
    seq_lens = jnp.where(
        target_forward_batch.seq_lens > 0,
        target_forward_batch.seq_lens + draft_token_num,
        0,
    )
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    metadata.seq_lens = seq_lens
    metadata.cu_kv_lens = _per_dp_cumsum_device(aligned_seq_lens, dp_size, per_dp_bs)


def _device_rotate_input_ids(input_ids, ext_lens, sel_pos, new_tokens):
    """Mirror MultiLayerDraftWorker._rotate_ids on device for topk=1."""
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


def _gather_rows_preserve_sharding(values, index):
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding):
        return values.at[index, :].get(out_sharding=sharding)
    return values[index, :]


def _topk1_index_from_logits(logits):
    topk_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
    return topk_idx


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
        all_leaves,
        forward_batch,
        all_memory_pools,
        logits_metadata,
        target_hidden,
        accept_lens,
        *,
        num_layers,
    ):
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None
        input_ids = forward_batch.input_ids
        sel_pos = jnp.clip(accept_lens - 1, 0).astype(jnp.int32)
        forward_batch.spec_info.accept_length = accept_lens
        logits_metadata.accept_lens = accept_lens

        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[i])
            model = nnx.merge(model_def, state)

            forward_batch.spec_info.hidden_states = target_hidden
            forward_batch.input_ids = input_ids

            output, pool_updates, _, _ = model(forward_batch, all_memory_pools[i], logits_metadata)
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else None

            if i == 0:
                layer0_hidden = output.hidden_states

            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            all_topk_index.append(topk_idx)

            if i < num_layers - 1:
                ext_lens = forward_batch.extend_seq_lens
                input_ids = _device_rotate_input_ids(input_ids, ext_lens, sel_pos, topk_idx[:, 0])

        select_index = jnp.arange(sel_pos.shape[0], dtype=jnp.int32) * (num_layers + 1) + sel_pos
        selected_layer0_hidden = _gather_rows_preserve_sharding(layer0_hidden, select_index)
        # Force P() replicated sharding on outputs that must be cross-process
        # consistent. Without this, donate_argnames may let XLA alias output
        # buffers with donated P("data")-sharded pool buffers, silently making
        # np.asarray() return per-process-different values.
        stacked_idx = jnp.stack(all_topk_index, axis=1)
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            selected_layer0_hidden = jax.sharding.reshard(selected_layer0_hidden, rep)
            stacked_idx = jax.sharding.reshard(stacked_idx, rep)

        return (
            selected_layer0_hidden,
            stacked_idx,
            tuple(all_pool_updates),
        )

    return fused_draft_extend


def _build_fused_greedy_decode_jit(num_layers: int, topk: int):
    """Build fused JIT: target verify forward + greedy sample + MTP extend."""
    assert topk == 1, "Fused greedy decode only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["target_memory_pools", "all_memory_pools"],
        static_argnames=[
            "target_model_state_def",
            "draft_model_state_def",
            "num_layers",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "return_target_logits",
            "return_target_hidden",
        ],
    )
    def fused_greedy_decode(
        target_model_def,
        target_model_state_def,
        target_leaves,
        target_forward_batch,
        target_memory_pools,
        target_logits_metadata,
        draft_model_def,
        draft_model_state_def,
        draft_all_leaves,
        draft_forward_batch,
        all_memory_pools,
        draft_logits_metadata,
        previous_verified_id,
        previous_token_list,
        *,
        num_layers,
        speculative_num_steps,
        speculative_num_draft_tokens,
        return_target_logits,
        return_target_hidden,
    ):
        target_bs = target_forward_batch.seq_lens.shape[0]
        (
            draft_tokens,
            positions,
            retrive_index_flat,
            retrive_next_token_flat,
            retrive_next_sibling_flat,
        ) = _build_topk1_chain_verify_inputs_device_tuple(
            verified_id=previous_verified_id,
            token_list=previous_token_list,
            seq_lens=target_forward_batch.seq_lens,
            num_verify_tokens=speculative_num_draft_tokens,
            batch_size=target_bs,
        )
        retrive_index = retrive_index_flat.reshape(target_bs, speculative_num_draft_tokens)
        retrive_next_token = retrive_next_token_flat.reshape(
            target_bs, speculative_num_draft_tokens
        )
        retrive_next_sibling = retrive_next_sibling_flat.reshape(
            target_bs, speculative_num_draft_tokens
        )

        target_forward_batch.input_ids = draft_tokens
        target_forward_batch.positions = positions
        target_forward_batch.spec_info.draft_token = draft_tokens
        target_forward_batch.spec_info.positions = positions
        target_forward_batch.spec_info.retrive_index = retrive_index
        target_forward_batch.spec_info.retrive_next_token = retrive_next_token
        target_forward_batch.spec_info.retrive_next_sibling = retrive_next_sibling

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
        target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(-1)

        prepared = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
            target_hidden=target_hidden,
            positions=target_forward_batch.positions,
            seq_lens=target_forward_batch.seq_lens,
            draft_tokens=draft_tokens,
            target_predict=target_predict,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        input_ids = prepared.verified_id

        draft_forward_batch.positions = prepared.positions
        draft_forward_batch.spec_info.hidden_states = prepared.hidden_states
        draft_forward_batch.spec_info.accept_length = prepared.accept_lens
        draft_logits_metadata.accept_lens = prepared.accept_lens

        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(draft_model_state_def, draft_all_leaves[i])
            model = nnx.merge(draft_model_def, state)

            draft_forward_batch.spec_info.hidden_states = prepared.hidden_states
            draft_forward_batch.input_ids = input_ids

            output, pool_updates, _, _ = model(
                draft_forward_batch, all_memory_pools[i], draft_logits_metadata
            )
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else mesh
            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            rep_hidden = output.hidden_states

            if i == 0:
                layer0_hidden = rep_hidden

            all_topk_index.append(topk_idx)

            if i < num_layers - 1:
                input_ids = _device_rotate_input_ids(
                    input_ids,
                    draft_forward_batch.extend_seq_lens,
                    prepared.sel_pos,
                    topk_idx[:, 0],
                )

        stacked_idx = jnp.stack(all_topk_index, axis=1)
        selected_layer0_hidden = _gather_rows_preserve_sharding(
            layer0_hidden, prepared.select_index
        )
        target_logits_for_host = target_logits if return_target_logits else None
        target_hidden_for_host = target_hidden if return_target_hidden else None
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            selected_layer0_hidden = jax.sharding.reshard(selected_layer0_hidden, rep)
            stacked_idx = jax.sharding.reshard(stacked_idx, rep)
            prepared_accept_lens = jax.sharding.reshard(prepared.accept_lens, rep)
            prepared_predict = jax.sharding.reshard(prepared.predict, rep)
            if return_target_logits:
                target_logits_for_host = jax.sharding.reshard(target_logits, rep)
            if return_target_hidden:
                target_hidden_for_host = jax.sharding.reshard(target_hidden, rep)
        else:
            prepared_accept_lens = prepared.accept_lens
            prepared_predict = prepared.predict

        return (
            selected_layer0_hidden,
            stacked_idx,
            target_pool_updates,
            tuple(all_pool_updates),
            prepared_accept_lens,
            prepared_predict,
            target_logits_for_host,
            target_hidden_for_host,
        )

    return fused_greedy_decode


def _prepare_topk1_verify_placeholders_for_batch(draft_worker, model_worker_batch):
    """Prepare fixed-shape verify placeholders while keeping chain build inside JIT."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    if not getattr(model_worker_batch, "skip_fused_verify_padding_for_decode", False):
        draft_worker.padding_for_decode(model_worker_batch)
    draft_input = model_worker_batch.spec_info_padded
    if isinstance(getattr(draft_input, "new_seq_lens", None), jax.Array):
        model_worker_batch.target_verify_seq_lens_device = draft_input.new_seq_lens
    elif getattr(model_worker_batch, "target_verify_seq_lens_device", None) is not None:
        model_worker_batch.target_verify_seq_lens_device = None
    bs = model_worker_batch.seq_lens.shape[0]
    n = draft_worker.speculative_num_draft_tokens
    placeholders = _get_fused_verify_zero_placeholders(draft_worker, bs=bs, n=n)
    model_worker_batch.spec_info_padded = EagleVerifyInput(
        draft_token=placeholders.draft_token,
        custom_mask=None,
        positions=placeholders.positions,
        retrive_index=placeholders.retrive_index,
        retrive_next_token=placeholders.retrive_next_token,
        retrive_next_sibling=placeholders.retrive_next_sibling,
        retrive_cum_len=None,
        spec_steps=draft_worker.speculative_num_steps,
        topk=draft_worker.topk,
        draft_token_num=draft_worker.speculative_num_draft_tokens,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        seq_lens_sum=model_worker_batch.seq_lens_sum,
        seq_lens_cpu=model_worker_batch.seq_lens,
    )


def _prepare_topk1_verify_placeholders_from_draft_state(draft_worker, model_worker_batch):
    """Prepare previous draft handles plus fixed-shape verify placeholders."""
    draft_input = model_worker_batch.spec_info_padded
    previous_verified_id = draft_input.verified_id
    if isinstance(previous_verified_id, np.ndarray):
        previous_verified_id = np.asarray(previous_verified_id, dtype=np.int32)
    previous_token_list = getattr(draft_input, "previous_token_list", None)
    if previous_token_list is None:
        previous_token_list = draft_input.topk_index[:, :, 0]
    if isinstance(previous_token_list, np.ndarray):
        previous_token_list = np.asarray(previous_token_list, dtype=np.int32)
    else:
        previous_token_list = previous_token_list.astype(jnp.int32)

    _prepare_topk1_verify_placeholders_for_batch(draft_worker, model_worker_batch)
    return previous_verified_id, previous_token_list


def _get_fused_verify_zero_placeholders(draft_worker, *, bs: int, n: int):
    cache = getattr(draft_worker, "_fused_verify_placeholder_cache", None)
    if cache is None:
        cache = {}
        draft_worker._fused_verify_placeholder_cache = cache
    key = (int(bs), int(n))
    cached = cache.get(key)
    if cached is not None:
        return cached

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    flat = bs * n
    cached = FusedVerifyZeroPlaceholders(
        draft_token=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
        positions=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
        retrive_index=jax.device_put(np.zeros((bs, n), dtype=np.int32), data_sharding),
        retrive_next_token=jax.device_put(np.zeros((bs, n), dtype=np.int32), data_sharding),
        retrive_next_sibling=jax.device_put(np.zeros((bs, n), dtype=np.int32), data_sharding),
    )
    cache[key] = cached
    return cached


def _select_next_verified_id_for_verify(verified_id_arr, accept_lens):
    accept_width = verified_id_arr.shape[0] // accept_lens.shape[0]
    select_index = (
        jnp.arange(accept_lens.shape[0], dtype=jnp.int32) * accept_width
        + accept_lens.astype(jnp.int32)
        - 1
    )
    return _take_with_index_sharding(verified_id_arr, select_index)


def _device_array_preserve_device(value, sharding):
    from sgl_jax.srt.utils.jax_utils import device_array

    if value is None:
        return None
    if isinstance(value, jax.Array):
        try:
            value_sharding = getattr(value, "sharding", None)
            if value_sharding == sharding:
                return value
            if (
                value_sharding is not None
                and hasattr(value_sharding, "is_equivalent_to")
                and value_sharding.is_equivalent_to(sharding, value.ndim)
            ):
                return value
        except Exception:
            pass
        return jax.device_put(value, sharding)
    return device_array(value, sharding=sharding)


def _cached_host_device_array_preserve_device(owner, field_name: str, value, sharding):
    if value is None or isinstance(value, jax.Array):
        return _device_array_preserve_device(value, sharding)
    if not isinstance(value, np.ndarray):
        return _device_array_preserve_device(value, sharding)

    cache = getattr(owner, "_fused_verify_host_device_array_cache", None)
    if cache is None:
        cache = {}
        owner._fused_verify_host_device_array_cache = cache
    cached = cache.get(field_name)
    if cached is not None:
        cached_host, cached_device = cached
        if (
            cached_host.shape == value.shape
            and cached_host.dtype == value.dtype
            and np.array_equal(cached_host, value)
        ):
            return cached_device

    device_value = _device_array_preserve_device(value, sharding)
    cache[field_name] = (np.array(value, copy=True), device_value)
    return device_value


def _logits_metadata_from_model_worker_batch_preserve_device(
    batch, mesh, *, include_accept_lens: bool = True, cache_owner=None
):
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata

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
        extend_seq_lens=(
            _cached_host_device_array_preserve_device(
                cache_owner,
                "logits_metadata_extend_seq_lens",
                batch.extend_seq_lens,
                sharding,
            )
            if cache_owner is not None
            else _device_array_preserve_device(batch.extend_seq_lens, sharding)
        ),
        logits_indices=(
            _cached_host_device_array_preserve_device(
                cache_owner,
                "logits_metadata_logits_indices",
                batch.logits_indices,
                sharding,
            )
            if cache_owner is not None
            else _device_array_preserve_device(batch.logits_indices, sharding)
        ),
        accept_lens=_device_array_preserve_device(accept_lens, sharding),
        extend_seq_lens_cpu=None,
        extend_logprob_start_lens_cpu=None,
        extend_logprob_pruned_lens_cpu=None,
        top_logprobs_nums=getattr(batch, "top_logprobs_nums", None),
        token_ids_logprobs=getattr(batch, "token_ids_logprobs", None),
        extend_input_logprob_token_ids_device=(
            _cached_host_device_array_preserve_device(
                cache_owner,
                "logits_metadata_extend_input_logprob_token_ids",
                getattr(batch, "extend_input_logprob_token_ids", None),
                sharding,
            )
            if cache_owner is not None
            else _device_array_preserve_device(
                getattr(batch, "extend_input_logprob_token_ids", None), sharding
            )
        ),
    )


def _clone_attn_backend_with_metadata(attn_backend, forward_metadata):
    if type(attn_backend).__name__ == "FlashAttention":
        cloned = type(attn_backend).__new__(type(attn_backend))
        cloned.__dict__.update(attn_backend.__dict__)
        cloned.forward_metadata = forward_metadata
        return cloned
    if hasattr(attn_backend, "tree_flatten") and hasattr(type(attn_backend), "tree_unflatten"):
        _, aux_data = attn_backend.tree_flatten()
        return type(attn_backend).tree_unflatten(aux_data, (forward_metadata,))
    cloned = copy.deepcopy(attn_backend)
    cloned.forward_metadata = forward_metadata
    return cloned


def _forward_batch_init_new_preserve_device(batch, model_runner, *, attn_backend=None):
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
        seq_lens=_device_array_preserve_device(
            (
                getattr(batch, "target_verify_seq_lens_device", None)
                if getattr(batch, "target_verify_seq_lens_device", None) is not None
                else batch.seq_lens
            ),
            data_sharding,
        ),
        out_cache_loc=_cached_host_device_array_preserve_device(
            model_runner,
            "target_verify_out_cache_loc",
            batch.out_cache_loc,
            data_sharding,
        ),
        positions=_device_array_preserve_device(batch.positions, data_sharding),
        mrope_positions=_device_array_preserve_device(batch.mrope_positions, replicated_2d),
        req_pool_indices=_cached_host_device_array_preserve_device(
            model_runner,
            "target_verify_req_pool_indices",
            batch.req_pool_indices,
            data_sharding,
        ),
        cache_loc=_cached_host_device_array_preserve_device(
            model_runner,
            "target_verify_cache_loc",
            batch.cache_loc,
            data_sharding,
        ),
        extend_prefix_lens=_device_array_preserve_device(batch.extend_prefix_lens, data_sharding),
        extend_seq_lens=_device_array_preserve_device(batch.extend_seq_lens, data_sharding),
        lora_ids=batch.lora_ids,
        lora_scalings=lora_scalings,
        lora_token_indices=lora_token_indices,
        lora_ranks=lora_ranks,
        attn_backend=attn_backend if attn_backend is not None else model_runner.attn_backend,
        spec_info=batch.spec_info_padded,
        spec_algorithm=batch.spec_algorithm,
        capture_hidden_mode=batch.capture_hidden_mode,
        input_embedding=input_embedding,
        apply_for_deepstack=batch.apply_for_deepstack,
        deepstack_visual_embedding=deepstack_visual_embedding,
        expert_location_metadata=get_global_expert_location_metadata(),
        recurrent_indices=_device_array_preserve_device(batch.recurrent_indices, data_sharding),
    )


def draft_extend_for_decode_fused(
    draft_worker, model_worker_batch, batch_output, *, materialize_hidden: bool = True
):
    """Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.

    Fuses all N MTP layer forwards into a single jit call.
    """
    dispatch = _dispatch_draft_extend_for_decode_fused(
        draft_worker,
        model_worker_batch,
        batch_output,
        materialize_hidden=materialize_hidden,
    )
    if dispatch is None:
        return
    _materialize_draft_extend_for_decode_fused(draft_worker, model_worker_batch, dispatch)


def _dispatch_draft_extend_for_decode_fused(
    draft_worker,
    model_worker_batch,
    batch_output,
    *,
    materialize_hidden: bool = True,
    materialize_topk: bool = True,
):
    """Prepare and dispatch fused draft_extend without materializing results.

    This lets the split verify path issue Phase B while the verify device work
    and Phase A D2H are still in flight. Materialization can happen later at
    the ordinary Phase B boundary.
    """
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None
    target_hidden = batch_output.logits_output.hidden_states

    draft_input = EagleDraftInput(
        hidden_states=target_hidden, allocate_lens=batch_output.allocate_lens
    )
    mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_worker.draft_model_runner,
        batch_output,
        draft_worker.speculative_num_draft_tokens,
        build_logits_metadata=False,
    )
    if mwb.input_ids.shape[0] <= 0:
        return None
    sel = np.asarray(model_worker_batch.logits_indices_selector)

    mr0 = draft_worker._workers[0].model_runner
    logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        mwb,
        draft_worker.mesh,
        include_accept_lens=False,
        cache_owner=mr0,
    )
    mwb.spec_info_padded.hidden_states = target_hidden
    shared_fb = _forward_batch_init_new_preserve_device(mwb, mr0)
    shared_fb.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_jit_fn"):
        draft_worker._fused_jit_fn = _build_fused_draft_extend_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    with jax.set_mesh(draft_worker.mesh):
        selected_layer0_hidden, topk_index_stacked, all_pool_updates = draft_worker._fused_jit_fn(
            mr0._model_def,
            mr0._model_state_def,
            tuple(all_leaves),
            shared_fb,
            tuple(all_memory_pools),
            logits_metadata,
            target_hidden,
            batch_output.accept_lens,
            num_layers=draft_worker.speculative_num_steps,
        )

    previous_token_list = (
        topk_index_stacked[:, :, 0] if len(topk_index_stacked.shape) == 3 else topk_index_stacked
    )
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    verified_id_arr = batch_output.next_draft_input.verified_id
    selected_verified_id = getattr(
        batch_output.next_draft_input,
        "selected_verified_id",
        None,
    )
    if selected_verified_id is None:
        selected_verified_id = _select_next_verified_id_for_verify(
            verified_id_arr,
            batch_output.accept_lens,
        )
    if materialize_hidden and hasattr(verified_id_arr, "copy_to_host_async"):
        jax.copy_to_host_async(verified_id_arr)
    if materialize_hidden:
        jax.copy_to_host_async(selected_layer0_hidden)
    if materialize_topk:
        jax.copy_to_host_async(topk_index_stacked)
    jax.copy_to_host_async(batch_output.accept_lens)

    return FusedDraftExtendDispatch(
        batch_output=batch_output,
        selector=sel,
        selected_layer0_hidden=selected_layer0_hidden,
        topk_index_stacked=topk_index_stacked,
        previous_token_list=previous_token_list,
        selected_verified_id=selected_verified_id,
        verified_id_arr=verified_id_arr,
        accept_lens_device=batch_output.accept_lens,
        materialize_hidden=materialize_hidden,
        materialize_topk=materialize_topk,
    )


def _materialize_draft_extend_for_decode_fused(draft_worker, model_worker_batch, dispatch):
    batch_output = dispatch.batch_output
    sel = dispatch.selector
    accept_host = (
        dispatch.accept_lens_host
        if dispatch.accept_lens_host is not None
        else np.asarray(dispatch.accept_lens_device)
    )
    materialize_sel = sel
    seq_lens = getattr(model_worker_batch, "seq_lens", None)
    if seq_lens is not None:
        seq_lens = np.asarray(seq_lens)
        if seq_lens.shape[0] > 0 and sel.shape[0] > 0 and int(np.max(sel)) < seq_lens.shape[0]:
            materialize_sel = sel[seq_lens[sel] > 0]
    if (
        getattr(model_worker_batch, "is_precompile_dummy", False)
        and materialize_sel.shape[0] > 0
        and not (accept_host[materialize_sel] >= 1).any()
    ):
        materialize_sel = materialize_sel[:0]
    assert (
        accept_host[materialize_sel] >= 1
    ).all(), f"accept_length < 1: {accept_host[materialize_sel]}"
    if dispatch.materialize_hidden:
        batch_output.next_draft_input.hidden_states = np.asarray(dispatch.selected_layer0_hidden)[
            materialize_sel
        ]
    else:
        batch_output.next_draft_input.hidden_states = None
    if dispatch.materialize_topk:
        topk_index = np.asarray(dispatch.topk_index_stacked)[materialize_sel]
        batch_output.next_draft_input.topk_p = np.ones(topk_index.shape, dtype=np.float32)
        batch_output.next_draft_input.topk_index = topk_index
    if dispatch.verified_id_host is not None:
        batch_output.next_draft_input.verified_id = dispatch.verified_id_host
    else:
        select_index = (
            materialize_sel * (draft_worker.speculative_num_steps + 1)
            + accept_host[materialize_sel]
            - 1
        )
        batch_output.next_draft_input.verified_id = np.asarray(dispatch.verified_id_arr)[
            select_index
        ]
    batch_output.allocate_lens = batch_output.allocate_lens[: len(materialize_sel)]
    next_new_seq_lens = getattr(batch_output.next_draft_input, "new_seq_lens", None)
    if next_new_seq_lens is not None:
        batch_output.next_draft_input.new_seq_lens = _host_array_global(next_new_seq_lens)[
            materialize_sel
        ]
    batch_output.next_draft_input.allocate_lens = batch_output.allocate_lens
    verify_write_lens = getattr(batch_output.next_draft_input, "verify_write_lens", None)
    if verify_write_lens is not None:
        batch_output.next_draft_input.verify_write_lens = _host_array_global(verify_write_lens)[
            materialize_sel
        ]
    batch_output.accept_lens = accept_host


def _prepare_model_worker_batch_for_draft_extend(draft_worker, model_worker_batch, batch_output):
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
    model_worker_batch.spec_info_padded.accept_length = None
    model_worker_batch.input_ids = batch_output.next_draft_input.verified_id

    forward_metadata = draft_worker.draft_model_runner.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )
    draft_worker.draft_model_runner.attn_backend.forward_metadata = forward_metadata
    logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, draft_worker.mesh, include_accept_lens=False
    )
    return model_worker_batch, logits_metadata


def _materialize_fused_greedy_batch_output_for_scheduler(
    *,
    batch_output,
    selector,
    real_bs,
    seq_lens_host,
    layer0_hidden,
    topk_index_stacked,
    accept_lens_device,
    predict_device,
    speculative_num_draft_tokens,
    target_logits,
    target_hidden,
):
    """Materialize the scheduler-facing fields after the single fused decode JIT."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput

    with jax.profiler.TraceAnnotation("fused_greedy_batch_output_d2h"):
        jax.copy_to_host_async(layer0_hidden)
        jax.copy_to_host_async(topk_index_stacked)
        jax.copy_to_host_async(accept_lens_device)
        jax.copy_to_host_async(predict_device)

        batch_output.logits_output = LogitsProcessorOutput(
            next_token_logits=target_logits,
            hidden_states=target_hidden,
        )
        batch_output.next_draft_input.hidden_states = np.asarray(layer0_hidden)[selector]
        topk_index = np.asarray(topk_index_stacked)[selector]
        batch_output.next_draft_input.topk_p = np.ones(topk_index.shape, dtype=np.float32)
        batch_output.next_draft_input.topk_index = topk_index
        accept_lens = np.asarray(accept_lens_device)
        predict = np.asarray(predict_device)
        seq_lens_host = np.asarray(seq_lens_host)
        verified_pos = selector * speculative_num_draft_tokens + accept_lens[selector] - 1
        batch_output.next_draft_input.verified_id = predict[verified_pos]
        original_seq_lens = seq_lens_host[selector] - speculative_num_draft_tokens + 1
        batch_output.next_draft_input.new_seq_lens = original_seq_lens + accept_lens[selector]
        batch_output.allocate_lens = batch_output.allocate_lens[:real_bs]
        batch_output.next_draft_input.allocate_lens = batch_output.allocate_lens
        verify_write_lens = getattr(batch_output.next_draft_input, "verify_write_lens", None)
        if verify_write_lens is not None:
            batch_output.next_draft_input.verify_write_lens = np.asarray(verify_write_lens)[
                selector
            ]
        batch_output.accept_lens = accept_lens
        batch_output.next_token_ids = predict
        return batch_output


def _build_fused_greedy_verify_jit(*, donate_target_memory_pools: bool = True):
    """Build JIT A: draft/target verify/sample only.

    Returns scheduler-visible values and a private draft_extend_state payload.
    The function must not run draft_extend.
    """

    jit_kwargs = {
        "static_argnames": [
            "target_model_state_def",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "return_target_logits",
            "return_target_hidden",
        ],
    }
    if donate_target_memory_pools:
        jit_kwargs["donate_argnames"] = ["target_memory_pools"]

    @partial(jax.jit, **jit_kwargs)
    def fused_greedy_verify(
        target_model_def,
        target_model_state_def,
        target_leaves,
        target_forward_batch,
        target_memory_pools,
        target_logits_metadata,
        previous_verified_id,
        previous_token_list,
        *,
        speculative_num_steps,
        speculative_num_draft_tokens,
        return_target_logits,
        return_target_hidden,
    ):
        target_bs = target_forward_batch.seq_lens.shape[0]
        (
            draft_tokens,
            positions,
            retrive_index_flat,
            retrive_next_token_flat,
            retrive_next_sibling_flat,
        ) = _build_topk1_chain_verify_inputs_device_tuple(
            verified_id=previous_verified_id,
            token_list=previous_token_list,
            seq_lens=target_forward_batch.seq_lens,
            num_verify_tokens=speculative_num_draft_tokens,
            batch_size=target_bs,
        )
        retrive_index = retrive_index_flat.reshape(target_bs, speculative_num_draft_tokens)
        retrive_next_token = retrive_next_token_flat.reshape(
            target_bs, speculative_num_draft_tokens
        )
        retrive_next_sibling = retrive_next_sibling_flat.reshape(
            target_bs, speculative_num_draft_tokens
        )

        target_forward_batch.input_ids = draft_tokens
        target_forward_batch.positions = positions
        target_forward_batch.spec_info.draft_token = draft_tokens
        target_forward_batch.spec_info.positions = positions
        target_forward_batch.spec_info.retrive_index = retrive_index
        target_forward_batch.spec_info.retrive_next_token = retrive_next_token
        target_forward_batch.spec_info.retrive_next_sibling = retrive_next_sibling

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
        target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(-1)

        prepared = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
            target_hidden=target_hidden,
            positions=target_forward_batch.positions,
            seq_lens=target_forward_batch.seq_lens,
            draft_tokens=draft_tokens,
            target_predict=target_predict,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        target_logits_for_host = target_logits if return_target_logits else None
        target_hidden_for_host = target_hidden if return_target_hidden else None
        selected_verified_id = _take_with_index_sharding(
            prepared.verified_id,
            prepared.select_index,
        )
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            prepared_hidden = jax.sharding.reshard(prepared.hidden_states, rep)
            prepared_positions = jax.sharding.reshard(prepared.positions, rep)
            prepared_new_seq_lens = prepared.new_seq_lens
            prepared_verified_id = jax.sharding.reshard(prepared.verified_id, rep)
            prepared_selected_verified_id = jax.sharding.reshard(selected_verified_id, rep)
            prepared_accept_lens = jax.sharding.reshard(prepared.accept_lens, rep)
            prepared_predict = jax.sharding.reshard(prepared.predict, rep)
            if return_target_logits:
                target_logits_for_host = jax.sharding.reshard(target_logits, rep)
            if return_target_hidden:
                target_hidden_for_host = jax.sharding.reshard(target_hidden, rep)
        else:
            prepared_hidden = prepared.hidden_states
            prepared_positions = prepared.positions
            prepared_new_seq_lens = prepared.new_seq_lens
            prepared_verified_id = prepared.verified_id
            prepared_selected_verified_id = selected_verified_id
            prepared_accept_lens = prepared.accept_lens
            prepared_predict = prepared.predict

        return (
            prepared_hidden,
            prepared_positions,
            prepared_new_seq_lens,
            prepared_verified_id,
            prepared_selected_verified_id,
            prepared_accept_lens,
            prepared_predict,
            target_pool_updates,
            target_logits_for_host,
            target_hidden_for_host,
        )

    return fused_greedy_verify


def _get_fused_greedy_verify_jit_fn(draft_worker, *, defer_target_pool_updates: bool):
    if defer_target_pool_updates:
        if not hasattr(draft_worker, "_fused_greedy_verify_jit_fn_no_target_donate"):
            draft_worker._fused_greedy_verify_jit_fn_no_target_donate = (
                _build_fused_greedy_verify_jit(donate_target_memory_pools=False)
            )
        return draft_worker._fused_greedy_verify_jit_fn_no_target_donate
    if not hasattr(draft_worker, "_fused_greedy_verify_jit_fn"):
        draft_worker._fused_greedy_verify_jit_fn = _build_fused_greedy_verify_jit()
    return draft_worker._fused_greedy_verify_jit_fn


def prepare_fused_greedy_verify_launch(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
    *,
    require_previous: bool = True,
):
    """Prepare host metadata and ForwardBatch for a fused greedy verify launch."""
    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    draft_verify_write_lens = getattr(
        model_worker_batch.spec_info_padded,
        "verify_write_lens",
        None,
    )
    if require_previous:
        previous_verified_id, previous_token_list = (
            _prepare_topk1_verify_placeholders_from_draft_state(draft_worker, model_worker_batch)
        )
    else:
        previous_verified_id = None
        previous_token_list = None
        _prepare_topk1_verify_placeholders_for_batch(draft_worker, model_worker_batch)

    spec_info = model_worker_batch.spec_info_padded
    return_target_logits = bool(
        getattr(model_worker_batch, "return_logprob", False)
        or getattr(model_worker_batch, "return_output_logprob_only", False)
    )
    return_target_hidden = bool(getattr(model_worker_batch, "return_hidden_states", False))

    spec_info.allocate_lens = padded_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch, spec_worker.page_size, target_worker)
    target_forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(model_worker_batch)
    target_attn_backend = _clone_attn_backend_with_metadata(
        target_mr.attn_backend,
        target_forward_metadata,
    )
    target_forward_batch = _forward_batch_init_new_preserve_device(
        model_worker_batch,
        target_mr,
        attn_backend=target_attn_backend,
    )
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch,
        spec_worker.mesh,
        cache_owner=target_mr,
    )
    return PreparedFusedGreedyVerifyLaunch(
        target_forward_batch=target_forward_batch,
        target_logits_metadata=target_logits_metadata,
        previous_verified_id=previous_verified_id,
        previous_token_list=previous_token_list,
        draft_verify_write_lens=draft_verify_write_lens,
        return_target_logits=return_target_logits,
        return_target_hidden=return_target_hidden,
    )


def spec_decode_verify_phase_enqueue(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
    *,
    prepared_launch=None,
):
    """Enqueue fused verify and Phase A D2H, returning before materialization."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    selector = np.asarray(model_worker_batch.logits_indices_selector)
    seq_lens_host = np.asarray(model_worker_batch.seq_lens)
    if prepared_launch is None:
        with jax.profiler.TraceAnnotation("prepare_fused_greedy_verify_launch"):
            prepared_launch = prepare_fused_greedy_verify_launch(
                spec_worker,
                model_worker_batch,
                padded_allocate_lens,
                compact_allocate_lens,
            )
    target_forward_batch = prepared_launch.target_forward_batch
    target_logits_metadata = prepared_launch.target_logits_metadata
    previous_verified_id = prepared_launch.previous_verified_id
    previous_token_list = prepared_launch.previous_token_list
    draft_verify_write_lens = prepared_launch.draft_verify_write_lens
    return_target_logits = prepared_launch.return_target_logits
    return_target_hidden = prepared_launch.return_target_hidden
    assert previous_verified_id is not None, "prepared verify launch missing previous_verified_id"
    assert previous_token_list is not None, "prepared verify launch missing previous_token_list"

    defer_target_pool_updates = bool(
        getattr(model_worker_batch, "defer_target_pool_updates", False)
    )
    fused_greedy_verify_jit_fn = _get_fused_greedy_verify_jit_fn(
        draft_worker,
        defer_target_pool_updates=defer_target_pool_updates,
    )

    with (
        jax.set_mesh(draft_worker.mesh),
        jax.profiler.TraceAnnotation("submit_fused_greedy_verify_jit"),
    ):
        (
            prepared_hidden,
            prepared_positions,
            prepared_new_seq_lens,
            prepared_verified_id,
            selected_verified_id,
            accept_lens_device,
            predict_device,
            target_pool_updates,
            target_logits,
            target_hidden,
        ) = fused_greedy_verify_jit_fn(
            target_mr._model_def,
            target_mr._model_state_def,
            tuple(target_mr.model_state_leaves),
            target_forward_batch,
            target_mr.memory_pools,
            target_logits_metadata,
            previous_verified_id,
            previous_token_list,
            speculative_num_steps=draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
            return_target_logits=return_target_logits,
            return_target_hidden=return_target_hidden,
        )

    if not defer_target_pool_updates:
        target_mr.memory_pools.replace_all(target_pool_updates)

    draft_next_draft_input = EagleDraftInput(
        verified_id=prepared_verified_id,
        new_seq_lens=prepared_new_seq_lens,
        allocate_lens=padded_allocate_lens,
        hidden_states=prepared_hidden,
    )
    draft_next_draft_input.verify_write_lens = draft_verify_write_lens
    draft_next_draft_input.selected_verified_id = selected_verified_id
    draft_batch_output = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=prepared_hidden,
        ),
        next_token_ids=predict_device,
        next_draft_input=draft_next_draft_input,
        accept_lens=accept_lens_device,
        allocate_lens=compact_allocate_lens,
        bid=model_worker_batch.bid,
        cache_miss_count=0,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )

    with jax.profiler.TraceAnnotation("fused_greedy_verify_phase_d2h_enqueue"):
        accept_lens_prefetch = jax.copy_to_host_async(accept_lens_device)
        predict_prefetch = jax.copy_to_host_async(predict_device)

    return FusedGreedyVerifyPhaseAsync(
        logits_output=LogitsProcessorOutput(
            next_token_logits=target_logits,
            hidden_states=target_hidden,
        ),
        next_token_ids_prefetch=predict_prefetch,
        accept_lens_prefetch=accept_lens_prefetch,
        accept_lens_device=accept_lens_device,
        allocate_lens=padded_allocate_lens,
        scheduler_next_draft_input_allocate_lens=compact_allocate_lens,
        selector=selector,
        seq_lens_host=seq_lens_host,
        draft_extend_state=FusedGreedyDraftExtendState(
            batch_output=draft_batch_output,
            positions=prepared_positions,
            predispatched=None,
        ),
        bid=model_worker_batch.bid,
        cache_miss_count=0,
        deferred_target_pool_updates=target_pool_updates if defer_target_pool_updates else None,
    )


def spec_decode_materialize_verify_phase(async_result):
    """Materialize Phase A host data and build scheduler-visible result."""
    from sgl_jax.srt.managers.scheduler import SpecVerifyPhaseResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    with jax.profiler.TraceAnnotation("fused_greedy_verify_phase_d2h"):
        accept_lens = np.asarray(async_result.accept_lens_prefetch)
        predict = np.asarray(async_result.next_token_ids_prefetch)

    state = async_result.draft_extend_state
    stride = state.batch_output.next_draft_input.verified_id.shape[0] // accept_lens.shape[0]
    selector = async_result.selector
    verified_pos = selector * stride + accept_lens[selector] - 1
    verified_id_host = predict[verified_pos]
    original_seq_lens = async_result.seq_lens_host - stride + 1
    verify_write_lens = getattr(
        state.batch_output.next_draft_input,
        "verify_write_lens",
        None,
    )
    if verify_write_lens is not None:
        verify_write_lens = np.asarray(verify_write_lens)[selector]
    scheduler_next_draft_input = EagleDraftInput(
        verified_id=verified_id_host,
        new_seq_lens=original_seq_lens[selector] + accept_lens[selector],
        allocate_lens=async_result.scheduler_next_draft_input_allocate_lens,
        verify_write_lens=verify_write_lens,
    )
    padded_new_seq_lens_host = original_seq_lens + accept_lens

    return SpecVerifyPhaseResult(
        logits_output=async_result.logits_output,
        next_token_ids=predict,
        accept_lens=accept_lens,
        allocate_lens=async_result.allocate_lens,
        scheduler_next_draft_input=scheduler_next_draft_input,
        draft_extend_state=state,
        bid=async_result.bid,
        cache_miss_count=async_result.cache_miss_count,
        padded_new_seq_lens_host=padded_new_seq_lens_host,
    )


def spec_decode_verify_phase(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
    *,
    predispatch_draft_extend: bool = False,
):
    """Run phase A and return SpecVerifyPhaseResult without draft_extend."""
    async_result = spec_decode_verify_phase_enqueue(
        spec_worker,
        model_worker_batch,
        padded_allocate_lens,
        compact_allocate_lens,
    )
    verify_result = spec_decode_materialize_verify_phase(async_result)

    predispatched = None
    if predispatch_draft_extend:
        with jax.profiler.TraceAnnotation("predispatch_spec_draft_extend_phase"):
            predispatched = _dispatch_draft_extend_for_decode_fused(
                spec_worker.draft_worker,
                model_worker_batch,
                verify_result.draft_extend_state.batch_output,
                materialize_hidden=False,
                materialize_topk=False,
            )
    if predispatched is not None:
        predispatched = predispatched._replace(
            accept_lens_host=verify_result.accept_lens,
            verified_id_host=verify_result.scheduler_next_draft_input.verified_id,
        )
        verify_result = verify_result.__class__(
            logits_output=verify_result.logits_output,
            next_token_ids=verify_result.next_token_ids,
            accept_lens=verify_result.accept_lens,
            allocate_lens=verify_result.allocate_lens,
            scheduler_next_draft_input=verify_result.scheduler_next_draft_input,
            draft_extend_state=verify_result.draft_extend_state._replace(
                predispatched=predispatched,
            ),
            bid=verify_result.bid,
            cache_miss_count=verify_result.cache_miss_count,
            padded_new_seq_lens_host=verify_result.padded_new_seq_lens_host,
        )

    return verify_result


def spec_decode_draft_extend_phase(spec_worker, model_worker_batch, verify_phase_result):
    """Run phase B and return the next-round spec forward state."""
    from sgl_jax.srt.managers.scheduler import SpecDraftExtendPhaseResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    state = verify_phase_result.draft_extend_state
    model_worker_batch.positions = state.positions
    padded_next_draft_input = None
    padded_req_pool_indices = None
    if state.predispatched is not None:
        with jax.profiler.TraceAnnotation("materialize_predispatched_spec_draft_extend_phase"):
            _materialize_draft_extend_for_decode_fused(
                spec_worker.draft_worker,
                model_worker_batch,
                state.predispatched,
            )
        if not state.predispatched.materialize_topk:
            padded_next_draft_input = EagleDraftInput(
                topk_index=state.predispatched.topk_index_stacked,
                verified_id=state.predispatched.selected_verified_id,
                new_seq_lens=state.batch_output.next_draft_input.new_seq_lens,
                allocate_lens=getattr(
                    state.batch_output.next_draft_input,
                    "allocate_lens",
                    None,
                ),
            )
            padded_next_draft_input.verify_write_lens = getattr(
                state.batch_output.next_draft_input,
                "verify_write_lens",
                None,
            )
            padded_next_draft_input.previous_token_list = state.predispatched.previous_token_list
            padded_req_pool_indices = np.asarray(model_worker_batch.req_pool_indices).copy()
    else:
        draft_extend_for_decode_fused(
            spec_worker.draft_worker,
            model_worker_batch,
            state.batch_output,
        )
    selector = np.asarray(model_worker_batch.logits_indices_selector)
    req_pool_indices = np.asarray(model_worker_batch.req_pool_indices)[selector]
    return SpecDraftExtendPhaseResult(
        next_draft_input=state.batch_output.next_draft_input,
        req_pool_indices=req_pool_indices,
        padded_next_draft_input=padded_next_draft_input,
        padded_req_pool_indices=padded_req_pool_indices,
        padded_new_seq_lens_host=getattr(
            verify_phase_result,
            "padded_new_seq_lens_host",
            None,
        ),
    )


def spec_decode_pending_draft_extend_result_from_predispatch(
    model_worker_batch,
    verify_phase_result,
):
    """Build next-round draft state from predispatched Phase B device handles.

    The split-overlap path already dispatches fused draft_extend before Phase A
    D2H materializes. For the common exact padded-layout decode case, the next
    verify needs only the padded topk/verified-token device handles. Returning
    those handles directly lets the ordered worker thread accept the next batch
    without synchronously materializing Phase B host rows.
    """
    from sgl_jax.srt.managers.scheduler import SpecDraftExtendPhaseResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    state = getattr(verify_phase_result, "draft_extend_state", None)
    dispatch = getattr(state, "predispatched", None)
    if dispatch is None or getattr(dispatch, "materialize_topk", True):
        return None
    batch_output = getattr(state, "batch_output", None)
    next_draft_input = getattr(batch_output, "next_draft_input", None)

    padded_next_draft_input = EagleDraftInput(
        topk_index=dispatch.topk_index_stacked,
        verified_id=dispatch.selected_verified_id,
        new_seq_lens=getattr(next_draft_input, "new_seq_lens", None),
        allocate_lens=getattr(next_draft_input, "allocate_lens", None),
    )
    padded_next_draft_input.verify_write_lens = getattr(
        next_draft_input,
        "verify_write_lens",
        None,
    )
    padded_next_draft_input.previous_token_list = dispatch.previous_token_list

    selector = np.asarray(model_worker_batch.logits_indices_selector)
    req_pool_indices = np.asarray(model_worker_batch.req_pool_indices)[selector]
    return SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=req_pool_indices,
        padded_next_draft_input=padded_next_draft_input,
        padded_req_pool_indices=np.asarray(model_worker_batch.req_pool_indices).copy(),
        padded_new_seq_lens_host=getattr(
            verify_phase_result,
            "padded_new_seq_lens_host",
            None,
        ),
    )


def spec_decode_dispatch_draft_extend_for_pending(
    spec_worker,
    model_worker_batch,
    verify_phase_result,
):
    """Dispatch Phase B after Phase A has already been published to scheduler."""
    from sgl_jax.srt.managers.scheduler import SpecDraftExtendPhaseResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    state = verify_phase_result.draft_extend_state
    model_worker_batch.positions = state.positions
    dispatch = _dispatch_draft_extend_for_decode_fused(
        spec_worker.draft_worker,
        model_worker_batch,
        state.batch_output,
        materialize_hidden=False,
        materialize_topk=False,
    )
    if dispatch is None:
        return None

    padded_next_draft_input = EagleDraftInput(
        topk_index=dispatch.topk_index_stacked,
        verified_id=dispatch.selected_verified_id,
        new_seq_lens=state.batch_output.next_draft_input.new_seq_lens,
        allocate_lens=getattr(
            state.batch_output.next_draft_input,
            "allocate_lens",
            None,
        ),
    )
    padded_next_draft_input.verify_write_lens = getattr(
        state.batch_output.next_draft_input,
        "verify_write_lens",
        None,
    )
    padded_next_draft_input.previous_token_list = dispatch.previous_token_list

    selector = np.asarray(model_worker_batch.logits_indices_selector)
    req_pool_indices = np.asarray(model_worker_batch.req_pool_indices)[selector]
    return SpecDraftExtendPhaseResult(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=req_pool_indices,
        padded_next_draft_input=padded_next_draft_input,
        padded_req_pool_indices=np.asarray(model_worker_batch.req_pool_indices).copy(),
        padded_new_seq_lens_host=getattr(
            verify_phase_result,
            "padded_new_seq_lens_host",
            None,
        ),
    )


def _convert_split_phase_to_generation_result(verify_phase_result, draft_extend_result):
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult

    return GenerationBatchResult(
        logits_output=verify_phase_result.logits_output,
        next_token_ids=verify_phase_result.next_token_ids,
        next_draft_input=draft_extend_result.next_draft_input,
        accept_lens=verify_phase_result.accept_lens,
        allocate_lens=verify_phase_result.allocate_lens,
        bid=verify_phase_result.bid,
        cache_miss_count=verify_phase_result.cache_miss_count,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )


def _spec_decode_monolithic(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
):
    """Run the original single-JIT fused verify + draft-extend path."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    draft_verify_write_lens = getattr(
        model_worker_batch.spec_info_padded,
        "verify_write_lens",
        None,
    )
    previous_verified_id, previous_token_list = _prepare_topk1_verify_placeholders_from_draft_state(
        draft_worker, model_worker_batch
    )
    spec_info = model_worker_batch.spec_info_padded
    return_target_logits = bool(
        getattr(model_worker_batch, "return_logprob", False)
        or getattr(model_worker_batch, "return_output_logprob_only", False)
    )
    return_target_hidden = bool(getattr(model_worker_batch, "return_hidden_states", False))

    spec_info.allocate_lens = padded_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch, spec_worker.page_size, target_worker)
    target_forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(model_worker_batch)
    target_attn_backend = _clone_attn_backend_with_metadata(
        target_mr.attn_backend,
        target_forward_metadata,
    )
    target_forward_batch = _forward_batch_init_new_preserve_device(
        model_worker_batch,
        target_mr,
        attn_backend=target_attn_backend,
    )
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch,
        spec_worker.mesh,
        cache_owner=target_mr,
    )

    hidden_size = target_worker.model_config.hidden_size
    placeholder_hidden = np.zeros((spec_info.draft_token.shape[0], hidden_size), dtype=np.float32)
    placeholder_logits = np.zeros((spec_info.draft_token.shape[0], 1), dtype=np.float32)

    next_draft_input = EagleDraftInput(
        verified_id=spec_info.draft_token,
        new_seq_lens=model_worker_batch.seq_lens,
        allocate_lens=padded_allocate_lens,
        hidden_states=placeholder_hidden,
    )
    next_draft_input.verify_write_lens = draft_verify_write_lens
    next_draft_input.draft_token = spec_info.draft_token
    next_draft_input.retrive_index = spec_info.retrive_index
    next_draft_input.retrive_next_token = spec_info.retrive_next_token
    next_draft_input.retrive_next_sibling = spec_info.retrive_next_sibling
    batch_output = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=placeholder_logits,
            hidden_states=placeholder_hidden,
        ),
        next_token_ids=spec_info.draft_token,
        next_draft_input=next_draft_input,
        accept_lens=np.ones(model_worker_batch.seq_lens.shape, dtype=np.int32),
        allocate_lens=compact_allocate_lens,
        bid=model_worker_batch.bid,
        cache_miss_count=0,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )
    model_worker_batch.spec_info_padded = next_draft_input

    draft_mwb, draft_logits_metadata = _prepare_model_worker_batch_for_draft_extend(
        draft_worker, model_worker_batch, batch_output
    )
    if draft_mwb.input_ids.shape[0] <= 0:
        return batch_output

    draft_mr0 = draft_worker._workers[0].model_runner
    draft_forward_batch = _forward_batch_init_new_preserve_device(draft_mwb, draft_mr0)
    draft_forward_batch.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_greedy_decode_jit_fn"):
        draft_worker._fused_greedy_decode_jit_fn = _build_fused_greedy_decode_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    with jax.set_mesh(draft_worker.mesh):
        (
            layer0_hidden,
            topk_index_stacked,
            target_pool_updates,
            all_pool_updates,
            accept_lens_device,
            predict_device,
            target_logits,
            target_hidden,
        ) = draft_worker._fused_greedy_decode_jit_fn(
            target_mr._model_def,
            target_mr._model_state_def,
            tuple(target_mr.model_state_leaves),
            target_forward_batch,
            target_mr.memory_pools,
            target_logits_metadata,
            draft_mr0._model_def,
            draft_mr0._model_state_def,
            tuple(all_leaves),
            draft_forward_batch,
            tuple(all_memory_pools),
            draft_logits_metadata,
            previous_verified_id,
            previous_token_list,
            num_layers=draft_worker.speculative_num_steps,
            speculative_num_steps=draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
            return_target_logits=return_target_logits,
            return_target_hidden=return_target_hidden,
        )

    target_mr.memory_pools.replace_all(target_pool_updates)
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    return _materialize_fused_greedy_batch_output_for_scheduler(
        batch_output=batch_output,
        selector=np.asarray(model_worker_batch.logits_indices_selector),
        real_bs=model_worker_batch.real_bs,
        seq_lens_host=model_worker_batch.seq_lens,
        layer0_hidden=layer0_hidden,
        topk_index_stacked=topk_index_stacked,
        accept_lens_device=accept_lens_device,
        predict_device=predict_device,
        speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
        target_logits=target_logits,
        target_hidden=target_hidden,
    )


def spec_decode(
    spec_worker,
    model_worker_batch,
    padded_allocate_lens,
    compact_allocate_lens,
):
    """Run one fused speculative decode round."""
    if _SPEC_DECODE_MONOLITHIC:
        return _spec_decode_monolithic(
            spec_worker,
            model_worker_batch,
            padded_allocate_lens,
            compact_allocate_lens,
        )
    verify_phase_result = spec_decode_verify_phase(
        spec_worker,
        model_worker_batch,
        padded_allocate_lens,
        compact_allocate_lens,
    )
    draft_extend_result = spec_decode_draft_extend_phase(
        spec_worker,
        model_worker_batch,
        verify_phase_result,
    )
    return _convert_split_phase_to_generation_result(
        verify_phase_result,
        draft_extend_result,
    )
