"""Fused greedy speculative decode and MTP draft extend."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


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
    accept_lens: object
    sel: np.ndarray


class SpecDecodePendingDraftExtendResult(NamedTuple):
    draft_worker: object
    model_worker_batch: object
    pending_result: FusedDraftExtendPendingResult | None


@partial(jax.jit, static_argnames=("num_steps",))
def _select_next_verified_id(verified_id, accept_lens, *, num_steps: int):
    accept_width = num_steps + 1
    safe_accept_lens = jnp.clip(accept_lens, 1, None)
    select_index = (
        jnp.arange(accept_lens.shape[0], dtype=jnp.int32) * accept_width + safe_accept_lens - 1
    )
    return _take_with_index_sharding(verified_id, select_index)


def build_padded_draft_input_from_pending(flat_spec, selector: np.ndarray, total_bs: int):
    """Build next decode's DP-padded draft state without host-restoring tensors."""
    pending = getattr(flat_spec, "pending_draft_extend_result", None)
    if pending is None or getattr(pending, "pending_result", None) is None:
        return None

    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    pending_result = pending.pending_result
    batch_output = pending_result.batch_output
    data_sharding = NamedSharding(pending.draft_worker.mesh, P("data"))
    topk_index = jax.device_put(pending_result.topk_index_stacked, data_sharding)
    hidden_states = pending_result.selected_layer0_hidden
    accept_lens = pending_result.accept_lens
    verified_id = jax.device_put(
        _select_next_verified_id(
            batch_output.next_draft_input.verified_id,
            accept_lens,
            num_steps=topk_index.shape[1],
        ),
        data_sharding,
    )

    def _scatter_host(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        out = np.zeros((total_bs,) + a.shape[1:], dtype=a.dtype)
        out[selector] = a
        return out

    return EagleDraftInput(
        topk_p=jax.device_put(jnp.ones_like(topk_index, dtype=jnp.float32), data_sharding),
        topk_index=topk_index,
        hidden_states=hidden_states,
        verified_id=verified_id,
        accept_length=accept_lens,
        allocate_lens=_scatter_host(flat_spec.allocate_lens),
        new_seq_lens=_scatter_host(flat_spec.new_seq_lens),
        capture_hidden_mode=flat_spec.capture_hidden_mode,
    )


def _take_with_index_sharding(values, index):
    index_sharding = jax.typeof(index).sharding
    if isinstance(index_sharding, NamedSharding):
        return values.reshape(-1).at[index].get(out_sharding=index_sharding)
    return jnp.take(values.reshape(-1), index)


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
        new_seq_lens=seq_lens + accept_length + 1,
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
        safe_index=safe_index,
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


def _device_rotate_prefill_input_ids(input_ids, extend_seq_lens, verified_id, dp_size, per_dp_bs):
    per_dp_tokens = input_ids.shape[0] // dp_size
    ids = input_ids.reshape(dp_size, per_dp_tokens)
    ext = extend_seq_lens.reshape(dp_size, per_dp_bs)
    verified = verified_id.reshape(dp_size, per_dp_bs)
    tok = jnp.arange(per_dp_tokens, dtype=jnp.int32)

    def rotate_rank(ids_rank, ext_rank, verified_rank):
        starts = jnp.cumsum(ext_rank, axis=0) - ext_rank
        ends = starts + ext_rank
        in_req = (tok[None, :] >= starts[:, None]) & (tok[None, :] < ends[:, None])
        has_req = jnp.any(in_req, axis=0)
        slot = jnp.argmax(in_req.astype(jnp.int32), axis=0)
        req_starts = starts.at[slot].get()
        req_lens = ext_rank.at[slot].get()
        req_verified = verified_rank.at[slot].get()
        shifted_index = jnp.minimum(tok + 1, per_dp_tokens - 1)
        shifted = ids_rank.at[shifted_index].get()
        is_last = has_req & ((tok - req_starts) == (req_lens - 1))
        rotated = jnp.where(is_last, req_verified, shifted)
        return jnp.where(has_req, rotated, ids_rank)

    return jax.vmap(rotate_rank)(ids, ext, verified).reshape(input_ids.shape)


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
        sel_pos,
        *,
        num_layers,
    ):
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None
        input_ids = forward_batch.input_ids

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


def _build_fused_greedy_verify_jit(topk: int):
    """Build target verify JIT for greedy NEXTN decode."""
    assert topk == 1, "Fused greedy verify only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["target_memory_pools"],
        static_argnames=[
            "target_model_state_def",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "return_target_logits",
        ],
    )
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

        target_logits_for_host = (
            _gather_rows_preserve_sharding(target_logits, prepared.safe_index)
            if return_target_logits
            else None
        )
        prepared_hidden = prepared.hidden_states
        prepared_verified_id = prepared.verified_id
        prepared_new_seq_lens = prepared.new_seq_lens
        prepared_accept_lens = prepared.accept_lens
        prepared_sel_pos = prepared.sel_pos
        prepared_predict = prepared.predict
        prepared_positions = prepared.positions

        if mesh is not None:
            rep = NamedSharding(mesh, P())
            prepared_hidden = jax.sharding.reshard(prepared_hidden, rep)
            prepared_verified_id = jax.sharding.reshard(prepared_verified_id, rep)
            prepared_new_seq_lens = jax.sharding.reshard(prepared_new_seq_lens, rep)
            prepared_accept_lens = jax.sharding.reshard(prepared_accept_lens, rep)
            prepared_sel_pos = jax.sharding.reshard(prepared_sel_pos, rep)
            prepared_predict = jax.sharding.reshard(prepared_predict, rep)
            prepared_positions = jax.sharding.reshard(prepared_positions, rep)
            if return_target_logits:
                target_logits_for_host = jax.sharding.reshard(target_logits_for_host, rep)

        return (
            target_pool_updates,
            prepared_hidden,
            prepared_verified_id,
            prepared_new_seq_lens,
            prepared_accept_lens,
            prepared_sel_pos,
            prepared_predict,
            prepared_positions,
            target_logits_for_host,
        )

    return fused_greedy_verify


def _build_fused_greedy_prefill_jit(num_layers: int, topk: int):
    """Build prefill JIT: target extend + all MTP draft-extend layers."""
    assert topk == 1, "Fused greedy prefill only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["target_memory_pools", "all_memory_pools"],
        static_argnames=[
            "target_model_state_def",
            "draft_model_state_def",
            "num_layers",
            "dp_size",
            "per_dp_bs",
        ],
    )
    def fused_greedy_prefill(
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
        draft_logits_indices,
        all_memory_pools,
        draft_logits_metadata,
        *,
        num_layers,
        dp_size,
        per_dp_bs,
    ):
        target_state = jax.tree_util.tree_unflatten(target_model_state_def, target_leaves)
        target_model = nnx.merge(target_model_def, target_state)
        target_output, target_pool_updates, _, _ = target_model(
            target_forward_batch,
            target_memory_pools,
            target_logits_metadata,
        )

        target_logits = target_output.next_token_logits
        target_hidden = target_output.hidden_states
        next_token_ids = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
        input_ids = _device_rotate_prefill_input_ids(
            draft_forward_batch.input_ids,
            draft_forward_batch.extend_seq_lens,
            next_token_ids,
            dp_size,
            per_dp_bs,
        )

        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None

        draft_forward_batch.spec_info.hidden_states = target_hidden
        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(draft_model_state_def, draft_all_leaves[i])
            model = nnx.merge(draft_model_def, state)

            draft_forward_batch.input_ids = input_ids
            draft_forward_batch.spec_info.hidden_states = target_hidden
            output, pool_updates, _, _ = model(
                draft_forward_batch, all_memory_pools[i], draft_logits_metadata
            )
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else mesh
            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            all_topk_index.append(topk_idx)
            if i == 0:
                layer0_hidden = output.hidden_states
            if i < num_layers - 1:
                input_ids = _device_rotate_prefill_input_ids(
                    input_ids,
                    draft_forward_batch.extend_seq_lens,
                    topk_idx[:, 0],
                    dp_size,
                    per_dp_bs,
                )

        last_idx = draft_logits_indices
        if dp_size > 1:
            per_dp_tokens = layer0_hidden.shape[0] // dp_size
            rank_ids = jnp.arange(last_idx.shape[0], dtype=jnp.int32) // per_dp_bs
            last_idx = last_idx + rank_ids * per_dp_tokens

        selected_layer0_hidden = _gather_rows_preserve_sharding(layer0_hidden, last_idx)
        stacked_idx = jnp.stack(all_topk_index, axis=1)
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            data_sharding = NamedSharding(mesh, P("data"))
            next_token_ids = jax.sharding.reshard(jnp.copy(next_token_ids), data_sharding)
            selected_layer0_hidden = jax.sharding.reshard(selected_layer0_hidden, rep)
            stacked_idx = jax.sharding.reshard(stacked_idx, rep)

        return (
            target_output,
            next_token_ids,
            target_pool_updates,
            tuple(all_pool_updates),
            selected_layer0_hidden,
            stacked_idx,
        )

    return fused_greedy_prefill


def _prepare_topk1_verify_placeholders_from_draft_state(draft_worker, model_worker_batch):
    """Prepare fixed-shape verify placeholders while keeping chain build inside JIT."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    draft_worker.padding_for_decode(model_worker_batch)
    draft_input = model_worker_batch.spec_info_padded
    previous_verified_id = draft_input.verified_id
    if isinstance(previous_verified_id, np.ndarray):
        previous_verified_id = np.asarray(previous_verified_id, dtype=np.int32)
    topk_index = draft_input.topk_index
    if len(topk_index.shape) == 3 and topk_index.shape[-1] == 1:
        previous_token_list = (
            np.squeeze(topk_index, axis=-1)
            if isinstance(topk_index, np.ndarray)
            else jnp.squeeze(topk_index, axis=-1)
        )
    else:
        previous_token_list = topk_index[:, :, 0]
    if isinstance(previous_token_list, np.ndarray):
        previous_token_list = np.asarray(previous_token_list, dtype=np.int32)
    else:
        previous_token_list = previous_token_list.astype(jnp.int32)

    bs = model_worker_batch.seq_lens.shape[0]
    n = draft_worker.speculative_num_draft_tokens
    flat = bs * n
    model_worker_batch.spec_info_padded = EagleVerifyInput(
        draft_token=np.zeros((flat,), dtype=np.int32),
        custom_mask=None,
        positions=np.zeros((flat,), dtype=np.int32),
        retrive_index=np.zeros((bs, n), dtype=np.int32),
        retrive_next_token=np.zeros((bs, n), dtype=np.int32),
        retrive_next_sibling=np.zeros((bs, n), dtype=np.int32),
        retrive_cum_len=None,
        spec_steps=draft_worker.speculative_num_steps,
        topk=draft_worker.topk,
        draft_token_num=draft_worker.speculative_num_draft_tokens,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        seq_lens_sum=model_worker_batch.seq_lens_sum,
        seq_lens_cpu=model_worker_batch.seq_lens,
    )
    return previous_verified_id, previous_token_list


def _device_array_preserve_device(value, sharding):
    from sgl_jax.srt.utils.jax_utils import device_array

    if value is None:
        return None
    if isinstance(value, jax.Array):
        return jax.device_put(value, sharding)
    return device_array(value, sharding=sharding)


def _logits_metadata_from_model_worker_batch_preserve_device(
    batch, mesh, *, include_accept_lens: bool = True
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


def prepare_spec_prefill_forward_batch(spec_worker, model_worker_batch):
    """Prepare the target ForwardBatch before speculative prefill is queued."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

    target_mr = spec_worker.target_worker.model_runner
    model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_forward_metadata(
        model_worker_batch
    )
    model_worker_batch.forward_batch = _forward_batch_init_new_preserve_device(
        model_worker_batch, target_mr
    )
    model_worker_batch.forward_batch.bid = model_worker_batch.bid
    return model_worker_batch.forward_batch


def launch_fused_draft_extend_for_decode(draft_worker, model_worker_batch, batch_output):
    """Launch fused MTP draft extend and return deferred host restore state."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None
    target_hidden = batch_output.logits_output.hidden_states

    draft_input = EagleDraftInput(
        hidden_states=target_hidden,
        allocate_lens=batch_output.next_draft_input.allocate_lens,
    )
    mwb, logits_metadata = draft_input.prepare_for_extend_after_verify(
        model_worker_batch,
        draft_worker.draft_model_runner,
        batch_output,
        draft_worker.speculative_num_draft_tokens,
    )
    if mwb.input_ids.shape[0] <= 0:
        return None

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    if hasattr(batch_output.next_draft_input, "sel_pos"):
        sel_pos = batch_output.next_draft_input.sel_pos
    else:
        sel_pos = jnp.clip(batch_output.accept_lens - 1, 0, None).astype(jnp.int32)

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
            sel_pos_device,
            num_layers=draft_worker.speculative_num_steps,
        )

    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    return FusedDraftExtendPendingResult(
        batch_output=batch_output,
        selected_layer0_hidden=selected_layer0_hidden,
        topk_index_stacked=topk_index_stacked,
        accept_lens=batch_output.accept_lens,
        sel=sel,
    )


def restore_fused_draft_extend_result(draft_worker, model_worker_batch, pending_result):
    if pending_result is None:
        return

    batch_output = pending_result.batch_output
    selected_layer0_hidden = pending_result.selected_layer0_hidden
    topk_index_stacked = pending_result.topk_index_stacked
    accept_host = np.asarray(jax.device_get(pending_result.accept_lens))
    sel = pending_result.sel

    verified_id_arr = batch_output.next_draft_input.verified_id
    if hasattr(verified_id_arr, "copy_to_host_async"):
        jax.copy_to_host_async(verified_id_arr)

    jax.copy_to_host_async(selected_layer0_hidden)
    jax.copy_to_host_async(topk_index_stacked)

    batch_output.next_draft_input.hidden_states = np.asarray(selected_layer0_hidden)[sel]
    topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.topk_p = np.ones(topk_index.shape, dtype=np.float32)
    batch_output.next_draft_input.topk_index = topk_index
    select_index = sel * (draft_worker.speculative_num_steps + 1) + accept_host[sel] - 1
    batch_output.next_draft_input.verified_id = np.asarray(verified_id_arr)[select_index]
    batch_output.next_draft_input.allocate_lens = batch_output.next_draft_input.allocate_lens[
        : model_worker_batch.real_bs
    ]
    batch_output.accept_lens = accept_host


def restore_spec_decode_pending_draft_extend_result(pending_draft_extend_result):
    if pending_draft_extend_result is None:
        return None
    if pending_draft_extend_result.pending_result is None:
        return None

    batch_output = pending_draft_extend_result.pending_result.batch_output
    if getattr(batch_output.next_draft_input, "topk_index", None) is not None:
        return batch_output

    restore_fused_draft_extend_result(
        pending_draft_extend_result.draft_worker,
        pending_draft_extend_result.model_worker_batch,
        pending_draft_extend_result.pending_result,
    )
    return batch_output


def draft_extend_for_decode_fused(draft_worker, model_worker_batch, batch_output):
    """Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.

    Fuses all N MTP layer forwards into a single jit call.
    """
    pending_result = launch_fused_draft_extend_for_decode(
        draft_worker, model_worker_batch, batch_output
    )
    restore_fused_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def spec_prefill(spec_worker, model_worker_batch, launch_done=None):
    """Run greedy prefill target forward and MTP draft-extend in one JIT."""
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner

    if getattr(model_worker_batch, "forward_batch", None) is None:
        target_forward_batch = prepare_spec_prefill_forward_batch(spec_worker, model_worker_batch)
    else:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_forward_metadata(
            model_worker_batch
        )
        target_forward_batch = model_worker_batch.forward_batch
        target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, spec_worker.mesh
    )

    hidden_size = target_worker.model_config.hidden_size
    model_worker_batch.spec_info_padded = EagleDraftInput(
        hidden_states=np.zeros((len(model_worker_batch.input_ids), hidden_size), dtype=np.float32),
        verified_id=np.zeros((len(model_worker_batch.seq_lens),), dtype=np.int32),
        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
        allocate_lens=model_worker_batch.seq_lens,
    )
    model_worker_batch.return_hidden_states = False
    model_worker_batch.spec_info_padded.capture_hidden_mode = CaptureHiddenMode.FULL
    model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

    draft_mr0 = draft_worker._workers[0].model_runner
    draft_mr0.attn_backend.forward_metadata = draft_mr0.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )
    draft_forward_batch = ForwardBatch.init_new(model_worker_batch, draft_mr0)
    draft_forward_batch.input_ids = target_forward_batch.input_ids
    draft_forward_batch.bid = model_worker_batch.bid
    draft_logits_indices = _device_array_preserve_device(
        model_worker_batch.logits_indices,
        NamedSharding(draft_worker.mesh, P("data")),
    )
    draft_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, draft_worker.mesh
    )

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_greedy_prefill_jit_fn"):
        draft_worker._fused_greedy_prefill_jit_fn = _build_fused_greedy_prefill_jit(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
        )

    with jax.set_mesh(draft_worker.mesh), jtu.count_pjit_cpp_cache_miss() as count:
        (
            logits_output,
            next_token_ids,
            target_pool_updates,
            all_pool_updates,
            layer0_hidden,
            topk_index_stacked,
        ) = draft_worker._fused_greedy_prefill_jit_fn(
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
            draft_logits_indices,
            tuple(all_memory_pools),
            draft_logits_metadata,
            num_layers=draft_worker.speculative_num_steps,
            dp_size=model_worker_batch.dp_size,
            per_dp_bs=model_worker_batch.per_dp_bs_size,
        )
        cache_miss_count = count()

    if launch_done is not None:
        launch_done.set()

    target_mr.memory_pools.replace_all(target_pool_updates)
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    relay_next_token_ids = next_token_ids
    host_next_token_ids = next_token_ids
    if model_worker_batch.dp_size > 1:
        from jax.experimental.multihost_utils import process_allgather

        host_next_token_ids = process_allgather(host_next_token_ids, tiled=True)

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    jax.copy_to_host_async(host_next_token_ids)
    jax.copy_to_host_async(layer0_hidden)
    jax.copy_to_host_async(topk_index_stacked)

    topk_index = np.asarray(topk_index_stacked)[sel]
    model_worker_batch.spec_info_padded.hidden_states = np.asarray(layer0_hidden)[sel]
    model_worker_batch.spec_info_padded.topk_p = np.ones(topk_index.shape, dtype=np.float32)
    model_worker_batch.spec_info_padded.topk_index = topk_index
    model_worker_batch.spec_info_padded.allocate_lens = np.asarray(model_worker_batch.seq_lens)[sel]
    model_worker_batch.spec_info_padded.verified_id = np.asarray(host_next_token_ids)[sel]

    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=relay_next_token_ids if launch_done is not None else host_next_token_ids,
        next_draft_input=model_worker_batch.spec_info_padded,
        bid=model_worker_batch.bid,
        cache_miss_count=cache_miss_count,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )


def spec_decode_verify_phase(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run target verify as the first speculative decode JIT."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    draft_worker = spec_worker.draft_worker
    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner
    previous_verified_id, previous_token_list = _prepare_topk1_verify_placeholders_from_draft_state(
        draft_worker, model_worker_batch
    )
    spec_info = model_worker_batch.spec_info_padded
    return_target_logits = bool(
        getattr(model_worker_batch, "return_logprob", False)
        or getattr(model_worker_batch, "return_output_logprob_only", False)
    )

    spec_info.allocate_lens = cur_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch, spec_worker.page_size, target_worker)
    target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )
    target_forward_batch = _forward_batch_init_new_preserve_device(model_worker_batch, target_mr)
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, spec_worker.mesh
    )

    if not hasattr(draft_worker, "_fused_greedy_verify_jit_fn"):
        draft_worker._fused_greedy_verify_jit_fn = _build_fused_greedy_verify_jit(
            topk=draft_worker.topk,
        )

    with jax.set_mesh(draft_worker.mesh), jtu.count_pjit_cpp_cache_miss() as count:
        (
            target_pool_updates,
            prepared_hidden,
            prepared_verified_id,
            prepared_new_seq_lens,
            prepared_accept_lens,
            prepared_sel_pos,
            prepared_predict,
            prepared_positions,
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
            speculative_num_steps=draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=draft_worker.speculative_num_draft_tokens,
            return_target_logits=return_target_logits,
        )
        cache_miss_count = count()

    target_mr.memory_pools.replace_all(target_pool_updates)

    next_draft_input = EagleDraftInput(
        verified_id=prepared_verified_id,
        new_seq_lens=prepared_new_seq_lens,
        allocate_lens=cur_allocate_lens,
        hidden_states=prepared_hidden,
    )
    next_draft_input.sel_pos = prepared_sel_pos
    next_draft_input.positions = prepared_positions
    batch_output = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=target_logits,
            hidden_states=prepared_hidden,
        ),
        next_token_ids=prepared_predict,
        next_draft_input=next_draft_input,
        accept_lens=prepared_accept_lens,
        bid=model_worker_batch.bid,
        cache_miss_count=cache_miss_count,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
    )
    model_worker_batch.spec_info_padded = next_draft_input
    return batch_output


def spec_decode_draft_extend_phase(spec_worker, model_worker_batch, batch_output):
    """Run MTP draft extend as the second speculative decode JIT."""
    spec_worker.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
    return batch_output


def spec_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run speculative decode as verify JIT followed by draft-extend JIT."""
    batch_output = spec_decode_verify_phase(spec_worker, model_worker_batch, cur_allocate_lens)
    return spec_decode_draft_extend_phase(spec_worker, model_worker_batch, batch_output)


def spec_decode_overlap(spec_worker, model_worker_batch, cur_allocate_lens):
    """Launch decode verify and draft-extend without restoring draft results inline."""
    batch_output = spec_decode_verify_phase(spec_worker, model_worker_batch, cur_allocate_lens)
    pending_draft_extend_result = SpecDecodePendingDraftExtendResult(
        draft_worker=spec_worker.draft_worker,
        model_worker_batch=model_worker_batch,
        pending_result=launch_fused_draft_extend_for_decode(
            spec_worker.draft_worker,
            model_worker_batch,
            batch_output,
        ),
    )
    from sgl_jax.srt.speculative.overlap_worker import publish_spec_decode_new_seq_lens

    published_new_seq_lens = publish_spec_decode_new_seq_lens(batch_output)
    batch_output.next_draft_input.pending_draft_extend_result = pending_draft_extend_result
    return batch_output, published_new_seq_lens
