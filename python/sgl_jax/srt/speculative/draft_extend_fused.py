"""Fused greedy speculative decode and MTP draft extend."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
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
    verified_id: jax.Array
    accept_lens: jax.Array
    sel_pos: jax.Array
    predict: jax.Array


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


def _build_fused_target_verify_greedy_jit(speculative_num_steps, speculative_num_draft_tokens):
    """Build the fused target-verify JIT (cached on the spec worker).

    Runs the target model forward AND the topk=1 greedy chain accept inside a
    SINGLE jit, so the large ``next_token_logits`` (bs*draft_tokens, vocab) is
    consumed by ``argmax`` in-graph and never crosses a JIT boundary / gets
    materialized. The draft_extend forward stays a separate dispatch (overlap
    boundary kept between target and draft, not inside verify).
    """

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
    def fused_target_verify_greedy(
        target_model_def,
        target_model_state_def,
        target_leaves,
        target_forward_batch,
        target_memory_pools,
        target_logits_metadata,
        *,
        speculative_num_steps,
        speculative_num_draft_tokens,
        return_target_logits,
    ):
        target_state = jax.tree_util.tree_unflatten(target_model_state_def, target_leaves)
        target_model = nnx.merge(target_model_def, target_state)
        target_output, target_pool_updates, _, _ = target_model(
            target_forward_batch,
            target_memory_pools,
            target_logits_metadata,
        )

        logits = target_output.next_token_logits
        hidden = target_output.hidden_states

        mesh = None
        for value in (logits, hidden):
            sharding = jax.typeof(value).sharding
            if isinstance(sharding, NamedSharding):
                mesh = sharding.mesh
                break

        # prepare_for_verify already set input_ids=draft_token, positions, and
        # decremented seq_lens, so read the chain inputs straight off the batch.
        target_predict = jnp.argmax(logits, axis=-1).astype(jnp.int32).reshape(-1)
        draft_tokens = target_forward_batch.input_ids
        positions = target_forward_batch.positions
        seq_lens = target_forward_batch.seq_lens
        # Replicate the chain inputs before the accept helper. This mirrors the
        # eager greedy verify (which replicate_to_mesh's hidden then gathers on
        # host-built indices), so it is NOT an extra cost vs eager: hidden was
        # already replicated there, and the index tensors are bs*draft_tokens
        # int32 (negligible all-gather even at dp>1). It also sidesteps the
        # P("data") 1-D -> 2-D reshape that otherwise yields an illegal
        # PartitionSpec("data","data") inside the helper under explicit sharding.
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            hidden = jax.sharding.reshard(hidden, rep)
            target_predict = jax.sharding.reshard(target_predict, rep)
            draft_tokens = jax.sharding.reshard(draft_tokens, rep)
            positions = jax.sharding.reshard(positions, rep)
            seq_lens = jax.sharding.reshard(seq_lens, rep)

        prepared = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
            target_hidden=hidden,
            positions=positions,
            seq_lens=seq_lens,
            draft_tokens=draft_tokens,
            target_predict=target_predict,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

        accept_lens = prepared.accept_lens
        verified_id = prepared.verified_id
        predict = prepared.predict
        new_seq_lens = prepared.new_seq_lens
        selected_hidden = prepared.hidden_states
        selected_positions = prepared.positions
        target_logits_out = logits if return_target_logits else None
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            accept_lens = jax.sharding.reshard(accept_lens, rep)
            verified_id = jax.sharding.reshard(verified_id, rep)
            predict = jax.sharding.reshard(predict, rep)
            new_seq_lens = jax.sharding.reshard(new_seq_lens, rep)
            selected_hidden = jax.sharding.reshard(selected_hidden, rep)
            selected_positions = jax.sharding.reshard(selected_positions, rep)
            if return_target_logits:
                target_logits_out = jax.sharding.reshard(target_logits_out, rep)

        return (
            accept_lens,
            verified_id,
            predict,
            selected_hidden,
            selected_positions,
            new_seq_lens,
            target_pool_updates,
            target_logits_out,
        )

    return fused_target_verify_greedy


def fused_target_verify_greedy_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    """Host entry: fused target verify forward + greedy chain accept (topk=1).

    Drop-in replacement for ``BaseSpecWorker.verify`` on the greedy topk=1 path.
    Does NOT run draft_extend (caller still invokes it afterwards), so the
    target<->draft overlap boundary is preserved.
    """
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput

    target_worker = spec_worker.target_worker
    target_mr = target_worker.model_runner

    spec_info: EagleVerifyInput = model_worker_batch.spec_info_padded
    spec_info.allocate_lens = cur_allocate_lens
    spec_info.prepare_for_verify(model_worker_batch, spec_worker.page_size, target_worker)
    target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(
        model_worker_batch
    )

    return_target_logits = bool(
        getattr(model_worker_batch, "return_logprob", False)
        or getattr(model_worker_batch, "return_output_logprob_only", False)
    )

    target_forward_batch = _forward_batch_init_new_preserve_device(model_worker_batch, target_mr)
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _logits_metadata_from_model_worker_batch_preserve_device(
        model_worker_batch, spec_worker.mesh
    )

    if not hasattr(spec_worker, "_fused_target_verify_jit_fn"):
        spec_worker._fused_target_verify_jit_fn = _build_fused_target_verify_greedy_jit(
            spec_worker.speculative_num_steps,
            spec_worker.speculative_num_draft_tokens,
        )

    with jax.set_mesh(spec_worker.mesh):
        (
            accept_lens,
            verified_id,
            predict,
            selected_hidden,
            selected_positions,
            new_seq_lens,
            target_pool_updates,
            target_logits_out,
        ) = spec_worker._fused_target_verify_jit_fn(
            target_mr._model_def,
            target_mr._model_state_def,
            tuple(target_mr.model_state_leaves),
            target_forward_batch,
            target_mr.memory_pools,
            target_logits_metadata,
            speculative_num_steps=spec_worker.speculative_num_steps,
            speculative_num_draft_tokens=spec_worker.speculative_num_draft_tokens,
            return_target_logits=return_target_logits,
        )

    target_mr.memory_pools.replace_all(target_pool_updates)

    # Mirror eager verify's mutations: gathered positions feed draft_extend
    # (prepare_for_extend_after_verify keeps model_worker_batch.positions as-is).
    model_worker_batch.positions = selected_positions

    next_draft_input = EagleDraftInput(
        verified_id=verified_id,
        new_seq_lens=new_seq_lens,
        allocate_lens=cur_allocate_lens,
        hidden_states=selected_hidden,
    )
    model_worker_batch.spec_info_padded = next_draft_input
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=target_logits_out,
            hidden_states=selected_hidden,
        ),
        next_token_ids=predict,
        next_draft_input=next_draft_input,
        accept_lens=accept_lens,
        allocate_lens=cur_allocate_lens,
        bid=model_worker_batch.bid,
        cache_miss_count=0,
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
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


def _prepare_topk1_verify_placeholders_from_draft_state(draft_worker, model_worker_batch):
    """Prepare fixed-shape verify placeholders while keeping chain build inside JIT."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    draft_worker.padding_for_decode(model_worker_batch)
    draft_input = model_worker_batch.spec_info_padded
    previous_verified_id = draft_input.verified_id
    if isinstance(previous_verified_id, np.ndarray):
        previous_verified_id = np.asarray(previous_verified_id, dtype=np.int32)
    previous_token_list = draft_input.topk_index[:, :, 0]
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
    batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
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
        batch_output.accept_lens = accept_lens
        batch_output.next_token_ids = predict
        return batch_output


def spec_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run one fused speculative decode round."""
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
    return_target_hidden = bool(getattr(model_worker_batch, "return_hidden_states", False))

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

    hidden_size = target_worker.model_config.hidden_size
    placeholder_hidden = np.zeros((spec_info.draft_token.shape[0], hidden_size), dtype=np.float32)
    placeholder_logits = np.zeros((spec_info.draft_token.shape[0], 1), dtype=np.float32)

    next_draft_input = EagleDraftInput(
        verified_id=spec_info.draft_token,
        new_seq_lens=model_worker_batch.seq_lens,
        allocate_lens=cur_allocate_lens,
        hidden_states=placeholder_hidden,
    )
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
        allocate_lens=cur_allocate_lens,
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
