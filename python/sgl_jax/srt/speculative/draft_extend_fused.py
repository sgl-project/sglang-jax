"""Fused greedy speculative decode and MTP draft extend."""

from __future__ import annotations

import copy
import logging
import os
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
from sgl_jax.srt.sampling.sampling_params import TOP_K_ALL
from sgl_jax.srt.speculative.base_worker import populate_speculative_output_logprobs
from sgl_jax.srt.speculative.relay_buffer import (
    gather_spec_relay_buffers,
    make_dp_valid_mask,
    update_spec_relay_buffers,
)

logger = logging.getLogger(__name__)

_TARGET_VERIFY_DECODE_LOOP_ENV = "SGL_JAX_TARGET_VERIFY_DECODE_LOOP"
_TARGET_VERIFY_MODES = {"auto", "batched", "decode-loop"}
_STEP3P5_DECODE_LOOP_WARNED_ENV = "SGL_JAX_STEP3P5_DECODE_LOOP_VERIFY_WARNED"


def _host_array_for_decode_loop(value):
    if value is None:
        return None
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if value.is_fully_replicated:
            value = value.addressable_data(0)
        else:
            from jax.experimental.multihost_utils import process_allgather

            value = process_allgather(value, tiled=True)
    return np.asarray(jax.device_get(value))


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
    draft_extend_hidden_states: jax.Array
    draft_extend_positions: jax.Array
    draft_extend_verified_id: jax.Array
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


def _target_verify_mode_from_env_or_args(server_args) -> str:
    override = os.environ.get(_TARGET_VERIFY_DECODE_LOOP_ENV)
    if override is not None:
        return "decode-loop" if override == "1" else "batched"
    mode = getattr(server_args, "speculative_target_verify_mode", "auto")
    if mode not in _TARGET_VERIFY_MODES:
        raise ValueError(
            "speculative_target_verify_mode must be one of "
            f"{sorted(_TARGET_VERIFY_MODES)}, got {mode!r}"
        )
    return mode


def _is_nextn_algorithm(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        value = SpeculativeAlgorithm.from_string(value)
    return bool(value.is_nextn())


def _get_hf_config_from_worker(worker):
    model_config = getattr(worker, "model_config", None)
    if model_config is None:
        model_runner = getattr(worker, "model_runner", None)
        model_config = getattr(model_runner, "model_config", None)
    return getattr(model_config, "hf_config", None)


def _is_step3p5_target_model(spec_worker) -> bool:
    hf_config = _get_hf_config_from_worker(spec_worker.target_worker)
    return _is_step3p5_hf_config(hf_config)


def _is_step3p5_hf_config(hf_config) -> bool:
    if hf_config is None:
        return False
    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower() == "step3p5":
        return True
    architectures = getattr(hf_config, "architectures", []) or []
    return "Step3p5ForCausalLM" in architectures


def _requires_non_overlap_target_verify(server_args, algorithm, hf_config) -> bool:
    mode = _target_verify_mode_from_env_or_args(server_args)
    if mode == "batched" or not _is_nextn_algorithm(algorithm):
        return False
    if getattr(server_args, "speculative_eagle_topk", 1) != 1:
        return False
    if mode == "decode-loop":
        return True
    return _is_step3p5_hf_config(hf_config)


def _warn_step3p5_decode_loop_once():
    if os.environ.get(_STEP3P5_DECODE_LOOP_WARNED_ENV) == "1":
        return
    os.environ[_STEP3P5_DECODE_LOOP_WARNED_ENV] = "1"
    logger.warning(
        "Step3p5 NEXTN greedy decoding is using correctness-preserving "
        "decode-loop verify in auto mode. This path preserves greedy "
        "equivalence but is slower than batched verify; use "
        "--speculative-target-verify-mode=batched only for performance "
        "experiments that can tolerate the known Step3p5 correctness risk."
    )


def _should_use_decode_loop_target_verify(
    *,
    spec_worker,
    draft_worker,
    model_worker_batch,
    is_greedy: bool,
    use_relay_state: bool,
) -> bool:
    mode = _target_verify_mode_from_env_or_args(spec_worker.server_args)
    if mode == "batched":
        return False
    algorithm = getattr(spec_worker, "speculative_algorithm", None)
    if algorithm is None:
        algorithm = getattr(model_worker_batch, "spec_algorithm", None)
    supported = _is_nextn_algorithm(algorithm) and is_greedy and draft_worker.topk == 1
    if mode == "decode-loop":
        if supported and use_relay_state:
            raise RuntimeError("decode-loop target verify cannot run with speculative relay state")
        return supported
    use_step3p5_fallback = supported and _is_step3p5_target_model(spec_worker)
    if use_step3p5_fallback and use_relay_state:
        raise RuntimeError("Step3p5 auto target verify cannot run with speculative relay state")
    if use_step3p5_fallback:
        _warn_step3p5_decode_loop_once()
    return use_step3p5_fallback


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
    preserve_gather_sharding=True,
    gather_out_sharding=None,
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
    if gather_out_sharding is not None:
        gathered_hidden = hidden_states.at[safe_index, :].get(out_sharding=gather_out_sharding)
    elif preserve_gather_sharding and isinstance(hidden_sharding, NamedSharding):
        gathered_hidden = hidden_states.at[safe_index, :].get(out_sharding=hidden_sharding)
    else:
        gathered_hidden = hidden_states[safe_index, :]
    if gather_out_sharding is not None:
        gathered_positions = positions.at[safe_index].get(out_sharding=gather_out_sharding)
    elif preserve_gather_sharding and isinstance(positions_sharding, NamedSharding):
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


def _verify_greedy(
    *,
    target_hidden,
    positions,
    seq_lens,
    draft_tokens,
    target_predict,
    speculative_num_steps,
    speculative_num_draft_tokens,
    preserve_gather_sharding=True,
    gather_out_sharding=None,
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
    prepared = _prepare_draft_inputs(
        target_hidden,
        positions,
        seq_lens,
        accept_index,
        accept_length,
        verified_id,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        preserve_gather_sharding=preserve_gather_sharding,
        gather_out_sharding=gather_out_sharding,
    )
    return GreedySampleAndPrepareOutput(
        hidden_states=prepared.hidden_states,
        positions=prepared.positions,
        draft_extend_hidden_states=target_hidden,
        draft_extend_positions=positions,
        draft_extend_verified_id=predict,
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
):
    """Non-greedy counterpart of the greedy chain verify.

    Mirrors `tree_speculative_sampling_target_only` (eagle_util.py) for the
    pure topk=1 chain: target-only typical acceptance. Accepted slots emit the
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
    # Everything is replicated here, so the renorm kernels behave exactly like
    # the non-overlap reference path (eagle_util.sample) when enabled.
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
        draft_extend_hidden_states=target_hidden,
        draft_extend_positions=positions,
        draft_extend_verified_id=predict,
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


def _rotate_input_ids(input_ids, ext_lens, sel_pos, new_tokens):
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


def _rotate_prefill_input_ids(input_ids, extend_seq_lens, verified_id, dp_size, per_dp_bs):
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


def _gather_rows_preserve_sharding(values, index, *, out_sharding=None):
    if out_sharding is not None:
        return values.at[index, :].get(out_sharding=out_sharding)
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding):
        return values.at[index, :].get(out_sharding=sharding)
    return values[index, :]


def _reshard_values(sharding, *values):
    return tuple(jax.sharding.reshard(value, sharding) for value in values)


def _device_put_values(sharding, *values):
    return tuple(jax.device_put(value, sharding) for value in values)


def _decode_loop_output_shardings(mesh):
    return NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))


def _topk1_index_from_logits(logits):
    topk_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
    return topk_idx


def _build_draft_extend(num_layers: int, topk: int, chain_mtp: bool = False):
    """Build the fused JIT. Called once, result cached on draft_worker.

    ``chain_mtp`` (chain-style MTP, e.g. Step-3.5-Flash): feed each layer's
    pre-norm hidden (captured as ``output.hidden_states``) into the next layer
    instead of the constant target hidden. Baked in at build time as a
    compile-time branch; non-chain models keep the target hidden every layer.
    """
    assert topk == 1, "Fused draft extend only supports topk=1"

    @partial(
        jax.jit,
        donate_argnames=["all_memory_pools"],
        static_argnames=["model_state_def", "num_layers", "update_relay", "dp_size"],
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
        draft_logits_indices,
        relay_buffers,
        relay_future_indices,
        relay_valid_mask,
        relay_verified_id,
        relay_new_seq_lens,
        draft_verify_seq_lens,
        draft_allocate_lens,
        *,
        num_layers,
        update_relay,
        dp_size,
    ):
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None
        input_ids = forward_batch.input_ids
        if draft_verify_seq_lens is not None:
            valid_draft_slots = draft_verify_seq_lens > 0
            forward_batch.seq_lens = jnp.where(
                valid_draft_slots,
                draft_verify_seq_lens + num_layers,
                jnp.zeros_like(draft_verify_seq_lens),
            )
            forward_batch.attn_backend.forward_metadata = _make_draft_extend_metadata(
                forward_batch.attn_backend.forward_metadata,
                forward_batch.seq_lens,
                draft_allocate_lens,
                page_size=forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )

        chained_hidden = target_hidden
        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[i])
            model = nnx.merge(model_def, state)

            forward_batch.spec_info.hidden_states = chained_hidden
            forward_batch.input_ids = input_ids

            output, pool_updates, _, _ = model(forward_batch, all_memory_pools[i], logits_metadata)
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else None

            if i == 0:
                layer0_hidden = output.hidden_states

            # Chain-style MTP: layer i+1 consumes layer i's pre-norm hidden
            # (captured as output.hidden_states via aux_hidden_states).
            if chain_mtp and output.hidden_states is not None:
                chained_hidden = output.hidden_states

            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            all_topk_index.append(topk_idx)

            if i < num_layers - 1:
                ext_lens = forward_batch.extend_seq_lens
                input_ids = _rotate_input_ids(input_ids, ext_lens, sel_pos, topk_idx[:, 0])

        last_idx = draft_logits_indices
        if logits_metadata.accept_lens is not None:
            last_idx = last_idx - (forward_batch.extend_seq_lens - logits_metadata.accept_lens)
            last_idx = jnp.where(forward_batch.extend_seq_lens > 0, last_idx, 0)
        if dp_size > 1:
            per_dp_tokens = layer0_hidden.shape[0] // dp_size
            per_dp_bs = last_idx.shape[0] // dp_size
            rank_ids = jnp.arange(last_idx.shape[0], dtype=jnp.int32) // per_dp_bs
            last_idx = last_idx + rank_ids * per_dp_tokens
        selected_layer0_hidden = _gather_rows_preserve_sharding(layer0_hidden, last_idx)
        if topk == 1:
            stacked_idx = jnp.stack([idx[:, 0] for idx in all_topk_index], axis=1)
        else:
            stacked_idx = jnp.stack(all_topk_index, axis=1)

        relay_hidden = selected_layer0_hidden
        relay_topk_index = stacked_idx
        relay_verified_id_for_update = relay_verified_id

        # Force P() replicated sharding only on outputs that may still be
        # materialized on host by the debug/legacy restore path. Relay buffers
        # are DP-local and must be updated with the original data-sharded values.
        if mesh is not None:
            rep = NamedSharding(mesh, P())
            selected_layer0_hidden = jax.sharding.reshard(selected_layer0_hidden, rep)
            stacked_idx = jax.sharding.reshard(stacked_idx, rep)

        updated_relay_buffers = relay_buffers
        if update_relay:
            updated_relay_buffers = update_spec_relay_buffers(
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                relay_topk_index,
                relay_hidden,
                relay_verified_id_for_update,
                relay_new_seq_lens,
                dp_size=dp_size,
            )

        return (
            selected_layer0_hidden,
            stacked_idx,
            tuple(all_pool_updates),
            updated_relay_buffers,
        )

    return fused_draft_extend


def _per_dp_cumsum_device(lens, dp_size: int):
    per_dp_bs = lens.shape[0] // dp_size
    lens_2d = lens.reshape((dp_size, per_dp_bs))
    zeros = jnp.zeros_like(lens_2d[:, :1], dtype=jnp.int32)
    return jnp.concatenate([zeros, jnp.cumsum(lens_2d, axis=1, dtype=jnp.int32)], axis=1).reshape(
        (dp_size * (per_dp_bs + 1),)
    )


def _repack_page_indices(
    page_indices,
    allocated_lens,
    metadata_seq_lens,
    *,
    page_size: int,
    dp_size: int,
):
    total_bs = metadata_seq_lens.shape[0]
    per_dp_bs = total_bs // dp_size
    pages_per_dp = page_indices.shape[0] // dp_size

    allocated_pages = ((allocated_lens + page_size - 1) // page_size).astype(jnp.int32)
    needed_pages = ((metadata_seq_lens + page_size - 1) // page_size).astype(jnp.int32)
    allocated_pages = allocated_pages.reshape((dp_size, per_dp_bs))
    needed_pages = needed_pages.reshape((dp_size, per_dp_bs))

    src_offsets = jnp.cumsum(allocated_pages, axis=1, dtype=jnp.int32) - allocated_pages
    dst_offsets = jnp.cumsum(needed_pages, axis=1, dtype=jnp.int32) - needed_pages

    local_page_ids = jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :, None]
    in_req = (local_page_ids >= dst_offsets[:, None, :]) & (
        local_page_ids < (dst_offsets + needed_pages)[:, None, :]
    )
    slot_ids = jnp.argmax(in_req.astype(jnp.int32), axis=2).astype(jnp.int32)
    valid = jnp.any(in_req, axis=2)

    dp_ids = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    offsets_sharding = jax.typeof(src_offsets).sharding
    offsets_out_sharding = offsets_sharding if isinstance(offsets_sharding, NamedSharding) else None
    src_slot_offsets = src_offsets.at[dp_ids, slot_ids].get(out_sharding=offsets_out_sharding)
    dst_slot_offsets = dst_offsets.at[dp_ids, slot_ids].get(out_sharding=offsets_out_sharding)
    gather_src = (
        dp_ids * pages_per_dp
        + src_slot_offsets
        + (jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :] - dst_slot_offsets)
    )
    page_sharding = jax.typeof(page_indices).sharding
    out_sharding = page_sharding if isinstance(page_sharding, NamedSharding) else None
    gathered = (
        page_indices.at[gather_src.reshape(-1)]
        .get(
            mode="fill",
            fill_value=0,
            out_sharding=out_sharding,
        )
        .reshape((dp_size, pages_per_dp))
    )
    return jnp.where(valid, gathered, jnp.zeros_like(gathered)).reshape(page_indices.shape)


def _make_target_verify_metadata(
    old_metadata,
    verify_seq_lens,
    allocated_lens,
    *,
    speculative_num_draft_tokens: int,
    page_size: int,
    dp_size: int,
):
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    valid = verify_seq_lens > 0
    extend_seq_lens = jnp.where(
        valid,
        jnp.full_like(verify_seq_lens, speculative_num_draft_tokens),
        jnp.zeros_like(verify_seq_lens),
    )
    cu_q_lens = _per_dp_cumsum_device(extend_seq_lens, dp_size)
    metadata_seq_lens = verify_seq_lens + extend_seq_lens
    aligned_seq_lens = ((metadata_seq_lens + page_size - 1) // page_size) * page_size
    cu_kv_lens = _per_dp_cumsum_device(aligned_seq_lens, dp_size)
    page_indices = _repack_page_indices(
        old_metadata.page_indices,
        allocated_lens,
        metadata_seq_lens,
        page_size=page_size,
        dp_size=dp_size,
    )
    swa_page_indices = None
    if old_metadata.swa_page_indices is not None:
        swa_page_indices = _repack_page_indices(
            old_metadata.swa_page_indices,
            allocated_lens,
            metadata_seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    per_dp_bs = verify_seq_lens.shape[0] // dp_size
    local_num_seqs = jnp.sum(valid.reshape((dp_size, per_dp_bs)).astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))

    return FlashAttentionMetadata(
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        swa_page_indices=swa_page_indices,
        seq_lens=metadata_seq_lens,
        distribution=distribution,
        custom_mask=old_metadata.custom_mask,
    )


def _make_draft_extend_metadata(
    old_metadata,
    draft_seq_lens,
    allocated_lens,
    *,
    page_size: int,
    dp_size: int,
):
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    valid = draft_seq_lens > 0
    # DRAFT_EXTEND always runs a fixed number of query tokens per request. The
    # host metadata already has the correct DP-padded query cumsum shape; only
    # seq_lens/page_indices need to be rebuilt from the actual verify base.
    cu_q_lens = old_metadata.cu_q_lens
    aligned_seq_lens = ((draft_seq_lens + page_size - 1) // page_size) * page_size
    cu_kv_lens = _per_dp_cumsum_device(aligned_seq_lens, dp_size)
    page_indices = _repack_page_indices(
        old_metadata.page_indices,
        allocated_lens,
        draft_seq_lens,
        page_size=page_size,
        dp_size=dp_size,
    )
    swa_page_indices = None
    if old_metadata.swa_page_indices is not None:
        swa_page_indices = _repack_page_indices(
            old_metadata.swa_page_indices,
            allocated_lens,
            draft_seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    per_dp_bs = draft_seq_lens.shape[0] // dp_size
    local_num_seqs = jnp.sum(valid.reshape((dp_size, per_dp_bs)).astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))

    return FlashAttentionMetadata(
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        swa_page_indices=swa_page_indices,
        seq_lens=draft_seq_lens,
        distribution=distribution,
        custom_mask=old_metadata.custom_mask,
    )


def _build_verify(topk: int):
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
            "use_relay_state",
            "dp_size",
            "is_greedy",
            "threshold_single",
            "threshold_acc",
            "enable_top_k",
            "enable_top_p",
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
            target_forward_batch.attn_backend.forward_metadata = _make_target_verify_metadata(
                target_forward_batch.attn_backend.forward_metadata,
                target_forward_batch.seq_lens,
                verify_allocate_lens,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                page_size=target_forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )
            previous_verified_id = relay_verified_id
            previous_token_list = relay_topk_index

        target_bs = target_forward_batch.seq_lens.shape[0]
        (
            draft_tokens,
            positions,
            retrive_index_flat,
            retrive_next_token_flat,
            retrive_next_sibling_flat,
        ) = _build_chain_verify_arrays(
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
        if is_greedy:
            target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(-1)
            prepared = _verify_greedy(
                target_hidden=target_hidden,
                positions=target_forward_batch.positions,
                seq_lens=target_forward_batch.seq_lens,
                draft_tokens=draft_tokens,
                target_predict=target_predict,
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            )
        else:
            # Generate rejection-sampling coins inside the JIT: avoids building them
            # on host and copying (tbs, n-1)+(tbs,) arrays in every step, and keeps
            # the threefry/uniform ops fused into this module instead of becoming
            # standalone eager dispatches (the reason the earlier host-side jax.random
            # attempt was reverted).
            sampling_rng = jax.random.fold_in(sampling_base_rng, sampling_step)
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
        prepared_hidden_for_draft_extend = prepared.draft_extend_hidden_states
        prepared_verified_id_data = prepared.draft_extend_verified_id
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
        prepared_positions = prepared.draft_extend_positions
        prepared_positions_data = prepared.draft_extend_positions
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
                prepared_hidden_for_draft_extend,
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
                prepared_hidden_for_draft_extend,
            )
            if return_target_logits:
                target_logits_for_host = jax.sharding.reshard(target_logits_for_host, rep)

        return (
            target_pool_updates,
            prepared_hidden,
            prepared_hidden_for_draft_extend,
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


def _build_prefill(num_layers: int, topk: int, chain_mtp: bool = False):
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
            "update_relay",
        ],
    )
    def fused_prefill(
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
        relay_buffers,
        relay_future_indices,
        relay_valid_mask,
        *,
        num_layers,
        dp_size,
        per_dp_bs,
        update_relay,
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
        input_ids = _rotate_prefill_input_ids(
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

        chained_hidden = target_hidden
        for i in range(num_layers):
            state = jax.tree_util.tree_unflatten(draft_model_state_def, draft_all_leaves[i])
            model = nnx.merge(draft_model_def, state)

            draft_forward_batch.input_ids = input_ids
            draft_forward_batch.spec_info.hidden_states = chained_hidden
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
            if chain_mtp and output.hidden_states is not None:
                chained_hidden = output.hidden_states
            if i < num_layers - 1:
                input_ids = _rotate_prefill_input_ids(
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
        if topk == 1:
            stacked_idx = jnp.stack([idx[:, 0] for idx in all_topk_index], axis=1)
        else:
            stacked_idx = jnp.stack(all_topk_index, axis=1)
        relay_hidden = selected_layer0_hidden
        relay_topk_index = stacked_idx
        relay_verified_id = next_token_ids
        relay_new_seq_lens = target_forward_batch.seq_lens + 1
        if mesh is not None and not update_relay:
            rep = NamedSharding(mesh, P())
            next_token_ids = jax.sharding.reshard(jnp.copy(next_token_ids), rep)
            selected_layer0_hidden = jax.sharding.reshard(selected_layer0_hidden, rep)
            stacked_idx = jax.sharding.reshard(stacked_idx, rep)

        updated_relay_buffers = relay_buffers
        if update_relay:
            updated_relay_buffers = update_spec_relay_buffers(
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                relay_topk_index,
                relay_hidden,
                relay_verified_id,
                relay_new_seq_lens,
                dp_size=dp_size,
            )

        return (
            target_output,
            next_token_ids,
            target_pool_updates,
            tuple(all_pool_updates),
            selected_layer0_hidden,
            stacked_idx,
            updated_relay_buffers,
        )

    return fused_prefill


def _prepare_verify(draft_worker, model_worker_batch):
    """Prepare fixed-shape verify placeholders while keeping chain build inside JIT."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.speculative.eagle_util import EagleVerifyInput

    draft_input = model_worker_batch.spec_info_padded
    use_relay_state = (
        getattr(draft_input, "future_indices", None) is not None
        and getattr(draft_input, "topk_index", None) is None
    )
    if use_relay_state:
        bs = len(model_worker_batch.seq_lens)
        draft_input.verified_id = np.zeros((bs,), dtype=np.int32)
        draft_input.topk_p = np.ones(
            (bs, draft_worker.speculative_num_steps),
            dtype=np.float32,
        )
        draft_input.topk_index = np.zeros(
            (bs, draft_worker.speculative_num_steps),
            dtype=np.int32,
        )
        draft_input.hidden_states = np.zeros(
            (bs, draft_worker.model_config.hidden_size),
            dtype=np.float32,
        )

    draft_worker.padding_for_decode(model_worker_batch)
    draft_input = model_worker_batch.spec_info_padded
    previous_verified_id = draft_input.verified_id
    if isinstance(previous_verified_id, np.ndarray):
        previous_verified_id = np.asarray(previous_verified_id, dtype=np.int32)
    topk_index = draft_input.topk_index
    if len(topk_index.shape) == 2:
        previous_token_list = topk_index
    elif len(topk_index.shape) == 3 and topk_index.shape[-1] == 1:
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


def prepare_forward_batch_for_prefill(spec_worker, model_worker_batch):
    """Prepare the target ForwardBatch before speculative prefill is queued."""
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

    target_mr = spec_worker.target_worker.model_runner
    model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
    target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_forward_metadata(
        model_worker_batch
    )
    model_worker_batch.forward_batch = _make_forward_batch(model_worker_batch, target_mr)
    model_worker_batch.forward_batch.bid = model_worker_batch.bid
    return model_worker_batch.forward_batch


def launch_fused_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
    *,
    relay_buffers=None,
    relay_future_indices=None,
    relay_valid_mask=None,
):
    """Launch fused MTP draft extend and return deferred host restore state."""
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None
    target_hidden = getattr(
        batch_output.next_draft_input,
        "hidden_states_for_draft_extend",
        batch_output.logits_output.hidden_states,
    )
    update_relay = relay_buffers is not None

    draft_input = EagleDraftInput(
        hidden_states=target_hidden,
        allocate_lens=batch_output.next_draft_input.allocate_lens,
        accept_length=getattr(batch_output.next_draft_input, "accept_length", None),
    )
    draft_input.verified_id_for_draft_extend = getattr(
        batch_output.next_draft_input, "verified_id_for_draft_extend", None
    )
    draft_input.extend_seq_lens_for_draft_extend = getattr(
        batch_output.next_draft_input, "extend_seq_lens_for_draft_extend", None
    )
    draft_input.logits_indices_for_draft_extend = getattr(
        batch_output.next_draft_input, "logits_indices_for_draft_extend", None
    )
    draft_input.positions_for_draft_extend = getattr(
        batch_output.next_draft_input, "positions_for_draft_extend", None
    )
    draft_input.sel_pos_for_draft_extend = getattr(
        batch_output.next_draft_input, "sel_pos_for_draft_extend", None
    )
    draft_input.allocate_lens_for_draft_extend = getattr(
        batch_output.next_draft_input, "allocate_lens_for_draft_extend", None
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

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    sel_pos_for_draft_extend = getattr(
        batch_output.next_draft_input, "sel_pos_for_draft_extend", None
    )
    if sel_pos_for_draft_extend is not None:
        sel_pos = sel_pos_for_draft_extend
    elif hasattr(batch_output.next_draft_input, "sel_pos"):
        sel_pos = batch_output.next_draft_input.sel_pos
    else:
        sel_pos = jnp.clip(batch_output.accept_lens - 1, 0, None).astype(jnp.int32)

    mr0 = draft_worker._workers[0].model_runner
    mwb.spec_info_padded.hidden_states = target_hidden
    shared_fb = _make_forward_batch(mwb, mr0)
    shared_fb.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    sel_pos_device = _prepare_device_array(sel_pos, data_sharding, "draft_extend.sel_pos")
    draft_logits_indices = _prepare_device_array(
        (
            getattr(mwb.spec_info_padded, "logits_indices_for_draft_extend", None)
            if getattr(mwb.spec_info_padded, "logits_indices_for_draft_extend", None) is not None
            else mwb.logits_indices
        ),
        data_sharding,
        "draft_extend.logits_indices",
    )
    draft_allocate_lens = getattr(
        batch_output.next_draft_input, "allocate_lens_for_draft_extend", None
    )
    if draft_allocate_lens is None:
        draft_allocate_lens = np.zeros_like(model_worker_batch.seq_lens, dtype=np.int32)
        draft_allocate_lens[sel] = np.asarray(batch_output.next_draft_input.allocate_lens)
    draft_allocate_lens = _prepare_device_array(
        draft_allocate_lens, data_sharding, "draft_extend.allocate_lens"
    )
    draft_verify_seq_lens = getattr(batch_output.next_draft_input, "verify_seq_lens", None)
    draft_verify_seq_lens = _prepare_device_array(
        draft_verify_seq_lens, data_sharding, "draft_extend.verify_seq_lens"
    )
    if relay_future_indices is None:
        relay_future_indices = np.zeros(model_worker_batch.req_pool_indices.shape, dtype=np.int32)
    if relay_valid_mask is None:
        relay_valid_mask = np.zeros(model_worker_batch.req_pool_indices.shape, dtype=np.bool_)
    relay_future_indices = _prepare_device_array(
        relay_future_indices, data_sharding, "draft_extend.relay_future_indices"
    )
    relay_valid_mask = _prepare_device_array(
        relay_valid_mask, data_sharding, "draft_extend.relay_valid_mask"
    )
    if not hasattr(draft_worker, "_fused_jit_fn"):
        draft_worker._fused_jit_fn = _build_draft_extend(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
            chain_mtp=getattr(draft_worker, "chain_mtp_hidden_states", False),
        )

    with jax.set_mesh(draft_worker.mesh):
        (
            selected_layer0_hidden,
            topk_index_stacked,
            all_pool_updates,
            updated_relay_buffers,
        ) = draft_worker._fused_jit_fn(
            mr0._model_def,
            mr0._model_state_def,
            tuple(all_leaves),
            shared_fb,
            tuple(all_memory_pools),
            logits_metadata,
            target_hidden,
            sel_pos_device,
            draft_logits_indices,
            relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            batch_output.next_draft_input.next_verified_id,
            batch_output.next_draft_input.new_seq_lens,
            draft_verify_seq_lens,
            draft_allocate_lens,
            num_layers=draft_worker.speculative_num_steps,
            update_relay=update_relay,
            dp_size=model_worker_batch.dp_size,
        )

    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])

    return FusedDraftExtendPendingResult(
        batch_output=batch_output,
        selected_layer0_hidden=selected_layer0_hidden,
        topk_index_stacked=topk_index_stacked,
        next_verified_id=batch_output.next_draft_input.next_verified_id,
        accept_lens=batch_output.accept_lens,
        sel=sel,
        updated_relay_buffers=updated_relay_buffers,
    )


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
    jax.copy_to_host_async(next_verified_id)

    batch_output.next_draft_input.hidden_states = np.asarray(selected_layer0_hidden)[sel]
    topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.topk_p = np.ones(topk_index.shape, dtype=np.float32)
    batch_output.next_draft_input.topk_index = topk_index
    batch_output.next_draft_input.verified_id = np.asarray(next_verified_id)[sel]
    batch_output.next_draft_input.hidden_states_for_draft_extend = None
    batch_output.next_draft_input.allocate_lens = batch_output.next_draft_input.allocate_lens[
        : model_worker_batch.real_bs
    ]
    batch_output.accept_lens = accept_host


def draft_extend_for_decode(draft_worker, model_worker_batch, batch_output):
    """Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.

    Fuses all N MTP layer forwards into a single jit call.
    """
    pending_result = launch_fused_draft_extend_for_decode(
        draft_worker, model_worker_batch, batch_output
    )
    restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def spec_prefill(spec_worker, model_worker_batch, launch_done=None, *, update_relay=False):
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
        target_forward_batch = prepare_forward_batch_for_prefill(spec_worker, model_worker_batch)
    else:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_forward_metadata(
            model_worker_batch
        )
        target_forward_batch = model_worker_batch.forward_batch
        target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _prepare_logits_metadata(model_worker_batch, spec_worker.mesh)

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
    draft_logits_indices = _prepare_device_array(
        model_worker_batch.logits_indices,
        NamedSharding(draft_worker.mesh, P("data")),
        "prefill.logits_indices",
    )
    draft_logits_metadata = _prepare_logits_metadata(model_worker_batch, draft_worker.mesh)

    all_memory_pools = []
    all_leaves = []
    for w in draft_worker._workers:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_greedy_prefill_jit_fn"):
        draft_worker._fused_greedy_prefill_jit_fn = _build_prefill(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
            chain_mtp=getattr(draft_worker, "chain_mtp_hidden_states", False),
        )

    data_sharding = NamedSharding(draft_worker.mesh, P("data"))
    relay_buffers = getattr(spec_worker, "spec_relay_buffers", None)
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
    relay_future_indices = _prepare_device_array(
        safe_indices, data_sharding, "prefill.relay_future_indices"
    )
    relay_valid_mask = _prepare_device_array(valid_mask, data_sharding, "prefill.relay_valid_mask")

    with jax.set_mesh(draft_worker.mesh), _count_pjit_cpp_cache_miss() as count:
        (
            logits_output,
            next_token_ids,
            target_pool_updates,
            all_pool_updates,
            layer0_hidden,
            topk_index_stacked,
            updated_relay_buffers,
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
            relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            num_layers=draft_worker.speculative_num_steps,
            dp_size=model_worker_batch.dp_size,
            per_dp_bs=model_worker_batch.per_dp_bs_size,
            update_relay=update_relay,
        )
        cache_miss_count = count()
    prefill_output_token_ids = None
    if update_relay:
        prefill_output_token_ids = _prepare_spec_prefill_output_token_ids(
            draft_worker,
            next_token_ids,
        )
        if hasattr(prefill_output_token_ids, "copy_to_host_async"):
            prefill_output_token_ids.copy_to_host_async()

    if launch_done is not None:
        launch_done.set()

    target_mr.memory_pools.replace_all(target_pool_updates)
    for i, w in enumerate(draft_worker._workers):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])
    if update_relay:
        spec_worker.spec_relay_buffers = updated_relay_buffers

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    if update_relay:
        from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

        future_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)[sel]
        model_worker_batch.spec_info_padded = EagleDraftInput(
            future_indices=future_indices,
            allocate_lens=np.asarray(model_worker_batch.seq_lens, dtype=np.int32)[sel],
            capture_hidden_mode=CaptureHiddenMode.FULL,
            num_tokens_per_batch=np.asarray(1, dtype=np.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=prefill_output_token_ids,
            next_draft_input=model_worker_batch.spec_info_padded,
            spec_relay_buffers=updated_relay_buffers,
            prefill_relay_future_indices=relay_future_indices,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    relay_next_token_ids = next_token_ids
    host_next_token_ids = next_token_ids
    if model_worker_batch.dp_size > 1:
        from jax.experimental.multihost_utils import process_allgather

        host_next_token_ids = process_allgather(host_next_token_ids, tiled=True)

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


def spec_prefill_overlap(spec_worker, model_worker_batch):
    return spec_prefill(spec_worker, model_worker_batch, update_relay=True)


def _build_decode_loop_cache_loc(
    req_to_token_pool,
    req_pool_indices: np.ndarray,
    seq_lens: np.ndarray,
    *,
    dp_size: int,
    per_dp_bs: int,
    page_size: int,
    total_cache_loc_size: int,
) -> np.ndarray:
    per_rank_sizes = []
    for r in range(dp_size):
        start = r * per_dp_bs
        end = start + per_dp_bs
        aligned = ((seq_lens[start:end] + page_size - 1) // page_size) * page_size
        per_rank_sizes.append(int(np.sum(np.where(seq_lens[start:end] > 0, aligned, 0))))
    if total_cache_loc_size % dp_size != 0:
        raise ValueError(
            f"cache_loc size {total_cache_loc_size} is not divisible by dp_size {dp_size}"
        )
    per_dp_cache_loc_size = total_cache_loc_size // dp_size
    required_size = max(max(per_rank_sizes, default=0), per_dp_bs * page_size)
    if required_size > per_dp_cache_loc_size:
        raise ValueError(
            "decode-loop cache_loc bucket is too small: "
            f"required_per_dp={required_size}, available_per_dp={per_dp_cache_loc_size}"
        )
    cache_loc = np.zeros(total_cache_loc_size, dtype=np.int32)
    req_to_token = req_to_token_pool.req_to_token
    for r in range(dp_size):
        dst = r * per_dp_cache_loc_size
        for j in range(per_dp_bs):
            slot = r * per_dp_bs + j
            seq_len = int(seq_lens[slot])
            req_idx = int(req_pool_indices[slot])
            if seq_len <= 0 or req_idx < 0:
                continue
            aligned = ((seq_len + page_size - 1) // page_size) * page_size
            cache_loc[dst : dst + seq_len] = req_to_token[req_idx, :seq_len]
            dst += aligned
    return cache_loc


def _build_decode_loop_out_cache_loc(
    req_to_token_pool,
    req_pool_indices: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    out_cache_loc = np.full(positions.shape, -1, dtype=np.int32)
    req_to_token = req_to_token_pool.req_to_token
    for i, is_valid in enumerate(valid):
        req_idx = int(req_pool_indices[i])
        pos = int(positions[i])
        if not is_valid or req_idx < 0 or pos < 0:
            continue
        out_cache_loc[i] = req_to_token[req_idx, pos]
    return out_cache_loc


def _decode_loop_target_verify(
    *,
    spec_worker,
    model_worker_batch,
    target_mr,
    previous_verified_id,
    previous_token_list,
    verify_allocate_lens,
    return_target_logits,
) -> tuple:
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    if os.environ.get("SGL_JAX_TARGET_VERIFY_DECODE_LOOP_LOGGED") != "1":
        os.environ["SGL_JAX_TARGET_VERIFY_DECODE_LOOP_LOGGED"] = "1"
        logger.info("[VERIFY_DECODE_LOOP] target verify decode-loop path active")

    req_to_token_pool, _ = spec_worker.target_worker.get_memory_pool()
    bs = model_worker_batch.seq_lens.shape[0]
    n = spec_worker.draft_worker.speculative_num_draft_tokens
    dp_size = model_worker_batch.dp_size
    per_dp_bs = model_worker_batch.per_dp_bs_size if dp_size > 1 else bs

    prev_verified_host = _host_array_for_decode_loop(previous_verified_id).astype(np.int32)
    prev_tokens_host = _host_array_for_decode_loop(previous_token_list).astype(np.int32)
    seq_lens_base = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)
    valid = seq_lens_base > 0
    draft_tokens_2d = np.concatenate(
        [prev_verified_host.reshape(bs, 1), prev_tokens_host.reshape(bs, -1)[:, : n - 1]],
        axis=1,
    ).astype(np.int32)
    positions_2d = (
        seq_lens_base.reshape(bs, 1) + np.arange(n, dtype=np.int32).reshape(1, n)
    ).astype(np.int32)

    logits_per_step = []
    hidden_per_step = []
    cache_miss_count = 0
    for step in range(n):
        step_seq_lens = np.where(valid, seq_lens_base + step + 1, 0).astype(np.int32)
        step_positions = np.where(valid, positions_2d[:, step], 0).astype(np.int32)
        step_input_ids = np.where(valid, draft_tokens_2d[:, step], 0).astype(np.int32)
        step_out_cache_loc = _build_decode_loop_out_cache_loc(
            req_to_token_pool,
            np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
            step_positions,
            valid,
        )
        step_cache_loc = _build_decode_loop_cache_loc(
            req_to_token_pool,
            np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
            step_seq_lens,
            dp_size=dp_size,
            per_dp_bs=per_dp_bs,
            page_size=spec_worker.page_size,
            total_cache_loc_size=model_worker_batch.cache_loc.shape[0],
        )

        decode_batch = copy.copy(model_worker_batch)
        decode_batch.forward_mode = ForwardMode.DECODE
        decode_batch.input_ids = step_input_ids
        decode_batch.positions = step_positions
        decode_batch.seq_lens = step_seq_lens
        decode_batch.out_cache_loc = step_out_cache_loc
        decode_batch.cache_loc = step_cache_loc
        decode_batch.extend_prefix_lens = None
        decode_batch.extend_seq_lens = None
        decode_batch.logits_indices = None
        decode_batch.spec_info_padded = None
        decode_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_forward_metadata(
            decode_batch
        )
        decode_forward_batch = _make_forward_batch(decode_batch, target_mr)
        decode_forward_batch.bid = model_worker_batch.bid
        decode_logits_metadata = _prepare_logits_metadata(
            decode_batch, spec_worker.mesh, include_accept_lens=False
        )
        logits_output, step_cache_miss_count, _ = target_mr.forward(
            decode_forward_batch, decode_logits_metadata
        )
        cache_miss_count += step_cache_miss_count
        logits_per_step.append(logits_output.next_token_logits)
        hidden_per_step.append(logits_output.hidden_states)

    with jax.set_mesh(spec_worker.mesh):
        rep_sharding = NamedSharding(spec_worker.mesh, P())
        target_logits = jnp.stack(logits_per_step, axis=1).reshape(bs * n, -1)
        target_hidden = jnp.stack(hidden_per_step, axis=1).reshape(bs * n, -1)
        draft_tokens = jnp.asarray(draft_tokens_2d.reshape(bs * n), dtype=jnp.int32)
        positions = jnp.asarray(positions_2d.reshape(bs * n), dtype=jnp.int32)
        target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(-1)
        prepared = _verify_greedy(
            target_hidden=target_hidden,
            positions=positions,
            seq_lens=jnp.asarray(seq_lens_base, dtype=jnp.int32),
            draft_tokens=draft_tokens,
            target_predict=target_predict,
            speculative_num_steps=spec_worker.draft_worker.speculative_num_steps,
            speculative_num_draft_tokens=n,
            preserve_gather_sharding=False,
            gather_out_sharding=rep_sharding,
        )

        target_logits_for_host = (
            _gather_rows_preserve_sharding(
                target_logits,
                prepared.safe_index,
                out_sharding=rep_sharding,
            )
            if return_target_logits
            else None
        )
        prepared_hidden = prepared.hidden_states
        prepared_verified_id = prepared.verified_id
        prepared_hidden_for_draft_extend = target_hidden
        prepared_verified_id_data = target_predict
        prepared_next_verified_id = _take_with_index_sharding(
            prepared.verified_id, prepared.select_index
        )
        prepared_new_seq_lens = prepared.new_seq_lens
        prepared_accept_lens_host = prepared.accept_lens
        prepared_accept_lens_data = prepared.accept_lens
        prepared_extend_seq_lens = jnp.where(
            jnp.asarray(seq_lens_base, dtype=jnp.int32) > 0,
            jnp.full((bs,), n, dtype=jnp.int32),
            jnp.zeros((bs,), dtype=jnp.int32),
        )
        prepared_logits_indices = (
            jnp.cumsum(
                prepared_extend_seq_lens.reshape(dp_size, bs // dp_size),
                axis=1,
            ).reshape(-1)
            - 1
        ).astype(jnp.int32)
        prepared_sel_pos = prepared.sel_pos
        prepared_sel_pos_data = prepared.sel_pos
        prepared_predict = prepared.predict
        prepared_positions = prepared.draft_extend_positions
        prepared_positions_data = prepared.draft_extend_positions
        prepared_verify_seq_lens = jnp.asarray(seq_lens_base, dtype=jnp.int32)
        prepared_allocate_lens_data = verify_allocate_lens

    mesh = spec_worker.mesh
    if mesh is not None:
        rep, data = _decode_loop_output_shardings(mesh)
        (
            prepared_hidden,
            prepared_verified_id,
            prepared_new_seq_lens,
            prepared_accept_lens_host,
            prepared_sel_pos,
            prepared_predict,
            prepared_positions,
        ) = _device_put_values(
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
            prepared_hidden_for_draft_extend,
        ) = _device_put_values(
            data,
            prepared_verified_id_data,
            prepared_next_verified_id,
            prepared_accept_lens_data,
            prepared_extend_seq_lens,
            prepared_logits_indices,
            prepared_sel_pos_data,
            prepared_positions_data,
            prepared_allocate_lens_data,
            prepared_hidden_for_draft_extend,
        )
        if return_target_logits:
            target_logits_for_host = jax.device_put(target_logits_for_host, rep)

    return (
        prepared_hidden,
        prepared_hidden_for_draft_extend,
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
        cache_miss_count,
    )


def spec_decode_verify(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run target verify as the first speculative decode JIT."""
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.speculative.eagle_util import EagleDraftInput

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
    previous_verified_id, previous_token_list = _prepare_verify(draft_worker, model_worker_batch)
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
    if use_relay_state and target_mr.attn_backend.forward_metadata.custom_mask is not None:
        raise NotImplementedError("Spec decode overlap relay path does not support custom_mask.")
    target_forward_batch = _make_forward_batch(model_worker_batch, target_mr)
    target_forward_batch.bid = model_worker_batch.bid
    target_logits_metadata = _prepare_logits_metadata(model_worker_batch, spec_worker.mesh)
    data_sharding = NamedSharding(spec_worker.mesh, P("data"))
    if relay_future_indices is None:
        relay_future_indices = np.zeros(model_worker_batch.seq_lens.shape, dtype=np.int32)
    relay_future_indices = _prepare_device_array(
        relay_future_indices, data_sharding, "verify.relay_future_indices"
    )
    verify_allocate_lens = np.zeros_like(model_worker_batch.seq_lens, dtype=np.int32)
    verify_allocate_lens[model_worker_batch.logits_indices_selector] = cur_allocate_lens
    verify_allocate_lens = _prepare_device_array(
        verify_allocate_lens, data_sharding, "verify.allocate_lens"
    )

    if not hasattr(draft_worker, "_fused_greedy_verify_jit_fn"):
        draft_worker._fused_greedy_verify_jit_fn = _build_verify(
            topk=draft_worker.topk,
        )

    si = model_worker_batch.sampling_info
    _sv_is_greedy = bool(getattr(si, "is_all_greedy", True))
    _sv_tbs = target_forward_batch.seq_lens.shape[0]
    _sv_enable_top_k = False
    _sv_enable_top_p = False
    if _sv_is_greedy:
        _sv_temps = _prepare_device_array(np.ones((_sv_tbs, 1), np.float32), data_sharding)
        _sv_topks = _prepare_device_array(np.full((_sv_tbs,), TOP_K_ALL, np.int32), data_sharding)
        _sv_topps = _prepare_device_array(np.ones((_sv_tbs,), np.float32), data_sharding)
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

    use_decode_loop_verify = _should_use_decode_loop_target_verify(
        spec_worker=spec_worker,
        draft_worker=draft_worker,
        model_worker_batch=model_worker_batch,
        is_greedy=_sv_is_greedy,
        use_relay_state=use_relay_state,
    )
    target_pool_updates = None
    if use_decode_loop_verify:
        (
            prepared_hidden,
            prepared_hidden_for_draft_extend,
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
            cache_miss_count,
        ) = _decode_loop_target_verify(
            spec_worker=spec_worker,
            model_worker_batch=model_worker_batch,
            target_mr=target_mr,
            previous_verified_id=previous_verified_id,
            previous_token_list=previous_token_list,
            verify_allocate_lens=verify_allocate_lens,
            return_target_logits=return_target_logits,
        )
    else:
        with jax.set_mesh(draft_worker.mesh), _count_pjit_cpp_cache_miss() as count:
            (
                target_pool_updates,
                prepared_hidden,
                prepared_hidden_for_draft_extend,
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
            )
            cache_miss_count = count()

    if target_pool_updates is not None:
        target_mr.memory_pools.replace_all(target_pool_updates)

    next_draft_input = EagleDraftInput(
        verified_id=prepared_verified_id,
        new_seq_lens=prepared_new_seq_lens,
        allocate_lens=cur_allocate_lens,
        hidden_states=prepared_hidden,
        accept_length=prepared_accept_lens_data,
    )
    next_draft_input.hidden_states_for_draft_extend = prepared_hidden_for_draft_extend
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


def spec_decode_draft_extend(spec_worker, model_worker_batch, batch_output):
    """Run MTP draft extend as the second speculative decode JIT."""
    spec_worker.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
    return batch_output


def spec_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run speculative decode as verify JIT followed by draft-extend JIT."""
    batch_output = spec_decode_verify(spec_worker, model_worker_batch, cur_allocate_lens)
    return spec_decode_draft_extend(spec_worker, model_worker_batch, batch_output)


def spec_decode_overlap(spec_worker, model_worker_batch, cur_allocate_lens):
    """Launch decode verify and draft-extend without restoring draft results inline."""
    batch_output = spec_decode_verify(spec_worker, model_worker_batch, cur_allocate_lens)
    sel = np.asarray(model_worker_batch.logits_indices_selector)
    batch_output.next_draft_input.future_indices = np.asarray(model_worker_batch.req_pool_indices)[
        sel
    ]

    from sgl_jax.srt.speculative.overlap_utils import publish_spec_decode_new_seq_lens
    from sgl_jax.srt.speculative.relay_buffer import make_dp_valid_mask

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
    pending_result = launch_fused_draft_extend_for_decode(
        spec_worker.draft_worker,
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
