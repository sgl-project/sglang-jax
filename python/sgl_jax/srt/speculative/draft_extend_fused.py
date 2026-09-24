"""Fused greedy speculative decode and MTP draft extend."""

from __future__ import annotations

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
from sgl_jax.srt.speculative.relay_buffer import (
    gather_spec_relay_buffers,
    make_dp_valid_mask,
    update_spec_relay_buffers,
)
from sgl_jax.srt.speculative.spec_utils import (
    SIMULATED_ACCEPTANCE_CONFIG,
    apply_simulated_acceptance,
)
from sgl_jax.srt.utils.common_utils import get_bool_env_var

logger = logging.getLogger(__name__)


# GLM-5.2 KVShare: draft steps >= 1 reuse step 0's selection AND KV; they must not
# write their own KV / indexer keys (opt-in A/B, see dsa_sparse_backend readonly).
_KVSHARE = os.environ.get("SGLANG_JAX_MTP_KVSHARE", "0") == "1"
# Single-layer MTP chain (GLM-5.2: num_nextn_predict_layers=1 applied num_steps times):
# feed draft step j >= 1 the previous step's output hidden (sglang EAGLE draft loop:
# hidden_states = logits_output.hidden_states) instead of reusing the target hidden.
_HIDDEN_RELAY_ENV = os.environ.get("SGLANG_JAX_MTP_HIDDEN_RELAY")
# Companion to the relay: advance the RoPE positions of the rotated window by one
# per draft step (slot k holds tok_{k+j} at step j), like the EAGLE decode loop's
# positions = seq_lens + step. On by default with the relay (measured not worse:
# gsm8k p1 acceptance 3.34 vs 3.33, position-3 conditional 0.774 vs 0.763);
# SGLANG_JAX_MTP_RELAY_POS=0 keeps the step-0 positions.
_RELAY_POS = os.environ.get("SGLANG_JAX_MTP_RELAY_POS", "1") != "0"
# Single-block chain, pool versions. The caller keeps only step 0's pool
# version (steps j >= 1 write the rotated window back into the same slots, so
# their versions are throwaway), but every step used to start from the input
# pool and every version was returned: XLA had to keep three versions of a
# donated pool = two whole-pool copies per draft-extend step. Now step 0 runs
# on the input pool and is the only version returned; step 1 still starts from
# the input pool (unchanged semantics) and step 2 continues on step 1's version.
# Step 2 only reads the window slots it rewrites itself first, so its inputs
# are identical either way; XLA keeps two versions = one copy.
# (Threading ONE version through all steps was measured NOT idempotent: the
# window slots follow seq_lens, not the shifted positions, so step 1 overwrote
# step 0's KV; gsm8k acceptance fell 3.34 -> 3.21. Do not do that.)
# SGLANG_JAX_MTP_CHAIN_POOL=0 restores three separate versions.
# Default "all" (measured on GLM-5.2 tp16, v7x: draft-extend 1.90 -> 1.22 ms at
# batch 1, gsm8k acceptance unchanged 3.34); "1" = keep step 0's version and one
# shared scratch version (one copy); "0" = one version per step (two copies).
_CHAIN_POOL_MODE = os.environ.get("SGLANG_JAX_MTP_CHAIN_POOL", "all")
_CHAIN_POOL = _CHAIN_POOL_MODE != "0"
# "all": additionally shift the draft-extend metadata by the step index, so
# step j >= 1 writes its rotated window into the slots of the SAME tokens /
# positions step 0 wrote (idempotent) and its new draft token into the next
# free slot; one pool version is then threaded through all steps (no copy).
# Needs the relay + position shift and allocation slack of num_steps - 1 slots
# past the step-0 window (overlap scheduling keeps >= 2 * ALLOC_LEN_PER_DECODE).
_CHAIN_ALL = _CHAIN_POOL_MODE == "all"


# Debug assertion: copy step 0's pool version and, after the later steps ran on
# their scratch version, report per pool leaf how many KV slots differ between
# that copy and the version handed back, plus the step-0 window slots. Must be
# 0 everywhere (the caller keeps exactly step 0's version).
_CHAIN_POOL_CHECK = os.environ.get("SGLANG_JAX_MTP_CHAIN_POOL_CHECK", "0") == "1"


def _chain_pool_leaf_names(tree):
    """Host-side names for the leaves compared by _chain_pool_diff_arrays."""
    from jax.tree_util import keystr, tree_flatten_with_path

    leaves, _ = tree_flatten_with_path(tree)
    return [f"{keystr(path)} shape={tuple(a.shape)}" for path, a in leaves]


def _chain_pool_diff_arrays(v0, vN, loc0, rep_sharding=None):
    """Per pool leaf (in tree_leaves order): #slots whose contents differ between
    two pool versions and the first 32 differing slot ids; plus the step-0 window
    slots. Arrays only (JIT outputs); names come from _chain_pool_leaf_names.
    Under an explicit-sharding mesh the reductions run as auto-sharded regions
    with a replicated result."""

    def _diff(a0, aN):
        rows0 = a0.reshape(-1, a0.shape[-1]) if a0.ndim >= 2 else a0.reshape(-1, 1)
        rowsN = aN.reshape(-1, aN.shape[-1]) if aN.ndim >= 2 else aN.reshape(-1, 1)
        d = jnp.any(rows0 != rowsN, axis=-1)
        return d.sum().astype(jnp.int32), jnp.nonzero(d, size=32, fill_value=-1)[0].astype(
            jnp.int32
        )

    diff = _diff
    if rep_sharding is not None:
        diff = jax.sharding.auto_axes(_diff, out_sharding=(rep_sharding, rep_sharding))
    counts, firsts = [], []
    for a0, aN in zip(jax.tree_util.tree_leaves(v0), jax.tree_util.tree_leaves(vN)):
        if a0.shape != aN.shape:
            counts.append(jnp.int32(-1))
            firsts.append(jnp.full((32,), -1, jnp.int32))
            continue
        n, f = diff(a0, aN)
        counts.append(n)
        firsts.append(f)
    return jnp.stack(counts), jnp.stack(firsts), loc0


def log_chain_pool_check(report, pool_updates):
    """Log the arrays produced by _chain_pool_diff_arrays (host side)."""
    import numpy as np

    counts, firsts, loc0 = report
    names = _chain_pool_leaf_names(pool_updates)
    loc0 = np.asarray(loc0)
    live = set(int(x) for x in loc0 if x >= 0)
    logger.info("[CHAIN_POOL_CHECK] step0 window slots (loc0, -1 = dropped): %s", loc0.tolist())
    for name, n, f in zip(names, np.asarray(counts), np.asarray(firsts)):
        rows = [int(x) for x in f if x >= 0]
        logger.info(
            "[CHAIN_POOL_CHECK] %s rows_differ=%d first_rows=%s overlap_with_loc0=%s",
            name,
            int(n),
            rows,
            sorted(set(rows) & live),
        )


def _chain_pool_report(v0, vN, md, num_tokens, page_size, like, sel_pos=None):
    """CHECK report inside the fused JIT: step-0 window slots (auto-sharded
    lookup, replicated result) + per-leaf slot diffs between two pool versions.
    Returns arrays only."""
    from sgl_jax.srt.layers.attention.dsa_sparse_backend import _spec_token_slots

    sh = jax.typeof(like).sharding
    rep = None
    if isinstance(sh, NamedSharding) and not sh.mesh.empty:
        rep = NamedSharding(sh.mesh, P())

    def _loc0(sl, cq, ck, pi):
        return _spec_token_slots(sl, cq, ck, pi, num_tokens, page_size)

    loc0_fn = _loc0 if rep is None else jax.sharding.auto_axes(_loc0, out_sharding=rep)
    loc0 = loc0_fn(md.seq_lens, md.cu_q_lens, md.cu_kv_lens, md.page_indices)
    if sel_pos is not None:
        # window rows 0..sel_pos[r] hold request r's verified tokens (sel_pos =
        # accept_length - 1); the rows after it are padding whose slots the later
        # steps legitimately overwrite. Only the verified rows must stay untouched.
        bs = sel_pos.shape[0]
        n = num_tokens // bs

        def _valid(sp):
            return (jnp.arange(n)[None, :] <= sp[:, None]).reshape(-1)

        valid_fn = _valid if rep is None else jax.sharding.auto_axes(_valid, out_sharding=rep)
        loc0 = jnp.where(valid_fn(sel_pos), loc0, -1)
    return _chain_pool_diff_arrays(v0, vN, loc0, rep_sharding=rep)


def chain_all_headroom_ok(allocate_lens, verify_seq_lens, num_steps: int) -> bool:
    """Host-side guard for SGLANG_JAX_MTP_CHAIN_POOL=all: every live request must
    have num_steps - 1 allocated slots past its step-0 draft window
    (verify_seq_lens + num_steps). False -> the caller falls back to the
    two-version mode instead of writing past the allocation."""
    import numpy as np

    if not isinstance(allocate_lens, np.ndarray) or not isinstance(verify_seq_lens, np.ndarray):
        raise TypeError("chain_all_headroom_ok expects host numpy arrays (no device sync here)")
    alloc = allocate_lens
    vsl = verify_seq_lens
    live = vsl > 0
    if not live.any():
        return True
    return bool(np.all(alloc[live] >= vsl[live] + 2 * num_steps - 1))


_DRAFT_MD_FIELDS = ("cu_q_lens", "cu_kv_lens", "page_indices", "seq_lens", "distribution")


def _pack_draft_extend_metadata(
    md_orig, base_seq_lens, allocate_lens, num_steps, *, page_size, dp_size
):
    """Draft-extend metadata whose page table already covers the last chained
    step (base + num_steps - 1) for every live sequence. Built once per
    draft-extend call; the per-step variants only swap ``seq_lens`` (see
    _shift_draft_extend_metadata), so no page-table repack runs per step."""
    span = jnp.where(base_seq_lens > 0, base_seq_lens + (num_steps - 1), 0).astype(
        base_seq_lens.dtype
    )
    return _make_draft_extend_metadata(
        md_orig, span, allocate_lens, page_size=page_size, dp_size=dp_size
    )


def _shift_draft_extend_metadata(md_pack, base_seq_lens, step):
    """Metadata of chained step ``step`` on the shared packing: sequence lengths
    (window slots and kv span) advanced by ``step``; padding sequences stay 0."""
    seq_lens = jnp.where(base_seq_lens > 0, base_seq_lens + step, 0).astype(base_seq_lens.dtype)
    kwargs = {f: getattr(md_pack, f) for f in _DRAFT_MD_FIELDS}
    kwargs["seq_lens"] = seq_lens
    for f in ("swa_page_indices", "custom_mask"):
        if hasattr(md_pack, f):
            kwargs[f] = getattr(md_pack, f)
    return seq_lens, type(md_pack)(**kwargs)


def _chain_pool_enabled(num_pools: int, relay_on: bool) -> bool:
    """Single-block chain: return step 0's pool version only, share one scratch
    version between the later steps (see _CHAIN_POOL)."""
    del relay_on
    return bool(_CHAIN_POOL and num_pools == 1)


def mtp_hidden_relay_enabled(hf_config=None):
    """Env override for the hidden relay: True/False when SGLANG_JAX_MTP_HIDDEN_RELAY is
    set, else None = decide by chaining (see ``_chained_relay``). ``hf_config`` is
    accepted for symmetry with the IndexShare gate, but the draft runner's config
    does not reliably carry ``num_nextn_predict_layers`` (dpa47nap read False on
    GLM-5.2), so the structural rule is the default.
    """
    if _HIDDEN_RELAY_ENV is not None:
        return _HIDDEN_RELAY_ENV.strip() not in ("", "0", "false", "False")
    return None


def _chained_relay(hidden_relay, num_steps: int, num_blocks: int) -> bool:
    """Resolve the relay flag inside the fused loop.

    ``num_blocks`` = number of distinct draft blocks handed to the loop
    (``len(all_leaves)``); when it is smaller than ``num_steps`` the last block is
    applied again (``leaf_idx = -1``), i.e. a single-block MTP chain (GLM-5.2,
    DeepSeek-style) whose later steps must see the previous step's output hidden.
    Multi-block MTP (one block per step) keeps the target hidden for every block.
    """
    if hidden_relay is not None:
        return bool(hidden_relay)
    return num_blocks < num_steps


def _spec_decode_compiler_options():
    """Per-executable XLA options for the decode-shaped speculative executables.

    The non-speculative decode path compiles its executables with the
    SparseCore gather offload disabled when
    ``SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD`` is set (jax 0.11.1 TPU
    regression, see ``aot_dispatch.decode_no_sc_gather_compiler_options_fn``).
    The speculative draft-extend / fused-verify executables are decode-shaped
    too but were compiled with plain ``jax.jit`` and therefore kept the
    offload; on v7x tp16 that left ~2/3 of a fused-verify step in offloaded
    small collectives. Mirror the same opt-in here so both paths agree; the
    prefill-phase speculative executable keeps the default (large prefill
    gathers profit from the offload).
    """
    if jax.default_backend() != "tpu" or not get_bool_env_var(
        "SGLANG_JAX_DECODE_DISABLE_SC_GATHER_OFFLOAD"
    ):
        return None
    return {
        "xla_tpu_offload_gather_to_sparsecore": "false",
        "xla_tpu_offload_all_supported_gathers_to_sparsecore": "false",
    }


_SPEC_DECODE_COMPILER_OPTIONS = _spec_decode_compiler_options()


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
    target_predict,
    speculative_num_steps,
    speculative_num_draft_tokens,
    simulation_rng=None,
):
    bs = seq_lens.shape[0]
    n = speculative_num_draft_tokens
    width = speculative_num_steps + 1
    draft_2d = draft_tokens.reshape(bs, n)
    target_predict_2d = target_predict.reshape(bs, n)
    predict_sharding = jax.typeof(target_predict).sharding
    mesh = predict_sharding.mesh if isinstance(predict_sharding, NamedSharding) else None
    if mesh is not None and not mesh.empty:
        data_2d = NamedSharding(mesh, P("data", None))
        draft_2d = jax.sharding.reshard(draft_2d, data_2d)
        target_predict_2d = jax.sharding.reshard(target_predict_2d, data_2d)

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

    predict = target_predict_2d.astype(jnp.int32).reshape(-1)
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
    """Build topk=1 linear-chain verify inputs in-JIT without stacking shardings."""
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
    # Pin the token-dim arrays to the batch's data placement. Under explicit
    # sharding the (bs, n) -> (bs * n) reshape above drops a size-1 "data" axis
    # (dp=1 gives P(None)), but the attention backends shard_map positions /
    # input_ids with P("data") exactly like the host-built extend batch
    # (_make_forward_batch), so a verify step must hand them over the same way.
    seq_sharding = jax.typeof(seq_lens).sharding
    if isinstance(seq_sharding, NamedSharding) and not seq_sharding.mesh.empty:
        if jax.typeof(positions).sharding != seq_sharding:
            positions = jax.sharding.reshard(positions, seq_sharding)
        if jax.typeof(draft_tokens).sharding != seq_sharding:
            draft_tokens = jax.sharding.reshard(draft_tokens, seq_sharding)
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

    def _rotate(input_ids, ext_lens, sel_pos, new_tokens):
        ids_2d = input_ids.reshape(bs, tokens_per_req)
        shifted_2d = jnp.concatenate([ids_2d[:, 1:], ids_2d[:, -1:]], axis=1)
        shifted_2d = shifted_2d.at[jnp.arange(bs), sel_pos].set(new_tokens)
        pad_mask = (ext_lens == 0)[:, None]
        shifted_2d = jnp.where(pad_mask, ids_2d, shifted_2d)
        return shifted_2d.reshape(-1)

    # Auto-sharded region: with an explicit mesh the [bs*n] -> [bs, n] split
    # hangs a size-1 "data" axis (dp=1) on the n dim and the two slices no
    # longer agree (concatenate raises ShardingTypeError). Let XLA infer the
    # interior and pin only the output to the input's placement (it feeds the
    # next draft layer's embedding).
    return _auto_sharded(_rotate, input_ids)(input_ids, ext_lens, sel_pos, new_tokens)


def _rotate_prefill_input_ids(input_ids, extend_seq_lens, verified_id, dp_size, per_dp_bs):
    per_dp_tokens = input_ids.shape[0] // dp_size
    ids = input_ids.reshape(dp_size, per_dp_tokens)
    ext = extend_seq_lens.reshape(dp_size, per_dp_bs)
    verified = verified_id.reshape(dp_size, per_dp_bs)
    tok = jnp.arange(per_dp_tokens, dtype=jnp.int32)

    def rotate_rank(ids_rank, ext_rank, verified_rank):
        # We can implement a cumsum as a matrix multiplication!
        # A lower triangular matrix of ones multiplied by ext_rank gives the cumsum!
        import jax
        import jax.numpy as jnp

        N = ext_rank.shape[0]
        idx = jnp.arange(N)
        mask = (idx[:, None] >= idx[None, :]).astype(jnp.int32)

        # This is exactly the inclusive cumsum!
        ext_rank_cumsum = jnp.dot(mask, ext_rank)

        starts = ext_rank_cumsum - ext_rank
        ends = starts + ext_rank
        in_req = (tok[None, :] >= starts[:, None]) & (tok[None, :] < ends[:, None])
        has_req = jnp.any(in_req, axis=0)
        slot = jnp.argmax(in_req.astype(jnp.int32), axis=0)

        # Matrix multiply bypassing JAX gather layout check
        one_hot_slot = jax.nn.one_hot(slot, N, dtype=starts.dtype)
        req_starts = jnp.dot(one_hot_slot, starts)
        req_lens = jnp.dot(one_hot_slot, ext_rank)
        req_verified = jnp.dot(one_hot_slot, verified_rank)

        shifted_index = jnp.minimum(tok + 1, per_dp_tokens - 1)
        one_hot_shifted = jax.nn.one_hot(shifted_index, per_dp_tokens, dtype=starts.dtype)
        shifted = jnp.dot(one_hot_shifted, ids_rank)

        # Bypass jnp.where ShardingTypeError by using algebraic boolean masks!
        # Multiplication automatically promotes sharding constraints!
        is_last = has_req & ((tok - req_starts) == (req_lens - 1))
        is_last_int = is_last.astype(jnp.int32)
        rotated = is_last_int * req_verified + (1 - is_last_int) * shifted

        has_req_int = has_req.astype(jnp.int32)
        return has_req_int * rotated + (1 - has_req_int) * ids_rank

    return jax.vmap(rotate_rank)(ids, ext, verified).reshape(input_ids.shape)


def _rotate_hidden(hidden, ext_lens, sel_pos, prev_out_hidden):
    """Hidden-state relay for draft step j >= 1 of the single-layer chain.

    Mirrors ``_rotate_input_ids``: each request's ``[tokens_per_req, H]`` window
    shifts left by one so row k stays paired with the rotated ``input_ids`` row k
    (the MTP block consumes ``(h_i, tok_{i+1})`` pairs), and the last valid slot
    takes the previous step's output hidden at that slot -- the draft's own
    ``h_{t+j}``, which the target never produced. Padding requests keep their rows.
    """
    bs = ext_lens.shape[0]
    tokens_per_req = hidden.shape[0] // bs

    def _rot(hidden, ext_lens, sel_pos, prev):
        h2 = hidden.reshape(bs, tokens_per_req, -1)
        p2 = prev.reshape(bs, tokens_per_req, -1)
        shifted = jnp.concatenate([h2[:, 1:], h2[:, -1:]], axis=1)
        rows = jnp.arange(bs)
        shifted = shifted.at[rows, sel_pos].set(p2[rows, sel_pos])
        pad_mask = (ext_lens == 0)[:, None, None]
        return jnp.where(pad_mask, h2, shifted).reshape(hidden.shape)

    return _auto_sharded(_rot, hidden)(hidden, ext_lens, sel_pos, prev_out_hidden)


def _rotate_prefill_hidden(hidden, extend_seq_lens, prev_out_hidden, dp_size, per_dp_bs):
    """Prefill-time counterpart of ``_rotate_hidden`` for the ragged per-rank layout
    used by ``_rotate_prefill_input_ids`` (rows shift left inside each request's
    segment; the segment's last row takes the previous step's output at that row).
    Every index array is built inside the auto-sharded region (an explicit-mesh
    ``arange`` closed over from outside fails the region's mesh check).
    """
    per_dp_tokens = hidden.shape[0] // dp_size

    def rotate_rank(h_rank, ext_rank, prev_rank):
        n_req = ext_rank.shape[0]
        tok = jnp.arange(per_dp_tokens, dtype=jnp.int32)
        idx = jnp.arange(n_req)
        tri = (idx[:, None] >= idx[None, :]).astype(jnp.int32)
        ends = jnp.dot(tri, ext_rank)  # inclusive cumsum
        starts = ends - ext_rank
        in_req = (tok[None, :] >= starts[:, None]) & (tok[None, :] < ends[:, None])
        has_req = jnp.any(in_req, axis=0)
        is_last = jnp.any(in_req & (tok[None, :] == (ends - 1)[:, None]), axis=0)
        shifted = jnp.take(h_rank, jnp.minimum(tok + 1, per_dp_tokens - 1), axis=0)
        out = jnp.where(is_last[:, None], prev_rank, shifted)
        return jnp.where(has_req[:, None], out, h_rank)

    def _rot(hidden, ext_flat, prev):
        h3 = hidden.reshape(dp_size, per_dp_tokens, -1)
        p3 = prev.reshape(dp_size, per_dp_tokens, -1)
        ext = ext_flat.reshape(dp_size, per_dp_bs)
        return jax.vmap(rotate_rank)(h3, ext, p3).reshape(hidden.shape)

    return _auto_sharded(_rot, hidden)(hidden, extend_seq_lens, prev_out_hidden)


def _gather_rows_preserve_sharding(values, index):
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding):
        return values.at[index, :].get(out_sharding=sharding)
    return values[index, :]


def _reshard_values(sharding, *values):
    return tuple(jax.sharding.reshard(value, sharding) for value in values)


def _auto_sharded(fn, out_like):
    """Run ``fn`` as an auto-sharded region whose result takes ``out_like``'s placement.

    The fused spec JITs run on an explicit-sharding mesh, where in-JIT shape
    plumbing (reshape / slice / concatenate / where on ``[bs*n]`` <-> ``[bs, n]``
    views) must type-check shardings and a size-1 ``data`` axis (dp=1) makes the
    inferred specs disagree. Inside ``jax.sharding.auto_axes`` XLA infers the
    interior freely; only the exit placement is pinned. Without an explicit
    mesh (plain CPU tests) ``fn`` is returned unchanged.
    """
    sharding = jax.typeof(out_like).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        return jax.sharding.auto_axes(fn, out_sharding=sharding)
    return fn


def _topk1_index_from_logits(logits):
    topk_idx = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
    return topk_idx


def _seed_topk_pages_from_step0(topk_pages, ext_lens, sel_pos):
    """GLM-5.2 MTP IndexShare: broadcast each request's step-0 selection.

    ``topk_pages`` is the draft layer's ``[T, k_pages]`` per-query page-topk from
    draft step 0, whose window rows are packed ``[bs, tokens_per_req]`` (same
    layout ``_rotate_input_ids`` assumes). Row ``sel_pos[b]`` of request ``b`` is
    its last verified token; every later draft step reuses that row for all of
    the request's window rows, mirroring the reference implementation which
    seeds the draft iterations from the draft-extend top-k of the last verified
    token. Padding requests (``ext_lens == 0``) keep their step-0 rows.
    """
    bs = ext_lens.shape[0]
    tokens_per_req = topk_pages.shape[0] // bs

    def _seed(topk_pages, ext_lens, sel_pos):
        pages_3d = topk_pages.reshape(bs, tokens_per_req, topk_pages.shape[1])
        onehot = jnp.arange(tokens_per_req, dtype=jnp.int32)[None, :] == sel_pos[:, None]
        seed = jnp.sum(jnp.where(onehot[:, :, None], pages_3d, 0), axis=1, dtype=pages_3d.dtype)
        seeded = jnp.broadcast_to(seed[:, None, :], pages_3d.shape)
        keep_step0 = (ext_lens == 0)[:, None, None]
        seeded = jnp.where(keep_step0, pages_3d, seeded)
        return seeded.reshape(topk_pages.shape)

    # Auto-sharded region: with an explicit mesh the [T, k] -> [bs, tpr, k]
    # split hangs a size-1 "data" axis (dp=1) on the tpr dim while ext_lens /
    # sel_pos keep it on dim 0, and the where() broadcast becomes an illegal
    # P("data", "data", None). Let XLA infer the interior and pin only the
    # output to the input's placement (the backend shard_maps it P("data", None)).
    return _auto_sharded(_seed, topk_pages)(topk_pages, ext_lens, sel_pos)


def mtp_index_share_enabled(hf_config, topk: int) -> bool:
    """Whether draft steps reuse the step-0 indexer selection (IndexShare).

    Default follows the checkpoint's ``index_share_for_mtp_iteration`` (GLM-5.2
    sets it); ``SGLANG_JAX_MTP_INDEX_SHARE=0/1`` forces it for A/B runs. Only the
    topk=1 chain is supported (rows are not reordered between steps).
    """
    forced = os.environ.get("SGLANG_JAX_MTP_INDEX_SHARE")
    if forced is not None:
        return forced.strip() not in ("", "0", "false", "False") and topk == 1
    return bool(getattr(hf_config, "index_share_for_mtp_iteration", False)) and topk == 1


def _build_draft_extend(
    num_layers: int, topk: int, index_share: bool = False, hidden_relay: bool = False
):
    """Build the fused JIT. Called once, result cached on draft_worker."""
    assert topk == 1, "Fused draft extend only supports topk=1"

    @partial(
        jax.jit,
        compiler_options=_SPEC_DECODE_COMPILER_OPTIONS,
        donate_argnames=["all_memory_pools"],
        static_argnames=[
            "model_state_def",
            "num_layers",
            "update_relay",
            "dp_size",
            "chain_all_ok",
        ],
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
        chain_all_ok=False,
    ):
        all_topk_index = []
        all_pool_updates = []
        layer0_hidden = None
        mesh = None
        seed_topk_pages = None
        input_ids = forward_batch.input_ids
        md_orig = None
        if draft_verify_seq_lens is not None:
            md_orig = forward_batch.attn_backend.forward_metadata
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

        step_hidden = target_hidden
        positions0 = forward_batch.positions
        relay_on = _chained_relay(hidden_relay, num_layers, len(all_leaves))
        chain_pool = _chain_pool_enabled(len(all_memory_pools), relay_on)
        chain_all = bool(
            _CHAIN_ALL
            and chain_all_ok
            and chain_pool
            and relay_on
            and _RELAY_POS
            and draft_verify_seq_lens is not None
            and md_orig is not None
        )
        base_seq_lens = forward_batch.seq_lens
        md_base = forward_batch.attn_backend.forward_metadata
        if chain_all:
            # one page table for all steps (covers base + num_layers - 1), step 0 included
            md_pack = _pack_draft_extend_metadata(
                md_orig,
                base_seq_lens,
                draft_allocate_lens,
                num_layers,
                page_size=forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )
            _, md_base = _shift_draft_extend_metadata(md_pack, base_seq_lens, 0)
            forward_batch.attn_backend.forward_metadata = md_base
        chain_check_report = None
        for i in range(num_layers):
            leaf_idx = i if i < len(all_leaves) else -1
            pool_idx = i if i < len(all_memory_pools) else -1
            state = jax.tree_util.tree_unflatten(model_state_def, all_leaves[leaf_idx])
            model = nnx.merge(model_def, state)

            forward_batch.spec_info.hidden_states = step_hidden
            forward_batch.input_ids = input_ids
            if relay_on and _RELAY_POS and i > 0:
                forward_batch.positions = positions0 + i
            if chain_all and i > 0:
                forward_batch.seq_lens, forward_batch.attn_backend.forward_metadata = (
                    _shift_draft_extend_metadata(md_pack, base_seq_lens, i)
                )
            forward_batch.spec_kvshare_readonly = bool(_KVSHARE and i >= 1)

            if index_share:
                output, pool_updates, _, _, step_topk_pages = model(
                    forward_batch,
                    all_memory_pools[pool_idx],
                    logits_metadata,
                    dsa_topk_pages_in=seed_topk_pages,
                    dsa_topk_reuse=seed_topk_pages is not None,
                    return_dsa_topk_pages=True,
                )
                if i == 0 and step_topk_pages is not None:
                    seed_topk_pages = _seed_topk_pages_from_step0(
                        step_topk_pages, forward_batch.extend_seq_lens, sel_pos
                    )
            else:
                output, pool_updates, _, _ = model(
                    forward_batch, all_memory_pools[pool_idx], logits_metadata
                )
            if chain_all:
                # one version through all steps (slots shifted per step, see _CHAIN_ALL)
                if _CHAIN_POOL_CHECK and i == 0:
                    check_v0 = jax.tree_util.tree_map(lambda x: x + 0, pool_updates)
                all_memory_pools[pool_idx].replace_all(pool_updates)
                if i == num_layers - 1:
                    all_pool_updates.append(pool_updates)
                    if _CHAIN_POOL_CHECK:
                        # assertion data vs step 0: differing slots must all lie at or
                        # past each request's first non-verified slot (valid rows unchanged)
                        chain_check_report = _chain_pool_report(
                            check_v0,
                            pool_updates,
                            md_base,
                            input_ids.shape[0],
                            forward_batch.attn_backend.page_size,
                            input_ids,
                            sel_pos=sel_pos,
                        )
            elif chain_pool:
                # step 0's version is the one the caller keeps; steps >= 1 share
                # one scratch version (step 2 continues on step 1's), see _CHAIN_POOL
                if i == 0:
                    all_pool_updates.append(pool_updates)
                    if _CHAIN_POOL_CHECK:
                        check_v0 = jax.tree_util.tree_map(lambda x: x + 0, pool_updates)
                else:
                    all_memory_pools[pool_idx].replace_all(pool_updates)
                    if _CHAIN_POOL_CHECK and i == num_layers - 1:
                        # assertion data: the version handed back must still be
                        # step 0's (rows_differ must be 0 everywhere, loc0 included)
                        chain_check_report = _chain_pool_report(
                            check_v0,
                            all_pool_updates[0],
                            md_base,
                            input_ids.shape[0],
                            forward_batch.attn_backend.page_size,
                            input_ids,
                        )
            else:
                all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else None

            if i == 0:
                layer0_hidden = output.hidden_states

            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            all_topk_index.append(topk_idx)
            # jax.debug.print("[SPEC_DRAFT_EXTEND] Step {step} predicted draft token IDs: {tok}", step=i, tok=topk_idx[:, 0])

            if i < num_layers - 1:
                ext_lens = forward_batch.extend_seq_lens
                input_ids = _rotate_input_ids(input_ids, ext_lens, sel_pos, topk_idx[:, 0])
                if relay_on:
                    step_hidden = _rotate_hidden(
                        step_hidden, ext_lens, sel_pos, output.hidden_states
                    )

        forward_batch.spec_kvshare_readonly = False
        forward_batch.positions = positions0
        if chain_all:
            forward_batch.seq_lens = base_seq_lens
            forward_batch.attn_backend.forward_metadata = md_base
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

        outs = (
            selected_layer0_hidden,
            stacked_idx,
            tuple(all_pool_updates),
            updated_relay_buffers,
        )
        if _CHAIN_POOL_CHECK:
            outs = outs + (chain_check_report,)
        return outs

    return fused_draft_extend


def _reshape_per_dp_rows(values, dp_size: int):
    per_dp_size = values.shape[0] // dp_size
    rows = values.reshape((dp_size, per_dp_size))
    sharding = jax.typeof(values).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        # Keep the DP axis sharded and the rank-local reduction axis unsharded after reshape.
        rows = jax.sharding.reshard(rows, NamedSharding(sharding.mesh, P("data", None)))
    return rows


def _per_dp_cumsum_device(lens, dp_size: int):
    per_dp_bs = lens.shape[0] // dp_size
    lens_2d = _reshape_per_dp_rows(lens, dp_size)
    rows_sharding = jax.typeof(lens_2d).sharding
    zeros = jnp.zeros_like(lens_2d[:, :1], dtype=jnp.int32)
    cumsum = jnp.cumsum(lens_2d, axis=1, dtype=jnp.int32)
    if isinstance(rows_sharding, NamedSharding) and not rows_sharding.mesh.empty:
        # Explicit-sharding JAX requires concatenate operands to carry the same
        # sharding. zeros_like may otherwise infer replicated sharding even
        # though lens_2d and its cumulative sum are data-sharded.
        zeros = jax.sharding.reshard(zeros, rows_sharding)
        cumsum = jax.sharding.reshard(cumsum, rows_sharding)
    result = jnp.concatenate([zeros, cumsum], axis=1).reshape((dp_size * (per_dp_bs + 1),))
    sharding = jax.typeof(lens).sharding
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        result = jax.sharding.reshard(result, sharding)
    return result


def _repack_page_indices(
    page_indices,
    allocated_lens,
    metadata_seq_lens,
    *,
    page_size: int,
    dp_size: int,
):
    pages_per_dp = page_indices.shape[0] // dp_size

    allocated_pages = ((allocated_lens + page_size - 1) // page_size).astype(jnp.int32)
    needed_pages = ((metadata_seq_lens + page_size - 1) // page_size).astype(jnp.int32)
    allocated_pages = _reshape_per_dp_rows(allocated_pages, dp_size)
    needed_pages = _reshape_per_dp_rows(needed_pages, dp_size)

    src_offsets = jnp.cumsum(allocated_pages, axis=1, dtype=jnp.int32) - allocated_pages
    dst_offsets = jnp.cumsum(needed_pages, axis=1, dtype=jnp.int32) - needed_pages

    local_page_ids = jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :, None]
    in_req = (local_page_ids >= dst_offsets[:, None, :]) & (
        local_page_ids < (dst_offsets + needed_pages)[:, None, :]
    )
    slot_ids = jnp.argmax(in_req.astype(jnp.int32), axis=2).astype(jnp.int32)
    valid = jnp.any(in_req, axis=2)

    dp_ids = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    offset_deltas = src_offsets - dst_offsets
    offsets_sharding = jax.typeof(offset_deltas).sharding
    offsets_out_sharding = offsets_sharding if isinstance(offsets_sharding, NamedSharding) else None
    slot_offset_deltas = offset_deltas.at[dp_ids, slot_ids].get(out_sharding=offsets_out_sharding)
    gather_src = (
        dp_ids * pages_per_dp
        + slot_offset_deltas
        + jnp.arange(pages_per_dp, dtype=jnp.int32)[None, :]
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
    if isinstance(page_sharding, NamedSharding) and not page_sharding.mesh.empty:
        gathered = jax.sharding.reshard(
            gathered, NamedSharding(page_sharding.mesh, P("data", None))
        )
    gathered_sharding = jax.typeof(gathered).sharding
    if isinstance(gathered_sharding, NamedSharding) and not gathered_sharding.mesh.empty:
        valid = jax.sharding.reshard(valid, gathered_sharding)
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
    if getattr(old_metadata, "swa_page_indices", None) is not None:
        swa_page_indices = _repack_page_indices(
            old_metadata.swa_page_indices,
            allocated_lens,
            metadata_seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    if type(old_metadata).__name__ == "MLAAttentionMetadata":
        distribution = jnp.stack(
            [jnp.zeros_like(local_num_seqs), jnp.zeros_like(local_num_seqs), local_num_seqs],
            axis=1,
        ).reshape((dp_size * 3,))
    else:
        distribution = jnp.stack(
            [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
            axis=1,
        ).reshape((dp_size * 3,))

    data_sharding = jax.typeof(verify_seq_lens).sharding
    if isinstance(data_sharding, NamedSharding) and not data_sharding.mesh.empty:
        cu_q_lens = jax.sharding.reshard(cu_q_lens, data_sharding)
        cu_kv_lens = jax.sharding.reshard(cu_kv_lens, data_sharding)
        page_indices = jax.sharding.reshard(page_indices, data_sharding)
        metadata_seq_lens = jax.sharding.reshard(metadata_seq_lens, data_sharding)
        distribution = jax.sharding.reshard(distribution, data_sharding)
        if swa_page_indices is not None:
            swa_page_indices = jax.sharding.reshard(swa_page_indices, data_sharding)

    kwargs = {
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "page_indices": page_indices,
        "seq_lens": metadata_seq_lens,
        "distribution": distribution,
    }
    if hasattr(old_metadata, "swa_page_indices"):
        kwargs["swa_page_indices"] = swa_page_indices
    if hasattr(old_metadata, "custom_mask"):
        kwargs["custom_mask"] = old_metadata.custom_mask
    return type(old_metadata)(**kwargs)


def _make_draft_extend_metadata(
    old_metadata,
    draft_seq_lens,
    allocated_lens,
    *,
    query_lens=None,
    page_size: int,
    dp_size: int,
):

    valid = draft_seq_lens > 0
    # Fused EAGLE3 passes device query lengths so this cumsum remains in the
    # parent JIT. Other callers can retain the host-prepared value.
    cu_q_lens = (
        old_metadata.cu_q_lens if query_lens is None else _per_dp_cumsum_device(query_lens, dp_size)
    )
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
    if getattr(old_metadata, "swa_page_indices", None) is not None:
        swa_page_indices = _repack_page_indices(
            old_metadata.swa_page_indices,
            allocated_lens,
            draft_seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    if type(old_metadata).__name__ == "MLAAttentionMetadata":
        distribution = jnp.stack(
            [jnp.zeros_like(local_num_seqs), jnp.zeros_like(local_num_seqs), local_num_seqs],
            axis=1,
        ).reshape((dp_size * 3,))
    else:
        distribution = jnp.stack(
            [jnp.zeros_like(local_num_seqs), local_num_seqs, local_num_seqs],
            axis=1,
        ).reshape((dp_size * 3,))

    data_sharding = jax.typeof(draft_seq_lens).sharding
    if isinstance(data_sharding, NamedSharding) and not data_sharding.mesh.empty:
        cu_q_lens = jax.sharding.reshard(cu_q_lens, data_sharding)
        cu_kv_lens = jax.sharding.reshard(cu_kv_lens, data_sharding)
        page_indices = jax.sharding.reshard(page_indices, data_sharding)
        draft_seq_lens = jax.sharding.reshard(draft_seq_lens, data_sharding)
        distribution = jax.sharding.reshard(distribution, data_sharding)
        if swa_page_indices is not None:
            swa_page_indices = jax.sharding.reshard(swa_page_indices, data_sharding)

    kwargs = {
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "page_indices": page_indices,
        "seq_lens": draft_seq_lens,
        "distribution": distribution,
    }
    if hasattr(old_metadata, "swa_page_indices"):
        kwargs["swa_page_indices"] = swa_page_indices
    if hasattr(old_metadata, "custom_mask"):
        kwargs["custom_mask"] = old_metadata.custom_mask
    return type(old_metadata)(**kwargs)


def _make_eagle3_decode_metadata(
    old_metadata,
    seq_lens,
    allocated_lens,
    *,
    page_size: int,
    dp_size: int,
):
    """Build one-token recurrent EAGLE3 decode metadata on device."""
    from sgl_jax.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )

    valid = seq_lens > 0
    total_bs = seq_lens.shape[0]
    per_dp_bs = total_bs // dp_size
    local_cu_q_lens = jnp.arange(per_dp_bs + 1, dtype=jnp.int32)
    cu_q_lens = jnp.tile(local_cu_q_lens, dp_size)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    cu_kv_lens = _per_dp_cumsum_device(aligned_seq_lens, dp_size)
    page_indices = _repack_page_indices(
        old_metadata.page_indices,
        allocated_lens,
        seq_lens,
        page_size=page_size,
        dp_size=dp_size,
    )
    swa_page_indices = None
    if old_metadata.swa_page_indices is not None:
        swa_page_indices = _repack_page_indices(
            old_metadata.swa_page_indices,
            allocated_lens,
            seq_lens,
            page_size=page_size,
            dp_size=dp_size,
        )

    valid_rows = _reshape_per_dp_rows(valid, dp_size)
    local_num_seqs = jnp.sum(valid_rows.astype(jnp.int32), axis=1)
    distribution = jnp.stack(
        [jnp.zeros_like(local_num_seqs), jnp.zeros_like(local_num_seqs), local_num_seqs],
        axis=1,
    ).reshape((dp_size * 3,))

    data_sharding = jax.typeof(seq_lens).sharding
    if isinstance(data_sharding, NamedSharding) and not data_sharding.mesh.empty:
        cu_q_lens = jax.sharding.reshard(cu_q_lens, data_sharding)
        cu_kv_lens = jax.sharding.reshard(cu_kv_lens, data_sharding)
        page_indices = jax.sharding.reshard(page_indices, data_sharding)
        seq_lens = jax.sharding.reshard(seq_lens, data_sharding)
        distribution = jax.sharding.reshard(distribution, data_sharding)
        if swa_page_indices is not None:
            swa_page_indices = jax.sharding.reshard(swa_page_indices, data_sharding)

    return FlashAttentionMetadata(
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        page_indices=page_indices,
        swa_page_indices=swa_page_indices,
        seq_lens=seq_lens,
        distribution=distribution,
        custom_mask=None,
    )


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


def _build_eagle3_recurrent_draft_extend(num_steps: int, topk: int):
    """Build EAGLE3 draft-extend followed by recurrent one-token draft steps."""
    assert topk == 1, "Fused recurrent EAGLE3 draft extend only supports topk=1"

    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    @partial(
        jax.jit,
        compiler_options=_SPEC_DECODE_COMPILER_OPTIONS,
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
        base_metadata = forward_batch.attn_backend.forward_metadata

        valid_draft_slots = draft_verify_seq_lens > 0
        forward_batch.seq_lens = jnp.where(
            valid_draft_slots,
            draft_verify_seq_lens + num_steps,
            jnp.zeros_like(draft_verify_seq_lens),
        )
        forward_batch.attn_backend.forward_metadata = _make_draft_extend_metadata(
            base_metadata,
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
            forward_batch.attn_backend.forward_metadata = _make_eagle3_decode_metadata(
                base_metadata,
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


def _build_verify(topk: int):
    """Build target forward + linear-chain verify JIT."""
    assert topk == 1, "Fused greedy verify only supports topk=1"

    @partial(
        jax.jit,
        compiler_options=_SPEC_DECODE_COMPILER_OPTIONS,
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
            # Force explicit P("data") without mesh using PartitionSpec
            valid_seq_lens = target_forward_batch.seq_lens > 0
            zeros = jnp.zeros_like(target_forward_batch.seq_lens)
            b = relay_new_seq_lens - 1 + zeros

            target_forward_batch.seq_lens = jnp.where(
                valid_seq_lens,
                b,
                zeros,
            )
            previous_verified_id = relay_verified_id
            previous_token_list = relay_topk_index

        if use_relay_state or rebuild_verify_metadata:
            target_forward_batch.attn_backend.forward_metadata = _make_target_verify_metadata(
                target_forward_batch.attn_backend.forward_metadata,
                target_forward_batch.seq_lens,
                verify_allocate_lens,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                page_size=target_forward_batch.attn_backend.page_size,
                dp_size=dp_size,
            )

        if draft_to_target_token_ids is not None:
            previous_token_list = _map_eagle3_token_ids(
                previous_token_list,
                draft_to_target_token_ids,
            )

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
        # Advance inside verify, avoiding an eager scalar-add dispatch on every decode.
        sampling_step = sampling_step + 1
        sampling_rng = jax.random.fold_in(sampling_base_rng, sampling_step)
        simulation_rng = jax.random.fold_in(sampling_rng, 1)
        if is_greedy:
            target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(-1)
            prepared = _verify_greedy(
                target_hidden=target_hidden,
                positions=target_forward_batch.positions,
                seq_lens=target_forward_batch.seq_lens,
                draft_tokens=draft_tokens,
                target_predict=target_predict,
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
        prepared_new_seq_lens_data = prepared.new_seq_lens
        prepared_accept_lens_host = prepared.accept_lens
        prepared_accept_lens_data = prepared.accept_lens
        prepared_extend_seq_lens = jnp.where(
            target_forward_batch.seq_lens > 0,
            jnp.full_like(target_forward_batch.seq_lens, speculative_num_draft_tokens),
            jnp.zeros_like(target_forward_batch.seq_lens),
        ).astype(jnp.int32)
        _extend_lens_2d = prepared_extend_seq_lens.reshape(dp_size, target_bs // dp_size)
        _ext_sh = jax.typeof(_extend_lens_2d).sharding
        if isinstance(_ext_sh, NamedSharding) and not _ext_sh.mesh.empty:
            _extend_lens_2d = jax.sharding.reshard(
                _extend_lens_2d, NamedSharding(_ext_sh.mesh, P())
            )
        prepared_logits_indices = (jnp.cumsum(_extend_lens_2d, axis=1).reshape(-1) - 1).astype(
            jnp.int32
        )
        prepared_sel_pos = prepared.sel_pos
        prepared_sel_pos_data = prepared.sel_pos
        prepared_predict = prepared.predict

        # jax.debug.print(
        #     "\n[SPEC_VERIFY]\n  Draft tokens: {d}\n  Target predicted: {t}\n  Accept length: {a}\n  Verified tokens: {v}",
        #     d=draft_tokens,
        #     t=prepared.predict,
        #     a=prepared.accept_lens,
        #     v=prepared.verified_id,
        # )
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
                prepared_new_seq_lens_data,
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
                prepared_new_seq_lens_data,
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
            prepared_new_seq_lens_data,
            sampling_step,
        )

    return fused_verify


def _build_prefill(num_layers: int, topk: int, hidden_relay=None):
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

        step_hidden = target_hidden
        draft_forward_batch.spec_info.hidden_states = step_hidden
        draft_positions0 = draft_forward_batch.positions
        relay_on = _chained_relay(hidden_relay, num_layers, len(draft_all_leaves))
        for i in range(num_layers):
            leaf_idx = i if i < len(draft_all_leaves) else -1
            pool_idx = i if i < len(all_memory_pools) else -1
            state = jax.tree_util.tree_unflatten(draft_model_state_def, draft_all_leaves[leaf_idx])
            model = nnx.merge(draft_model_def, state)

            draft_forward_batch.input_ids = input_ids
            draft_forward_batch.spec_kvshare_readonly = bool(_KVSHARE and i >= 1)
            draft_forward_batch.spec_info.hidden_states = step_hidden
            if relay_on and _RELAY_POS and i > 0:
                draft_forward_batch.positions = draft_positions0 + i
            output, pool_updates, _, _ = model(
                draft_forward_batch, all_memory_pools[pool_idx], draft_logits_metadata
            )
            all_pool_updates.append(pool_updates)

            sh = jax.typeof(output.next_token_logits).sharding
            mesh = sh.mesh if isinstance(sh, NamedSharding) else mesh
            topk_idx = _topk1_index_from_logits(output.next_token_logits)
            all_topk_index.append(topk_idx)
            if i == 0:
                layer0_hidden = output.hidden_states
            if i < num_layers - 1:
                input_ids = _rotate_prefill_input_ids(
                    input_ids,
                    draft_forward_batch.extend_seq_lens,
                    topk_idx[:, 0],
                    dp_size,
                    per_dp_bs,
                )
                if relay_on:
                    step_hidden = _rotate_prefill_hidden(
                        step_hidden,
                        draft_forward_batch.extend_seq_lens,
                        output.hidden_states,
                        dp_size,
                        per_dp_bs,
                    )

        draft_forward_batch.positions = draft_positions0
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

    if not draft_padding_prepared:
        # Relay buffers keep recurrent EAGLE3 ids in draft-vocabulary space;
        # fused_verify gathers and maps that chain itself.  The host-side
        # placeholders above are never consumed, so mapping them here launches
        # an eager gather (and its broadcast) on every overlap round.
        draft_worker.padding_for_decode(
            model_worker_batch,
            map_hot_token_ids=not use_relay_state,
        )
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
        data_2d_sharding = NamedSharding(draft_worker.mesh, P("data", None))
        verify_input = EagleVerifyInput(
            draft_token=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
            custom_mask=None,
            positions=jax.device_put(np.zeros((flat,), dtype=np.int32), data_sharding),
            retrive_index=jax.device_put(np.zeros((bs, n), dtype=np.int32), data_2d_sharding),
            retrive_next_token=jax.device_put(np.zeros((bs, n), dtype=np.int32), data_2d_sharding),
            retrive_next_sibling=jax.device_put(
                np.zeros((bs, n), dtype=np.int32), data_2d_sharding
            ),
            spec_steps=draft_worker.speculative_num_steps,
            draft_token_num=draft_worker.speculative_num_draft_tokens,
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
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

    if batch_output.next_draft_input.verified_id.shape[0] <= 0:
        return None
    target_hidden = batch_output.logits_output.hidden_states
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

    mr0 = draft_worker._worker.model_runner
    mwb.spec_info_padded.hidden_states = target_hidden
    shared_fb = _make_forward_batch(mwb, mr0)
    shared_fb.bid = model_worker_batch.bid

    all_memory_pools = []
    all_leaves = []
    for w in [draft_worker._worker]:
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
    draft_verify_seq_lens = getattr(batch_output.next_draft_input, "verify_seq_lens", None)
    chain_all_ok = False
    if _CHAIN_ALL and draft_verify_seq_lens is not None:
        # Host-side arrays only: verify_seq_lens on the draft input is a device
        # array (the verify batch's seq_lens); reading it here would force a
        # device-to-host sync every draft step (measured +1.5 ms per cycle at
        # batch 1). model_worker_batch.seq_lens is the same value on the host.
        chain_all_ok = chain_all_headroom_ok(
            np.asarray(draft_allocate_lens),
            np.asarray(model_worker_batch.seq_lens),
            draft_worker.speculative_num_steps,
        )
        if not chain_all_ok and not getattr(draft_worker, "_chain_all_fallback_logged", False):
            draft_worker._chain_all_fallback_logged = True
            logger.warning(
                "SGLANG_JAX_MTP_CHAIN_POOL=all: allocation headroom below num_steps - 1 slots "
                "for some request; falling back to the two-version draft pool for this batch"
            )
    draft_allocate_lens = _prepare_device_array(
        draft_allocate_lens, data_sharding, "draft_extend.allocate_lens"
    )
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
        hf_config = getattr(getattr(mr0, "model_config", None), "hf_config", None)
        index_share = mtp_index_share_enabled(hf_config, draft_worker.topk)
        hidden_relay = mtp_hidden_relay_enabled(hf_config)
        n_blocks = len([draft_worker._worker])
        relay_resolved = _chained_relay(hidden_relay, draft_worker.speculative_num_steps, n_blocks)
        logger.info(
            "Fused draft extend: MTP IndexShare %s, hidden relay %s (%s; steps=%d blocks=%d)%s",
            "on" if index_share else "off",
            "on" if relay_resolved else "off",
            "forced by env" if hidden_relay is not None else "auto by chaining",
            draft_worker.speculative_num_steps,
            n_blocks,
            " (+positions)" if relay_resolved and _RELAY_POS else "",
        )
        draft_worker._fused_jit_fn = _build_draft_extend(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
            index_share=index_share,
            hidden_relay=hidden_relay,
        )

    with jax.set_mesh(draft_worker.mesh):
        _fused_out = draft_worker._fused_jit_fn(
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
            chain_all_ok=chain_all_ok,
        )
        if _CHAIN_POOL_CHECK and len(_fused_out) == 5:
            *_fused_out, _chain_report = _fused_out
            if _chain_report is not None:
                log_chain_pool_check(_chain_report, _fused_out[2][0])
        (
            selected_layer0_hidden,
            topk_index_stacked,
            all_pool_updates,
            updated_relay_buffers,
        ) = _fused_out

    for i, w in enumerate([draft_worker._worker]):
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
    if not pending_result.host_outputs_prefetched:
        jax.copy_to_host_async(next_verified_id)

    batch_output.next_draft_input.hidden_states = np.asarray(selected_layer0_hidden)[sel]
    topk_index = np.asarray(topk_index_stacked)[sel]
    batch_output.next_draft_input.topk_p = np.ones(topk_index.shape, dtype=np.float32)
    batch_output.next_draft_input.topk_index = topk_index
    batch_output.next_draft_input.verified_id = np.asarray(next_verified_id)[sel]
    batch_output.next_draft_input.allocate_lens = batch_output.next_draft_input.allocate_lens[
        : model_worker_batch.real_bs
    ]
    batch_output.next_draft_input.accept_length = accept_host
    batch_output.next_draft_input.accept_length_cpu = accept_host
    batch_output.accept_lens = accept_host


def draft_extend_for_decode(draft_worker, model_worker_batch, batch_output):
    """Drop-in replacement for MultiLayerDraftWorker.draft_extend_for_decode.

    Fuses all N MTP layer forwards into a single jit call.
    """
    pending_result = launch_fused_draft_extend_for_decode(
        draft_worker, model_worker_batch, batch_output
    )
    restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def launch_eagle3_recurrent_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
    *,
    relay_buffers=None,
    relay_future_indices=None,
    relay_valid_mask=None,
):
    """Launch all recurrent EAGLE3 draft stages and optionally publish relay state."""
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
        use_device_metadata=True,
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
    # Verify provides a device-sharded copy separately from the scheduler copy.
    # Host device_put(P() -> P("data")) can materialize the array on CPU and
    # block draft submission until verify finishes.
    next_new_seq_lens = _prepare_device_array(
        batch_output.next_draft_input.new_seq_lens_for_draft_extend,
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
                topk=draft_worker.topk,
            )
        )

    with jax.set_mesh(draft_worker.mesh):
        (
            selected_stage0_hidden,
            topk_index_stacked,
            pool_updates,
            updated_relay_buffers,
        ) = draft_worker._fused_eagle3_recurrent_draft_extend_jit_fn(
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


def eagle3_recurrent_draft_extend_for_decode(
    draft_worker,
    model_worker_batch,
    batch_output,
):
    """Run and restore recurrent EAGLE3 draft state for no-overlap decode."""
    pending_result = launch_eagle3_recurrent_draft_extend_for_decode(
        draft_worker,
        model_worker_batch,
        batch_output,
    )
    restore_draft_extend_result(draft_worker, model_worker_batch, pending_result)


def spec_prefill(spec_worker, model_worker_batch, launch_done=None, *, update_relay=False):
    """Run greedy prefill target forward and MTP draft-extend in one JIT."""
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
    )
    from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

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

    draft_mr0 = draft_worker._worker.model_runner
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
    for w in [draft_worker._worker]:
        mr = w.model_runner
        all_memory_pools.append(mr.memory_pools)
        all_leaves.append(tuple(mr.model_state_leaves))

    if not hasattr(draft_worker, "_fused_greedy_prefill_jit_fn"):
        hf_config = getattr(
            getattr(draft_worker._worker.model_runner, "model_config", None), "hf_config", None
        )
        draft_worker._fused_greedy_prefill_jit_fn = _build_prefill(
            num_layers=draft_worker.speculative_num_steps,
            topk=draft_worker.topk,
            hidden_relay=mtp_hidden_relay_enabled(hf_config),
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
    for i, w in enumerate([draft_worker._worker]):
        w.model_runner.memory_pools.replace_all(all_pool_updates[i])
    if update_relay:
        spec_worker.spec_relay_buffers = updated_relay_buffers

    sel = np.asarray(model_worker_batch.logits_indices_selector)
    if update_relay:
        from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

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
    if rebuild_verify_metadata or use_relay_state:
        # Relay verify replaces seq_lens from the device relay buffer and
        # rebuilds dynamic FA metadata inside fused_verify.  Upload only the
        # allocated page ids here; constructing the full host metadata would
        # enqueue several H2Ds whose values are immediately overwritten.
        target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_eagle_base_metadata(
            model_worker_batch
        )
    else:
        target_mr.attn_backend.forward_metadata = target_mr.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
    if (
        use_relay_state
        and getattr(target_mr.attn_backend.forward_metadata, "custom_mask", None) is not None
    ):
        raise NotImplementedError("Spec decode overlap relay path does not support custom_mask.")
    target_forward_batch = _make_forward_batch(model_worker_batch, target_mr)
    if rebuild_verify_metadata:
        target_forward_batch.attn_backend.forward_metadata.seq_lens = target_forward_batch.seq_lens
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
        draft_worker._fused_greedy_verify_jit_fn = _build_verify(
            topk=draft_worker.topk,
        )

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
                _prepare_device_array(np.full((_sv_tbs,), TOP_K_ALL, np.int32), data_sharding),
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
            prepared_new_seq_lens_data,
            target_mr._sampler_step,
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
    next_draft_input.new_seq_lens_for_draft_extend = prepared_new_seq_lens_data
    if draft_padding_prepared or use_relay_state:
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


def spec_decode_draft_extend(spec_worker, model_worker_batch, batch_output):
    """Run MTP draft extend as the second speculative decode JIT."""
    spec_worker.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)
    return batch_output


def spec_decode(spec_worker, model_worker_batch, cur_allocate_lens):
    """Run speculative decode as verify JIT followed by draft-extend JIT."""
    batch_output = spec_decode_verify(spec_worker, model_worker_batch, cur_allocate_lens)
    return spec_decode_draft_extend(spec_worker, model_worker_batch, batch_output)


def spec_decode_eagle3_overlap(spec_worker, model_worker_batch, cur_allocate_lens):
    """Launch fused EAGLE3 verify/recurrent draft and publish the next relay state."""
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
        draft_to_target_token_ids = draft_worker.prepare_for_fused_verify(model_worker_batch)
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
    pending_result = launch_eagle3_recurrent_draft_extend_for_decode(
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
