"""Standalone Pallas TPU kernel for biased grouped top-k MoE routing.

This is the routing of `gate.py:TopK._biased_grouped_topk` (DeepSeek-V3 noaux_tc) done
WITHOUT any `sort` — entirely via `max`/`argmax` selection, fully VMEM-resident inside one
Pallas kernel. It mirrors the in-kernel `get_top_k` of the v1 fused-MoE kernel
(`kernels/fused_moe/v1/kernel.py`) but is a self-contained, separately benchmarkable op.

Why no sort: on TPU `jax.lax.top_k` lowers to a `stablehlo.sort` (a bitonic comparison
network) that is bound by the VPU's cross-lane permute throughput (~8% of VPU peak). Selecting
top-k by iterated `argmax` (+ masking the winner) is a sequence of plain reduces — it runs on
the much faster reduce path and touches fewer elements for small k. See
`work/group-topk-kernel/analysis-zh.md`.

Algorithm (matches `_biased_grouped_topk` exactly, id-for-id):
  scores = router_logits + correction_bias                         # post-bias "scores_for_choice"
  ① group score = sum of top-2 per group, via 2-pass max           # no sort
  ② select `topk_group` groups, via iterated argmax                # no sort
  ③ mask dropped groups to -inf, select `topk` experts via argmax  # no sort
  weights = router_logits[selected_ids]   (PRE-bias logits)        # like gate.py
Renormalize / routed_scaling are left to the caller (as in `TopK.__call__`).
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

try:
    from sgl_jax.srt.kernels.grouped_topk.tuned_block_sizes import get_tuned_bt
except Exception:  # noqa: BLE001  (e.g. base64-embedded standalone copy without the package)

    def get_tuned_bt(*_a, **_k):  # noqa: ANN002, ANN003
        return None


NEG_INF = -jnp.inf


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _grouped_topk_kernel(
    logits_ref,  # [BT, E] f32  (router_logits, post-score-func, PRE-bias)
    bias_ref,  # [E]     f32  (correction_bias)
    w_ref,  # [BT, padded_topk] f32  out: weights
    ids_ref,  # [BT, padded_topk] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    padded_topk: int,
):
    S = num_experts // n_group
    logits = logits_ref[...].astype(jnp.float32)  # pre-bias
    with jax.named_scope("bias_add"):
        scores = logits + bias_ref[...][None, :]  # post-bias [BT, E]
    bt = scores.shape[0]

    # ① group score = sum of top-2 within each group, via 2-pass max (no sort)
    with jax.named_scope("group_top2"):
        g_scores = []
        for g in range(n_group):
            sl = scores[:, g * S : (g + 1) * S]  # [BT, S]
            v1 = jnp.max(sl, axis=1, keepdims=True)
            i1 = jnp.argmax(sl, axis=1, keepdims=True)
            io = jax.lax.broadcasted_iota(jnp.int32, sl.shape, 1)
            sl_masked = jnp.where(io == i1, NEG_INF, sl)
            v2 = jnp.max(sl_masked, axis=1, keepdims=True)
            g_scores.append(v1 + v2)
        group_scores = jnp.concatenate(g_scores, axis=1)  # [BT, G]

    # ② select `topk_group` groups, via iterated argmax (no sort)
    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((bt, n_group), dtype=jnp.bool_)
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, n_group), 1)
        tmp = group_scores
        for _ in range(topk_group):
            gi = jnp.argmax(tmp, axis=1, keepdims=True)
            m = g_iota == gi
            group_mask = jnp.logical_or(group_mask, m)
            tmp = jnp.where(m, NEG_INF, tmp)

    # mask experts in dropped groups → -inf (per-group where + concat)
    with jax.named_scope("expert_mask"):
        masked_slices = []
        for g in range(n_group):
            gm = group_mask[:, g : g + 1]  # [BT, 1]
            masked_slices.append(jnp.where(gm, scores[:, g * S : (g + 1) * S], NEG_INF))
        masked = jnp.concatenate(masked_slices, axis=1)  # [BT, E]

    # ③ select `topk` experts, via iterated argmax (no sort).
    #    weight is taken from the PRE-bias logits at the selected id (matches gate.py).
    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts), 1)
        cur = masked
        id_cols, w_cols = [], []
        for k in range(topk):
            idx = jnp.argmax(cur, axis=1, keepdims=True)  # [BT, 1]
            sel = e_iota == idx  # [BT, E]
            wval = jnp.sum(jnp.where(sel, logits, 0.0), axis=1, keepdims=True)  # [BT, 1]
            id_cols.append(idx.astype(jnp.int32))
            w_cols.append(wval.astype(jnp.float32))
            if k != topk - 1:
                cur = jnp.where(sel, NEG_INF, cur)

    # Pad the minor (lane) dim topk -> padded_topk (multiple of 128) so the output VMEM tile is
    # lane-aligned on TPU. Filler ids = -1, filler weights = 0.0; the wrapper slices [:, :topk]
    # outside the kernel, so the padding is transparent to callers.
    n_pad = padded_topk - topk
    if n_pad > 0:
        id_cols.append(jnp.full((bt, n_pad), -1, dtype=jnp.int32))
        w_cols.append(jnp.zeros((bt, n_pad), dtype=jnp.float32))

    ids_ref[...] = jnp.concatenate(id_cols, axis=1)  # [BT, padded_topk]
    w_ref[...] = jnp.concatenate(w_cols, axis=1)  # [BT, padded_topk]


def grouped_topk_pallas(
    router_logits: jax.Array,  # [BS, E] (any float; cast to f32 inside)
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
):
    """Biased grouped top-k via argmax-selection. Returns (topk_weights[BS,k], topk_ids[BS,k]).

    Drop-in for `gate.py:TopK._biased_grouped_topk` (renormalize / routed_scaling_factor are
    applied by the caller, exactly as in `TopK.__call__`).

    `block_tokens="auto"` (default) looks up the tuned BT for this device + (BS, E, G, Gtop, k)
    and falls back to a safe default on a miss; an explicit int forces that block size.
    """
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        tuned = get_tuned_bt(bs, e, num_expert_group, topk_group, topk)
        if tuned is not None and bs % tuned == 0:
            bt = tuned
        elif bs % min(512, bs) == 0:
            bt = min(512, bs)  # safe default
        else:
            bt = bs  # single block always divides
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)  # TPU VMEM output tile needs a 128-multiple minor dim
    kernel = functools.partial(
        _grouped_topk_kernel,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        padded_topk=padded_topk,
    )
    weights, ids = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.float32),
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.int32),
        ],
        interpret=interpret,
        name="grouped-topk",
    )(router_logits, bias)
    return weights[:, :topk], ids[:, :topk]
