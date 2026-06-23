"""Stable (lowest-index tie-break) variant of the inference grouped top-k Pallas kernel.

V1 (`grouped_topk.py`) selects experts/groups with `jnp.argmax`. `jnp.argmax` is documented to
return the lowest index on ties, but on TPU the Mosaic lane-reduction does NOT guarantee that — it
can return a higher index — so experts/groups with equal post-bias scores can be emitted in a
different ORDER than `jax.lax.top_k` (which breaks ties toward the lowest index). The model output
is unchanged by a topk reorder (the {expert -> weight} set is identical and the combine sum is
order-invariant), but exact id-for-id parity with `lax.top_k` matters for validation and for any
consumer where the topk position is significant.

V2 keeps the exact same algorithm and still returns weights from the kernel (gathered from the
PRE-bias logits at the selected id), but replaces each selection `argmax` with an explicit
"max value, then smallest index achieving it" (`max` + masked `min`). This reproduces
`jax.lax.top_k`'s lowest-index tie-break deterministically, independent of the hardware argmax
tie-break. `group_top2` is unchanged: it only needs the SUM of the group's top-2 scores, which is
identical regardless of which tied element is masked first.

Separate module (not an in-place edit of `grouped_topk.py`) so V1/V2 can be A/B benchmarked.
"""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

try:
    from sgl_jax.srt.kernels.grouped_topk.grouped_topk import (
        NEG_INF,
        SAFE_AUTO_BT,
        _align_to,
        _largest_safe_divisor,
        get_interpret,
        logger,
    )
    from sgl_jax.srt.kernels.grouped_topk.tuned_block_sizes import get_tuned_bt
except Exception:  # noqa: BLE001  (base64-embedded standalone copy: names already at module scope)
    pass


def _grouped_topk_kernel_v2(
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

    # ① group score = sum of top-2 within each group, via 2-pass max. Tie-invariant (the SUM of the
    #    top-2 is the same regardless of which tied max is masked first), so it keeps the cheap argmax.
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

    # ② select `topk_group` groups, lowest-index tie-break (matches jax.lax.top_k).
    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((bt, n_group), dtype=jnp.bool_)
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, n_group), 1)
        tmp = group_scores
        for _ in range(topk_group):
            gmax = jnp.max(tmp, axis=1, keepdims=True)
            gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=1, keepdims=True)
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

    # ③ select `topk` experts, lowest-index tie-break (matches jax.lax.top_k). Weight is taken from
    #    the PRE-bias logits at the selected id (matches gate.py).
    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts), 1)
        cur = masked
        id_cols, w_cols = [], []
        for k in range(topk):
            cmax = jnp.max(cur, axis=1, keepdims=True)
            idx = jnp.min(
                jnp.where(cur == cmax, e_iota, num_experts), axis=1, keepdims=True
            )  # [BT,1]
            sel = e_iota == idx  # [BT, E]
            wval = jnp.sum(jnp.where(sel, logits, 0.0), axis=1, keepdims=True)  # [BT, 1]
            id_cols.append(idx.astype(jnp.int32))
            w_cols.append(wval.astype(jnp.float32))
            if k != topk - 1:
                cur = jnp.where(sel, NEG_INF, cur)

    # Pad the minor (lane) dim topk -> padded_topk (multiple of 128); wrapper slices [:, :topk].
    n_pad = padded_topk - topk
    if n_pad > 0:
        id_cols.append(jnp.full((bt, n_pad), -1, dtype=jnp.int32))
        w_cols.append(jnp.zeros((bt, n_pad), dtype=jnp.float32))

    ids_ref[...] = jnp.concatenate(id_cols, axis=1)  # [BT, padded_topk]
    w_ref[...] = jnp.concatenate(w_cols, axis=1)  # [BT, padded_topk]


def grouped_topk_pallas_v2(
    router_logits: jax.Array,  # [BS, E] (any float; cast to f32 inside)
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
):
    """Biased grouped top-k via argmax-selection with a stable lowest-index tie-break.

    Drop-in for `grouped_topk_pallas`; matches `jax.lax.top_k` id-for-id including on tied
    post-bias scores. Returns (topk_weights[BS,k], topk_ids[BS,k]).
    """
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        tuned = get_tuned_bt(bs, e, num_expert_group, topk_group, topk)
        if tuned is not None and bs % tuned == 0:
            bt = tuned
        elif bs % 512 == 0:
            bt = min(512, bs)
        else:
            bt = _largest_safe_divisor(bs) or bs
        if bt > SAFE_AUTO_BT:
            logger.warning(
                "grouped_topk_v2: auto block_tokens fell back to whole-batch BT=%d (BS=%d has no "
                "VMEM-safe divisor); a single [%d,%d] tile may exceed VMEM.",
                bt,
                bs,
                bs,
                e,
            )
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)
    kernel = functools.partial(
        _grouped_topk_kernel_v2,
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
        name="grouped-topk-v2",
    )(router_logits, bias)
    return weights[:, :topk], ids[:, :topk]
