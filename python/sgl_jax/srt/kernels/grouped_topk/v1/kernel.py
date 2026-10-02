"""Grouped top-k MoE routing — Pallas TPU kernel (stable lowest-index tie-break).

This is the routing of `gate.py:TopK._biased_grouped_topk` (DeepSeek-V3 noaux_tc) done WITHOUT any
`sort`, entirely via `max`/`argmax` selection, fully VMEM-resident in one Pallas kernel. It is
**id-for-id identical to `jax.lax.top_k`** including exact-tie order (lowest expert index wins).

Design — tokens in the lane dim (`[E, BT]`):
    The block is loaded `[BT, E]` and transposed to `[E, BT]` so experts sit in the sublane/major
    dim and tokens in the 128-wide lane/minor dim. Every top-k reduction then runs over the
    sublane axis, processing 128 tokens in parallel per step — no cross-lane permute (the slow path
    a `[BT, E]` layout would hit reducing over experts-in-lanes). Outputs are written `[topk, BT]`
    (BT in lanes, dense) and returned as `[BS, topk]` via a `.T` that lowers to a free bitcast.

Algorithm (matches `_biased_grouped_topk` exactly, ties included):
    scores = router_logits + correction_bias                    # post-bias "scores_for_choice"
    ① group score = sum of top-2 per group (2-pass max, no sort)
    ② select `topk_group` groups        (max + masked-min: lowest-index tie-break)
    ③ mask dropped groups to -inf, select `topk` experts (max + masked-min), weight = PRE-bias logit
Renormalize / routed_scaling_factor are applied by the caller (`TopK.__call__`).

Tie-break: selection uses `max` + masked `min(iota)` (smallest index achieving the max) rather than
`argmax`, because TPU Mosaic's reduction argmax does not break ties toward the lowest index.
"""

from __future__ import annotations

import functools
import logging
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

logger = logging.getLogger(__name__)

NEG_INF = -jnp.inf
_I32_MIN = jnp.iinfo(jnp.int32).min

# Largest token block (grid>1) known to fit v7x VMEM with double-buffered [E,BT] inputs. The "auto"
# path never tiles above this without warning.
SAFE_AUTO_BT = 2048


def _largest_safe_divisor(bs: int, cap: int = SAFE_AUTO_BT, align: int = 128) -> int | None:
    """Largest d dividing bs with d <= cap and d % align == 0, else None.

    Tokens land in the lane dim, so the block must be a 128-multiple (lane width). Returns None when
    bs has no such divisor (e.g. prime / not 128-aligned) so the caller falls back to one block.
    """
    hi = (min(cap, bs) // align) * align
    for d in range(hi, 0, -align):
        if bs % d == 0:
            return d
    return None


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _grouped_topk_kernel(
    logits_ref,  # [BT, E] f32  (router_logits, PRE-bias) — loaded token-major
    bias_ref,  # [E]     f32  (correction_bias)
    w_ref,  # [topk, BT] f32  out: weights   (topk in sublane, BT in lane)
    ids_ref,  # [topk, BT] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    packed: bool = False,
):
    S = num_experts // n_group
    E = num_experts
    bt = logits_ref.shape[0]
    bias_f = bias_ref[...].astype(jnp.float32)

    if topk_group < n_group:
        # Transpose to [E, BT]: experts in sublane, tokens in lane. Every reduction below is over axis 0.
        logits = logits_ref[...].astype(jnp.float32).T  # [E, BT] pre-bias
        with jax.named_scope("bias_add"):
            scores = logits + bias_f[:, None]  # [E, BT] post-bias
        # ① group score = sum of top-2 within each group, via 2-pass max (no sort). argmax tie-break is
        #    irrelevant here — the top-2 sum is identical whichever of two equal maxima is masked first.
        with jax.named_scope("group_top2"):
            sg = jnp.reshape(scores, (n_group, S, bt))  # [G, S, BT]
            v1 = jnp.max(sg, axis=1, keepdims=True)
            i1 = jnp.argmax(sg, axis=1, keepdims=True)
            s_iota = jax.lax.broadcasted_iota(jnp.int32, (n_group, S, bt), 1)
            v2 = jnp.max(jnp.where(s_iota == i1, NEG_INF, sg), axis=1, keepdims=True)
            group_scores = jnp.squeeze(v1 + v2, axis=1)  # [G, BT]

        # ② select `topk_group` groups, lowest-index tie-break (max + masked-min).
        with jax.named_scope("group_select"):
            group_mask = jnp.zeros((n_group, bt), dtype=jnp.bool_)
            g_iota = jax.lax.broadcasted_iota(jnp.int32, (n_group, bt), 0)
            tmp = group_scores
            for _ in range(topk_group):
                gmax = jnp.max(tmp, axis=0, keepdims=True)
                gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=0, keepdims=True)
                m = g_iota == gi
                group_mask = jnp.logical_or(group_mask, m)
                tmp = jnp.where(m, NEG_INF, tmp)

        # ③ mask experts in dropped groups -> -inf. Applied ONCE (loop-invariant) before the pick loop.
        with jax.named_scope("expert_mask"):
            masked = jnp.reshape(
                jnp.where(group_mask[:, None, :], jnp.reshape(scores, (n_group, S, bt)), NEG_INF),
                (E, bt),
            )  # [E, BT]
    else:
        with jax.named_scope("bias_add"):
            masked = (
                logits_ref[...].astype(jnp.float32) + jax.lax.expand_dims(bias_f, (0,))
            ).T  # [E, BT] post-bias

    # ④ select `topk` experts, lowest-index tie-break; weight = PRE-bias logit at the winner.
    #    Decouple Phase A (selecting the `topk` winning expert indices from `cur`) from Phase B
    #    (gathering pre-bias weights and assembling `[topk, BT]` outputs) so `logits`,
    #    `ids_out`, and `w_out` are not live during the `cur` reduction/masking loop.
    with jax.named_scope("final_select"):
        e_iota_1d = jax.lax.broadcasted_iota(jnp.int32, (E, 1), 0)

        if packed:
            logits = logits_ref[...].astype(jnp.float32).T if topk_group >= n_group else logits
            e_iota = jnp.broadcast_to(e_iota_1d, (E, bt))
            low_mask = jnp.int32(0xFFFF)
            clear_mask = jnp.int32(-(1 << 16))  # 0xFFFF0000: clears the low 16 index bits
            with jax.named_scope("build_key"):
                sb = masked.astype(jnp.bfloat16).astype(jnp.float32)  # bf16-round: low 16 bits zero
                si = jax.lax.bitcast_convert_type(sb, jnp.int32)
                # flip low 31 bits for negatives so signed int32 compares in float order (incl -inf)
                key_score = si ^ ((si >> 31) & jnp.int32(0x7FFFFFFF))
                cur = (key_score & clear_mask) | (E - 1 - e_iota)  # [E, BT] packed key

            idx_i_list = []
            for k in range(topk):
                kmax = jnp.max(cur, axis=0, keepdims=True)  # [1, BT] single reduction
                idx_i = (E - 1) - (kmax & low_mask)  # [1, BT] lowest-index winner from the low bits
                idx_i_list.append(idx_i)
                if k < topk - 1:
                    cur = jnp.where(e_iota == idx_i, _I32_MIN, cur)

            row_iota = jnp.broadcast_to(
                jax.lax.broadcasted_iota(jnp.int32, (topk, 1), 0), (topk, bt)
            )
            for k in range(topk):
                idx_i = idx_i_list[k]
                wval = jnp.sum(jnp.where(e_iota == idx_i, logits, 0.0), axis=0, keepdims=True)
                if k == 0:
                    ids_out = jnp.broadcast_to(idx_i, (topk, bt))
                    w_out = jnp.broadcast_to(wval.astype(jnp.float32), (topk, bt))
                else:
                    write = row_iota == k
                    ids_out = jnp.where(write, idx_i, ids_out)
                    w_out = jnp.where(write, wval.astype(jnp.float32), w_out)
        else:
            e_iota_f = jnp.broadcast_to(e_iota_1d.astype(jnp.float32), (E, bt))
            e_bound_f = float(E)
            cur = masked  # [E, BT] f32 working scores

            idx_f_list = []
            for k in range(topk):
                cmax = jnp.max(cur, axis=0, keepdims=True)
                idx_f = jnp.min(
                    jnp.where(cur == cmax, e_iota_f, e_bound_f), axis=0, keepdims=True
                )  # [1, BT] lowest expert id achieving the max via f32 minimumf
                idx_f_list.append(idx_f)
                if k < topk - 1:
                    cur = jnp.where(e_iota_f == idx_f, NEG_INF, cur)

            row_iota_f = jnp.broadcast_to(
                jax.lax.broadcasted_iota(jnp.int32, (topk, 1), 0).astype(jnp.float32),
                (topk, bt),
            )
            ids_f_out = jnp.broadcast_to(idx_f_list[0], (topk, bt))
            for k in range(1, topk):
                ids_f_out = jnp.where(row_iota_f == float(k), idx_f_list[k], ids_f_out)
            ids_out = ids_f_out.astype(jnp.int32)

            logits = logits_ref[...].astype(jnp.float32).T if topk_group >= n_group else logits
            e_iota = jnp.broadcast_to(e_iota_1d, (E, bt))
            for k in range(topk):
                idx_i = idx_f_list[k].astype(jnp.int32)
                wval = jnp.sum(jnp.where(e_iota == idx_i, logits, 0.0), axis=0, keepdims=True)
                if k == 0:
                    w_out = jnp.broadcast_to(wval.astype(jnp.float32), (topk, bt))
                else:
                    w_out = jnp.where(row_iota_f == float(k), wval.astype(jnp.float32), w_out)

    ids_ref[...] = ids_out  # [topk, BT]
    w_ref[...] = w_out


def grouped_topk_pallas(
    router_logits: jax.Array,  # [BS, E] (any float; cast to f32 inside)
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
    packed: bool = False,
):
    """Biased grouped top-k via argmax-selection. Returns (topk_weights[BS,k], topk_ids[BS,k]).

    Drop-in for `gate.py:TopK._biased_grouped_topk` (renormalize / routed_scaling_factor applied by
    the caller). `block_tokens="auto"` picks the largest 128-aligned divisor of BS (tokens are in the
    lane dim), falling back to a single whole-batch block. The final-select `fori_loop` is fully
    unrolled (topk is small and static).

    `packed=True` uses the bf16 packed-key final select (single reduction per pick, bit-exact to
    `lax.top_k` at bf16 precision). It is lossless only for bf16 inputs, so the caller enables it
    exactly when router_logits is bf16; the default f32 path is unchanged.
    """
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        bt = _largest_safe_divisor(bs, cap=SAFE_AUTO_BT, align=128) or bs
        if bt > SAFE_AUTO_BT:
            logger.warning(
                "grouped_topk: auto block_tokens fell back to whole-batch BT=%d (BS=%d has no "
                "128-aligned VMEM-safe divisor); a single [%d,%d] tile may exceed VMEM. Pad local "
                "tokens to a multiple of 128 or pass an explicit block_tokens.",
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

    kernel = functools.partial(
        _grouped_topk_kernel,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        packed=packed,
    )
    # Kernel emits [topk, BS] (BS in lanes, dense); the `.T` to the [BS, topk] contract lowers to a
    # free bitcast ([topk,BS]{1,0} == [BS,topk]{0,1}), avoiding an output relayout copy.
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((topk, bt), lambda i: (0, i)),
            pl.BlockSpec((topk, bt), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, bs), jnp.float32),
            jax.ShapeDtypeStruct((topk, bs), jnp.int32),
        ],
        interpret=interpret,
        name="grouped-topk-packed" if packed else "grouped-topk",
    )(router_logits, bias)
    return weights_t.T, ids_t.T
