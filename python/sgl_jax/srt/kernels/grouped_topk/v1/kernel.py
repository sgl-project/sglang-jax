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
`argmax`, because TPU Mosaic's reduction argmax does not break ties toward the lowest index. The
iota is carried in f32 (exact for E <= 2**24): the vector core has float min/max but no integer
min/max, so an int32 masked-min costs a compare+select on every vreg of the [E,BT] block.

All-groups fast path (`topk_group == n_group`, e.g. `n_group = 1`): ①-② cannot change the expert
set, so they are skipped and the correction bias is added in the native `[BT, E]` layout BEFORE
the transpose -- the `[E]` bias broadcasts along sublanes instead of along lanes, and only the
post-bias scores are transposed for selection (the pre-bias logits are transposed separately for
the weight gather). Same pick loop, same arithmetic, bit-identical ids and weights
(`test/kernels/grouped_topk_parity_test.py` pins both paths against the pre-#1758 kernel).

Contract (checked in `grouped_topk_pallas` before tracing, `ValueError` otherwise):
`correction_bias.shape == (E,)`, `1 <= num_expert_group` dividing `E`,
`1 <= topk_group <= num_expert_group`, `1 <= topk <= topk_group * E / num_expert_group`, and
`E <= 65536` when `packed=True`.
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

# `packed=True` stores `E-1-id` in the low 16 bits of the int32 sort key (see `build_key` below).
_PACKED_MAX_EXPERTS = 1 << 16


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


def _validate_config(
    *,
    router_logits: jax.Array,
    correction_bias: jax.Array,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    packed: bool,
) -> None:
    """Rejects configurations the routing contract does not define, before anything is traced.

    The JAX reference (`gate.py:_biased_grouped_topk_jax`) cannot evaluate these (e.g.
    `lax.top_k(group_scores, k=topk_group)` with `topk_group > num_expert_group`) or would route to
    experts of dropped groups; the kernel used to return input-dependent ids instead of failing.
    """
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be [BS, E], got shape {router_logits.shape}")
    e = router_logits.shape[-1]
    if correction_bias.shape != (e,):
        raise ValueError(
            f"correction_bias must have shape ({e},) to match router_logits {router_logits.shape}, "
            f"got {correction_bias.shape}"
        )
    if num_expert_group < 1:
        raise ValueError(f"num_expert_group must be >= 1, got {num_expert_group}")
    if e % num_expert_group != 0:
        raise ValueError(
            f"num_experts={e} must be divisible by num_expert_group={num_expert_group}"
        )
    if not 1 <= topk_group <= num_expert_group:
        raise ValueError(
            f"topk_group={topk_group} must satisfy 1 <= topk_group <= num_expert_group="
            f"{num_expert_group} (the reference `lax.top_k(group_scores, k=topk_group)` cannot "
            f"select more groups than exist)"
        )
    retained = topk_group * (e // num_expert_group)
    if not 1 <= topk <= retained:
        raise ValueError(
            f"topk={topk} must satisfy 1 <= topk <= topk_group * experts_per_group = "
            f"{topk_group} * {e // num_expert_group} = {retained}; picking more experts than the "
            f"retained groups hold would route to experts of dropped groups"
        )
    if packed and e > _PACKED_MAX_EXPERTS:
        raise ValueError(
            f"packed=True keeps the expert id in 16 bits of the sort key: num_experts={e} exceeds "
            f"{_PACKED_MAX_EXPERTS}"
        )


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

    if topk_group == n_group:
        # Every group is retained (`grouped_topk_pallas` enforces topk_group <= n_group): group
        # scores cannot affect the expert set, so ①-③ are skipped. Add the bias in the native
        # [BT, E] layout, where the [E] vector broadcasts along sublanes (one vreg row replicated)
        # instead of along lanes, and transpose the post-bias scores directly. The pre-bias logits
        # are transposed separately for the weight gather in ④.
        with jax.named_scope("bias_add"):
            masked = (
                logits_ref[...].astype(jnp.float32) + jax.lax.expand_dims(bias_ref[...], (0,))
            ).T  # [E, BT] post-bias
        logits = logits_ref[...].astype(jnp.float32).T  # [E, BT] pre-bias (weights)
    else:
        # Transpose to [E, BT]: experts in sublane, tokens in lane. Every reduction below is over axis 0.
        logits = logits_ref[...].astype(jnp.float32).T  # [E, BT] pre-bias
        with jax.named_scope("bias_add"):
            scores = logits + bias_ref[...][:, None]  # [E, BT] post-bias

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

    # ④ select `topk` experts, lowest-index tie-break; weight = PRE-bias logit at the winner. A
    #    fori_loop carries the [E,BT] working array and writes each pick into ROW k of the [topk,BT]
    #    outputs, so per-block VMEM stays O(E*BT), independent of topk. Fully unrolled (topk is
    #    small and static) so the picks overlap.
    #
    #    Two selection modes (compile-time `packed`):
    #      packed=False — the f32 contract: `max` + masked-`min` finds the smallest expert id at the
    #        max score, bit-exact to `lax.top_k` on the f32 scores.
    #      packed=True  — the bf16 contract: bf16-round each score into an int32 order-preserving key
    #        (plain int order == (score DESC, index ASC)), so each pick is ONE reduction + a low-bit
    #        decode instead of the max+masked-min pair. Lossless for bf16 inputs (the low 16 mantissa
    #        bits are zero, so packing the id into those 16 bits discards nothing). See gate.py:
    #        the caller selects packed only when router_logits is bf16.
    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (E, bt), 0)
        # The same expert ids carried in f32 (`tpu.iota` is integer-only, so this is one
        # loop-invariant convert). Exact for every E a TPU can hold (E <= 2**24), so 0..E
        # round-trip through f32 unchanged; `_pick` reduces and compares against this copy
        # because the vector core has float min/max but no integer min/max.
        e_iota_f = e_iota.astype(jnp.float32)
        e_sentinel_f = jnp.float32(E)
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (topk, bt), 0)
        ids_init = jnp.full((topk, bt), -1, dtype=jnp.int32)
        w_init = jnp.zeros((topk, bt), dtype=jnp.float32)

        if packed:
            # Index goes in the low 16 bits (bf16-rounded mantissa, all zero; score lives in bits
            # 16-31). op count is identical to a tighter b (masks are compile-time constants).
            # E-1 (<=511) fits in 16 bits.
            low_mask = jnp.int32(0xFFFF)
            clear_mask = jnp.int32(-(1 << 16))  # 0xFFFF0000: clears the low 16 index bits
            with jax.named_scope("build_key"):
                sb = masked.astype(jnp.bfloat16).astype(jnp.float32)  # bf16-round: low 16 bits zero
                si = jax.lax.bitcast_convert_type(sb, jnp.int32)
                # flip low 31 bits for negatives so signed int32 compares in float order (incl -inf)
                key_score = si ^ ((si >> 31) & jnp.int32(0x7FFFFFFF))
                work0 = (key_score & clear_mask) | (E - 1 - e_iota)  # [E, BT] packed key
        else:
            work0 = masked  # [E, BT] f32 working scores

        def _pick(k, carry):
            cur, ids_buf, w_buf = carry
            if packed:
                kmax = jnp.max(cur, axis=0, keepdims=True)  # [1, BT] single reduction
                idx = (E - 1) - (kmax & low_mask)  # [1, BT] lowest-index winner from the low bits
                idxf = idx.astype(jnp.float32)
            else:
                cmax = jnp.max(cur, axis=0, keepdims=True)
                # Masked-min over the f32 expert ids: same exact integers as the int32 form,
                # but the reduction lowers to `multi_reduction<minimumf>` -> one `vmin` per
                # vreg, whereas `<minsi>` has no integer-min instruction behind it and is
                # expanded to a `vlt.s32` + `vsel` pair on every vreg of the [E,BT] block.
                idxf = jnp.min(
                    jnp.where(cur == cmax, e_iota_f, e_sentinel_f), axis=0, keepdims=True
                )  # [1, BT] lowest expert id achieving the max (E when no element matches)
                idx = idxf.astype(jnp.int32)
            # The winner one-hot feeds two consumers (weight gather, drop-the-winner). Kept as
            # ONE [E,BT] i1 value it outlives both uses, so Mosaic materialises it, spills it to
            # VMEM and re-expands it (`vnez.u8`) at each use. Comparing twice — once in f32 for
            # the gather, once in i32 for the update, on the same winner id — is one compare per
            # use with no shared array: neither CSE nor Mosaic can merge a cmpf with a cmpi, and
            # each mask dies into its consumer inside a vector mask register.
            # weight from PRE-bias logits via masked sum (gather is unsupported in Pallas/Mosaic).
            wval = jnp.sum(
                jnp.where(e_iota_f == idxf, logits, 0.0), axis=0, keepdims=True
            )  # [1, BT]
            write = row_iota == k  # [topk, BT] one-hot on row k (loop index)
            ids_buf = jnp.where(write, idx.astype(jnp.int32), ids_buf)
            w_buf = jnp.where(write, wval.astype(jnp.float32), w_buf)
            # drop the winner
            cur = jnp.where(e_iota == idx, _I32_MIN if packed else NEG_INF, cur)
            return cur, ids_buf, w_buf

        _, ids_out, w_out = jax.lax.fori_loop(
            0, topk, _pick, (work0, ids_init, w_init), unroll=True
        )

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

    Raises `ValueError` (before tracing the kernel) for configurations outside the routing
    contract: `correction_bias` not `[E]`, `num_expert_group < 1` or not dividing `E`,
    `topk_group` outside `[1, num_expert_group]`, `topk` outside
    `[1, topk_group * E / num_expert_group]`, or `packed=True` with more than 65536 experts.
    """
    _validate_config(
        router_logits=router_logits,
        correction_bias=correction_bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        packed=packed,
    )
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
