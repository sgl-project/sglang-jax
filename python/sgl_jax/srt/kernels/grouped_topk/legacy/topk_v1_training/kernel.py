"""Training-safe ids-only Pallas kernel for biased grouped top-k routing.

The Pallas kernel returns only discrete top-k expert ids. The differentiable top-k weights are
gathered outside the kernel by ``grouped_topk_pallas_training`` so training can use JAX's normal
``take_along_axis`` VJP instead of requiring a custom transpose for the Pallas call.
"""

from __future__ import annotations

import functools
import logging
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

try:
    from sgl_jax.srt.kernels.grouped_topk.tuned_block_sizes import get_tuned_bt
except Exception:  # noqa: BLE001  (e.g. standalone embedded copies without the package)

    def get_tuned_bt(*_a, **_k):  # noqa: ANN002, ANN003
        return None


logger = logging.getLogger(__name__)

NEG_INF = -jnp.inf
SAFE_AUTO_BT = 2048


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _largest_safe_divisor(bs: int, cap: int = SAFE_AUTO_BT, align: int = 8) -> int | None:
    hi = (min(cap, bs) // align) * align
    for d in range(hi, 0, -align):
        if bs % d == 0:
            return d
    return None


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _grouped_topk_ids_kernel(
    logits_ref,  # [BT, E] f32  (router_logits, post-score-func, PRE-bias)
    bias_ref,  # [E]     f32  (correction_bias)
    ids_ref,  # [BT, padded_topk] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    padded_topk: int,
):
    experts_per_group = num_experts // n_group
    logits = logits_ref[...].astype(jnp.float32)
    scores = logits + bias_ref[...][None, :]
    bt = scores.shape[0]

    with jax.named_scope("group_top2"):
        group_score_cols = []
        for g in range(n_group):
            group_scores = scores[:, g * experts_per_group : (g + 1) * experts_per_group]
            v1 = jnp.max(group_scores, axis=1, keepdims=True)
            i1 = jnp.argmax(group_scores, axis=1, keepdims=True)
            group_iota = jax.lax.broadcasted_iota(jnp.int32, group_scores.shape, 1)
            group_scores_without_top1 = jnp.where(group_iota == i1, NEG_INF, group_scores)
            v2 = jnp.max(group_scores_without_top1, axis=1, keepdims=True)
            group_score_cols.append(v1 + v2)
        group_scores = jnp.concatenate(group_score_cols, axis=1)

    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((bt, n_group), dtype=jnp.bool_)
        group_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, n_group), 1)
        tmp_group_scores = group_scores
        for _ in range(topk_group):
            group_idx = jnp.argmax(tmp_group_scores, axis=1, keepdims=True)
            selected_group = group_iota == group_idx
            group_mask = jnp.logical_or(group_mask, selected_group)
            tmp_group_scores = jnp.where(selected_group, NEG_INF, tmp_group_scores)

    with jax.named_scope("expert_mask"):
        masked_slices = []
        for g in range(n_group):
            group_selected = group_mask[:, g : g + 1]
            masked_slices.append(
                jnp.where(
                    group_selected,
                    scores[:, g * experts_per_group : (g + 1) * experts_per_group],
                    NEG_INF,
                )
            )
        masked_scores = jnp.concatenate(masked_slices, axis=1)

    with jax.named_scope("final_select_ids"):
        expert_iota = jax.lax.broadcasted_iota(jnp.int32, (bt, num_experts), 1)
        cur = masked_scores
        id_cols = []
        for k_idx in range(topk):
            expert_idx = jnp.argmax(cur, axis=1, keepdims=True)
            selected_expert = expert_iota == expert_idx
            id_cols.append(expert_idx.astype(jnp.int32))
            if k_idx != topk - 1:
                cur = jnp.where(selected_expert, NEG_INF, cur)

    n_pad = padded_topk - topk
    if n_pad > 0:
        id_cols.append(jnp.full((bt, n_pad), -1, dtype=jnp.int32))

    ids_ref[...] = jnp.concatenate(id_cols, axis=1)


def _resolve_block_tokens(
    bs: int,
    e: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str,
) -> int:
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
                "grouped_topk_ids: auto block_tokens fell back to whole-batch BT=%d "
                "(BS=%d has no VMEM-safe divisor); a single [%d,%d] tile may exceed VMEM.",
                bt,
                bs,
                bs,
                e,
            )
        return bt

    bt = min(block_tokens, bs)
    if bs % bt != 0:
        raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    return bt


def grouped_topk_ids_pallas(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
):
    """Return biased grouped top-k expert ids only."""
    bs, e = router_logits.shape
    if e % num_expert_group != 0:
        raise ValueError(f"E={e} must be divisible by num_expert_group={num_expert_group}")
    if not (1 <= topk_group <= num_expert_group):
        raise ValueError(
            f"topk_group={topk_group} must be in [1, num_expert_group={num_expert_group}]"
        )

    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)
    bt = _resolve_block_tokens(bs, e, num_expert_group, topk_group, topk, block_tokens)
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)
    kernel = functools.partial(
        _grouped_topk_ids_kernel,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        padded_topk=padded_topk,
    )
    ids = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((bs, padded_topk), jnp.int32),
        interpret=interpret,
        name="grouped-topk-ids",
    )(router_logits, bias)
    return ids[:, :topk]


def grouped_topk_pallas_training(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
):
    """Training-friendly grouped top-k: ids from Pallas, weights from JAX gather."""
    topk_ids = grouped_topk_ids_pallas(
        jax.lax.stop_gradient(router_logits),
        jax.lax.stop_gradient(correction_bias),
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        block_tokens=block_tokens,
        interpret=interpret,
    )
    topk_weights = jnp.take_along_axis(router_logits.astype(jnp.float32), topk_ids, axis=1)
    return topk_weights, topk_ids
