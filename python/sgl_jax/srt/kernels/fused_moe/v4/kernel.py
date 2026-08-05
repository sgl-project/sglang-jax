# Copyright 2025.
"""TPU MoE kernel v4 — TP (tensor-parallel) grouped/ragged matmul (bf16-only).

In EP (v1/v2) each chip holds 1/tp of the experts at full intermediate width
and communicates tokens via all-to-all scatter/gather + barrier. The profile
upstream (AInfer v4 commit) showed 40% of decode wall-clock spent on this
ICI coordination (barrier 16%, AR/AG 20%, scatter/gather 3%).

v4 flips to TP: every chip holds ALL experts but only a 1/tp slice of the
intermediate dimension (gate: I/tp, up: I/tp, down: I/tp). Each chip
independently runs the full grouped FFN over the active experts with its
weight slice, then a single psum (all-reduce) across tp combines the partial
down-projection results. This eliminates scatter/gather/barrier entirely.

Weight HBM traffic is conserved: EP reads ~active_experts × full width ≈
TP reads (more) active experts × 1/tp width. The win is purely from
eliminating the ICI coordination overhead at decode.

Gate/up are kept as SEPARATE w1/w3 arrays, each TP-sharded on the I
dimension. Pre-fusing them into one [E, H, 2I] and sharding the last dim
would interleave gate/up columns per-chip and silently corrupt outputs.

Pure JAX + lax.ragged_dot, no Pallas. bf16 only.

Notes on this port:
- The AInfer original has an optional w13-fuse (REPLACE) mode + Pallas DMA
  decode kernel + bf16-psum env flag. None of those are ported here; the
  minimal port keeps split w1/w3 and the f32-partial / bf16-cast psum path.
- TP axis name defaults to "tensor" to match sglang-jax mesh conventions
  (AInfer used "tp").
- _DECODE_THRESHOLD=64 chooses between the gather+einsum decode path and the
  sort+ragged_dot prefill path. With sglang-jax's typical
  precompile_token_paddings (>=128) the prefill path dominates; the decode
  branch is kept for callers that lower their token padding floor.
"""

import jax
import jax.numpy as jnp
from jax import lax


def _swiglu(gate_up: jax.Array, intermediate_size: int) -> jax.Array:
    """gate_up: [M, 2*I] -> silu(gate) * up : [M, I]."""
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    return (gate * jax.nn.sigmoid(gate)) * up


def _grouped_ffn_tp(
    sorted_tokens: jax.Array,   # [M, H]
    group_sizes: jax.Array,     # [n_groups] int32, sum == M
    w13: jax.Array,             # [n_groups, H, 2*I_local]
    w2: jax.Array,              # [n_groups, I_local, H]
) -> jax.Array:
    """Two ragged_dot FFN layers with SwiGLU. Returns PARTIAL [M, H]
    (caller must psum across TP axis to get full output)."""
    i_local = w2.shape[1]
    gate_up = lax.ragged_dot(sorted_tokens, w13, group_sizes)   # [M, 2*I_local]
    act = _swiglu(gate_up, i_local)                              # [M, I_local]
    return lax.ragged_dot(act, w2, group_sizes)                  # [M, H] partial


def _find_active_experts(
    flat_eid: jax.Array,        # [P] expert ids (>=0 valid, sentinel = num_experts)
    num_experts: int,
    max_active: int,
) -> tuple[jax.Array, jax.Array]:
    """Find unique active expert ids using indicator + argsort.

    Returns:
      active_ids: [max_active] int32, sorted expert ids (0-indexed). Positions
                  beyond n_active are filled with 0 (safe for gather).
      n_active:   scalar int32, number of distinct active experts.
    """
    indicator = jnp.zeros(num_experts, dtype=jnp.bool_)
    safe = jnp.where(flat_eid < num_experts, flat_eid, 0)
    valid = flat_eid < num_experts
    indicator = indicator.at[safe].max(valid)

    active_count = jnp.sum(indicator).astype(jnp.int32)
    expert_indices = jnp.arange(num_experts, dtype=jnp.int32)
    sort_key = jnp.where(indicator, 0, 1)
    order = jnp.argsort(sort_key, stable=True)
    active_ids = expert_indices[order][:max_active]
    active_ids = jnp.where(
        jnp.arange(max_active) < active_count, active_ids, 0
    )
    return active_ids, active_count


def _remap_expert_ids(
    flat_eid: jax.Array,        # [P] original expert ids
    active_ids: jax.Array,      # [max_active] sorted active expert ids
    n_active: jax.Array,        # scalar
    sentinel: int,              # original sentinel value (= num_experts)
) -> jax.Array:
    """Remap expert ids to compact 0..n_active-1 space.

    Padding/sentinel ids map to n_active (the last group, which will have
    zero-weight sentinel rows in the weight arrays).
    """
    compact = jnp.searchsorted(active_ids, flat_eid, side='left')
    is_sentinel = flat_eid >= sentinel
    compact = jnp.where(is_sentinel, n_active, compact)
    return compact.astype(jnp.int32)


def tp_moe(
    tokens: jax.Array,          # [T, H] bf16
    w1: jax.Array,              # [E, H, I_local] bf16 (gate, this device's TP slice)
    w2: jax.Array,              # [E, I_local, H] bf16 (down)
    w3: jax.Array,              # [E, H, I_local] bf16 (up, this device's TP slice)
    topk_ids: jax.Array,        # [T, top_k] int32 (expert id; <0 = padding)
    topk_weights: jax.Array,    # [T, top_k] (gate weights; 0 for padding)
    *,
    num_experts: int,
) -> jax.Array:
    """TP MoE prefill: runs grouped FFN over active experts using this device's
    weight slice. Returns PARTIAL output [T, H] — caller must psum across TP
    axis.

    w1 (gate) and w3 (up) are kept separate because TP shards them independently
    on the I dimension. They are concatenated here per-expert to form w13_local.
    """
    T, H = tokens.shape
    top_k = topk_ids.shape[1]
    P = T * top_k

    flat_eid = topk_ids.reshape(P)
    flat_w = topk_weights.reshape(P)
    tok_idx = jnp.repeat(jnp.arange(T, dtype=jnp.int32), top_k)

    safe_eid = jnp.where(flat_eid < 0, num_experts, flat_eid)

    max_active = min(top_k * T, num_experts)
    active_ids, n_active = _find_active_experts(safe_eid, num_experts, max_active)

    active_ids_sorted = jnp.sort(active_ids)

    w1_active = w1[active_ids_sorted]     # [max_active, H, I_local]
    w3_active = w3[active_ids_sorted]     # [max_active, H, I_local]
    w13_active = jnp.concatenate([w1_active, w3_active], axis=-1)  # [max_active, H, 2*I_local]
    w2_active = w2[active_ids_sorted]     # [max_active, I_local, H]

    compact_eid = _remap_expert_ids(safe_eid, active_ids_sorted, n_active, num_experts)

    order = jnp.argsort(compact_eid, stable=True)
    sorted_compact_eid = compact_eid[order]
    sorted_tok_idx = tok_idx[order]
    sorted_tokens = tokens[sorted_tok_idx]    # [P, H]

    n_groups = max_active + 1
    group_sizes = jnp.bincount(sorted_compact_eid, length=n_groups).astype(jnp.int32)

    w13_pad = jnp.concatenate(
        [w13_active, jnp.zeros((1,) + w13_active.shape[1:], w13_active.dtype)],
        axis=0,
    )
    w2_pad = jnp.concatenate(
        [w2_active, jnp.zeros((1,) + w2_active.shape[1:], w2_active.dtype)],
        axis=0,
    )

    ffn_out = _grouped_ffn_tp(sorted_tokens, group_sizes, w13_pad, w2_pad)

    sorted_w = flat_w[order].astype(ffn_out.dtype)
    weighted = ffn_out * sorted_w[:, None]
    out = jnp.zeros((T, H), dtype=jnp.float32)
    out = out.at[sorted_tok_idx].add(weighted.astype(jnp.float32))
    return out.astype(tokens.dtype)


# ---------------------------------------------------------------------------
# Decode fast path: gather + einsum (no sort/bincount/ragged_dot overhead)
# ---------------------------------------------------------------------------

def tp_moe_decode(
    tokens: jax.Array,          # [T, H] bf16 (T small, typically 1-64)
    w1: jax.Array,              # [E, H, I_local] bf16 (gate)
    w2: jax.Array,              # [E, I_local, H] bf16 (down)
    w3: jax.Array,              # [E, H, I_local] bf16 (up)
    topk_ids: jax.Array,        # [T, top_k] int32
    topk_weights: jax.Array,    # [T, top_k]
    *,
    num_experts: int,
) -> jax.Array:
    """Decode-optimized TP MoE: direct gather + einsum, zero sort/remap.

    For decode (T=1..64, top_k=8), the ragged_dot pipeline (sort -> bincount ->
    remap -> ragged_dot -> unsort) overhead dwarfs actual FFN compute. This
    path directly gathers the top_k expert weights per token and runs batched
    einsum.

    Returns PARTIAL [T, H] — caller must psum across TP axis.
    """
    T, H = tokens.shape
    top_k = topk_ids.shape[1]

    safe_ids = jnp.where(topk_ids >= 0, topk_ids, 0)    # [T, top_k]
    weights = jnp.where(topk_ids >= 0, topk_weights, 0.0)  # [T, top_k]

    w1_k = w1[safe_ids]    # [T, top_k, H, I_local]
    w3_k = w3[safe_ids]    # [T, top_k, H, I_local]
    w2_k = w2[safe_ids]    # [T, top_k, I_local, H]

    K = weights.shape[1]
    tok_exp = jnp.broadcast_to(tokens[:, None, :], (T, K, H))
    gate = jnp.einsum('tkh,tkhi->tki', tok_exp, w1_k)    # [T, K, I_local]
    up = jnp.einsum('tkh,tkhi->tki', tok_exp, w3_k)      # [T, K, I_local]
    act = (gate * jax.nn.sigmoid(gate)) * up              # SwiGLU
    down = jnp.einsum('tki,tkih->tkh', act, w2_k)        # [T, K, H]

    out = (down * weights[:, :, None]).sum(axis=1)        # [T, H]
    return out


_DECODE_THRESHOLD = 64


def tp_moe_per_device(
    toks: jax.Array,            # [T, H] (replicated across tp)
    w1_local: jax.Array,        # [E, H, I_local] gate TP slice
    w2_local: jax.Array,        # [E, I_local, H] down TP slice
    w3_local: jax.Array,        # [E, H, I_local] up TP slice
    ids: jax.Array,             # [T, top_k] GLOBAL expert ids (<0 pad)
    wts: jax.Array,             # [T, top_k]
    *,
    num_experts: int,
    tp_axis_name: str,
) -> jax.Array:
    """Per-device body for TP MoE. MUST run inside a shard_map context where
    tp_axis_name is a Manual/collective axis.

    Tokens and routing are REPLICATED across tp (each device sees all tokens).
    Weights are TP-sharded on the intermediate dimension. Each device runs the
    full grouped FFN with its weight slice, then psum combines partials.

    Decode (T <= _DECODE_THRESHOLD): gather + einsum direct path.
    Prefill (T > _DECODE_THRESHOLD): sort + ragged_dot grouped path.

    T is static under JIT trace, so this Python branch is resolved at trace
    time and a fixed graph is emitted for the chosen path.
    """
    T = toks.shape[0]
    if T <= _DECODE_THRESHOLD:
        partial = tp_moe_decode(
            toks, w1_local, w2_local, w3_local, ids, wts,
            num_experts=num_experts,
        )
    else:
        partial = tp_moe(
            toks, w1_local, w2_local, w3_local, ids, wts,
            num_experts=num_experts,
        )
    # f32 reduce then downcast — preserves the AInfer default accuracy contract
    # (cross-device sum in f32, output bf16). bf16-psum is a perf knob deferred
    # to a follow-up.
    return lax.psum(partial, tp_axis_name).astype(toks.dtype)


def fused_tp_moe_v4(
    mesh,
    tokens: jax.Array,          # [T, H] bf16
    w1: jax.Array,              # [E, H, I] bf16 (gate, TP-sharded on last dim)
    w2: jax.Array,              # [E, I, H] bf16 (down, TP-sharded on middle dim)
    w3: jax.Array,              # [E, H, I] bf16 (up, TP-sharded on last dim)
    topk_ids: jax.Array,        # [T, top_k] int32
    topk_weights: jax.Array,    # [T, top_k]
    *,
    num_experts: int,
    tp_axis_name: str = "tensor",
    data_axis_name: str = "data",
) -> jax.Array:
    """Standalone shard_map entry for TP MoE (for tests and standalone use).

    From the engine wrapper (FusedTPMoEV4.__call__), prefer calling
    tp_moe_per_device directly inside the existing shard_map to avoid nested
    mesh conflicts.
    """
    try:
        from jax import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    def _body(toks, w1l, w2l, w3l, ids, wts):
        return tp_moe_per_device(
            toks, w1l, w2l, w3l, ids, wts,
            num_experts=num_experts, tp_axis_name=tp_axis_name)

    in_specs = (
        P(data_axis_name, None),            # tokens (replicated across tp)
        P(None, None, tp_axis_name),        # w1 [E, H, I/tp]  (gate)
        P(None, tp_axis_name, None),        # w2 [E, I/tp, H]  (down)
        P(None, None, tp_axis_name),        # w3 [E, H, I/tp]  (up)
        P(data_axis_name, None),            # topk_ids
        P(data_axis_name, None),            # topk_weights
    )
    out_specs = P(data_axis_name, None)
    try:
        return shard_map(_body, mesh=mesh, in_specs=in_specs,
                         out_specs=out_specs, check_rep=False)(
            tokens, w1, w2, w3, topk_ids.astype(jnp.int32), topk_weights)
    except TypeError:
        return shard_map(_body, mesh=mesh, in_specs=in_specs,
                         out_specs=out_specs)(
            tokens, w1, w2, w3, topk_ids.astype(jnp.int32), topk_weights)
