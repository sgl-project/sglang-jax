"""Gated Delta-Rule primitives, kernels, and causal conv1d helpers for Qwen3-Next.

The recurrence math matches HuggingFace
``torch_recurrent_gated_delta_rule`` (``transformers/models/qwen3_next/
modeling_qwen3_next.py``):

    scale     = 1 / sqrt(K)
    q_t       = q_t * scale
    S_t       = S_{t-1} * exp(g_t)
    kv_mem_t  = (S_t * k_t[..., None]).sum(axis=-2)
    delta_t   = (v_t - kv_mem_t) * beta_t
    S_t       = S_t + k_t[..., None] * delta_t[..., None, :]
    o_t       = (S_t * q_t[..., None]).sum(axis=-2)

with ``S`` stored in ``[K, V]`` order per-head.

Public entry points:

Recurrence kernels — both take post-conv ``mixed_qkv`` plus the full
per-layer ``recurrent_state`` table and ``state_indices``, gather per-seq
state internally, and return per-request new state plus per-token output:

* :func:`ragged_gated_delta_rule_ref` — token-by-token ``lax.scan`` over a
  packed ragged batch (used in extend / chunked-prefill paths).
* :func:`decode_gated_delta_rule_ref` — single recurrence step parallelised
  across the batch axis (used in the decode fast path; one token per
  request, no scan).

Conv1d helpers that front-run the delta rule:

* :func:`jax_causal_conv1d_prefill` — depthwise causal conv1d over a
  ragged-batched packed sequence; gathers per-seq prior state from a full
  per-layer table.
* :func:`jax_causal_conv1d_update` — single-token causal conv1d update for
  decode; takes per-request state directly.

Internal helper :func:`_gated_delta_step` is leading-dim-agnostic and
shared between the two recurrence kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1, keepdims=True) + eps)
    return (x.astype(jnp.float32) / norm).astype(x.dtype)


def _gated_delta_step(
    state: jax.Array,  # [..., H, K, V] float32
    q_t: jax.Array,  # [..., H, K]
    k_t: jax.Array,  # [..., H, K]
    v_t: jax.Array,  # [..., H, V]
    g_t: jax.Array,  # [..., H] log-decay
    beta_t: jax.Array,  # [..., H]
) -> tuple[jax.Array, jax.Array]:
    """Single gated delta step.

    Leading-dim-agnostic — broadcasts over any prefix shape (e.g. ``[B, H]``
    for the dense scan, ``[H]`` for a per-token ragged scan). Returns
    ``(new_state [..., H, K, V], out [..., H, V])``.
    """
    decay = jnp.exp(g_t.astype(jnp.float32))[..., None, None]  # [..., H, 1, 1]
    state = state * decay
    kv_mem = jnp.einsum("...hkv,...hk->...hv", state, k_t)
    delta = (v_t - kv_mem) * beta_t[..., None]  # [..., H, V]
    # Outer product across K and V: k along K axis × delta along V axis.
    state = state + k_t[..., None] * delta[..., None, :]  # [..., H, K, V]
    out = jnp.einsum("...hkv,...hk->...hv", state, q_t)
    return state, out


# ---------------------------------------------------------------------------
# Causal conv1d (depthwise, kernel_size=K, stride=1, dilation=1)
# ---------------------------------------------------------------------------

def jax_causal_conv1d_prefill(
    x: jax.Array,  # [D, T]  packed activations
    weight: jax.Array,  # [D, kernel_size]  depthwise weight
    bias: jax.Array | None = None,  # [D] optional
    cu_seqlens: jax.Array | None = None,  # [B+1]
    conv_state: jax.Array | None = None,  # [num_blocks, D, kernel_size-1] full per-layer table
    state_indices: jax.Array | None = None,  # [B] req → slot
    activation: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d over a ragged-batched packed sequence.

    Sequences are concatenated along the token axis. Boundaries are given by
    ``cu_seqlens`` (``[0, len_0, len_0+len_1, ...]``). Each output position
    only mixes inputs from its own request — boundary lookbacks are served
    from ``conv_state`` (gathered via ``state_indices``) if provided, else
    zero.

    Returns ``(y [D, T], final_state [B, D, K-1])``. ``final_state`` holds
    the last ``K-1`` logical tokens of each request, with left-padding (zero
    or from the gathered initial state) when the request is shorter than
    ``K-1``. Scattering ``final_state`` back into the pool is the caller's
    responsibility (``RecurrentStatePool.write_layer``).
    """
    if activation not in (None, "silu"):
        raise ValueError(f"Unsupported causal conv1d activation: {activation}")

    D, T = x.shape
    K = int(weight.shape[1])
    assert cu_seqlens is not None, "cu_seqlens is required"
    B = int(cu_seqlens.shape[0]) - 1
    assert weight.shape == (D, K), f"weight {weight.shape} vs x {x.shape}"
    assert (conv_state is None) == (state_indices is None), (
        "conv_state and state_indices must be provided together"
    )

    if conv_state is not None:
        assert conv_state.shape[1:] == (D, K - 1), (
            f"conv_state {conv_state.shape} channels/kernel != ({D}, {K - 1})"
        )
        assert state_indices.shape == (B,), (
            f"state_indices {state_indices.shape} != expected ({B},)"
        )
        # Gather per-seq prior state once up front; later lookups index by
        # local seq id rather than walking the full table per token.
        state = conv_state[state_indices]  # [B, D, K-1]
    else:
        state = None

    starts = cu_seqlens[:-1]  # [B] inclusive
    ends = cu_seqlens[1:]  # [B] exclusive
    seq_lens = ends - starts  # [B]

    # Map each packed token index to its request id and intra-request position.
    t_idx = jnp.arange(T)
    seq_idx = jnp.searchsorted(cu_seqlens, t_idx, side="right") - 1  # [T]
    pos = t_idx - starts[seq_idx]  # [T]

    # Build the depthwise window. For each lookback o in [0, K-1] the source
    # logical position is p' = pos[t] - o; in-request when p' >= 0, otherwise
    # the lookback predates this batch and must come from the saved
    # `conv_state`. The state holds the K-1 most-recent pre-batch tokens
    # with newest at index K-2, so logical position p' (negative when
    # pre-batch) maps to state slot (K-1) + p'.
    o = jnp.arange(K)
    src_t = t_idx[:, None] - o[None, :]  # [T, K]
    in_seq = src_t >= starts[seq_idx][:, None]  # [T, K]
    src_t_safe = jnp.clip(src_t, 0, T - 1)
    x_gathered = x[:, src_t_safe]  # [D, T, K]

    if state is not None and K > 1:
        p_prime = pos[:, None] - o[None, :]  # [T, K]
        is_idx = jnp.clip((K - 1) + p_prime, 0, K - 2)  # [T, K]
        # Advanced indexing into [B, D, K-1] with two index arrays of shape
        # [T, K] (seq_idx broadcast and is_idx) plus a full slice on D
        # yields [T, K, D] (the slice axis trails the advanced ones per
        # numpy rules). Transpose back to [D, T, K] to match `x_gathered`.
        init_pulled = state[seq_idx[:, None], :, is_idx]  # [T, K, D]
        init_pulled = jnp.transpose(init_pulled, (2, 0, 1))  # [D, T, K]
        x_gathered = jnp.where(in_seq[None], x_gathered, init_pulled)
    elif K > 1:
        x_gathered = jnp.where(in_seq[None], x_gathered, jnp.zeros_like(x_gathered))
    # K == 1: no lookback, `src_t == t_idx` and `in_seq` is all-True; no
    # masking needed.

    # weight[d, K-1-o] is the coefficient for lookback o.
    w_flipped = weight[:, ::-1].astype(x.dtype)  # [D, K]
    y = jnp.einsum("dtk,dk->dt", x_gathered, w_flipped)
    if bias is not None:
        y = y + bias.astype(x.dtype)[:, None]
    if activation == "silu":
        y = jax.nn.silu(y)

    # Final state: the K-1 most-recent logical tokens of each request.
    # logical_idx[b, j] = (seq_lens[b] - (K-1)) + j, indexing into the per-
    # request "logical token stream" (state-padding ++ in-batch tokens).
    # When >= 0 the token came from x; when < 0 the token came from the
    # prior conv_state (or zero pad).
    if K > 1:
        j = jnp.arange(K - 1)[None, :]  # [1, K-1]
        logical_idx = seq_lens[:, None] - (K - 1) + j  # [B, K-1]
        take_from_x = logical_idx >= 0
        src_t_end_safe = jnp.clip(starts[:, None] + logical_idx, 0, T - 1)
        from_x = jnp.transpose(x[:, src_t_end_safe], (1, 0, 2))  # [B, D, K-1]
        if state is not None:
            is_slot = jnp.clip((K - 1) + logical_idx, 0, K - 2)  # [B, K-1]
            b_idx = jnp.arange(B)[:, None]
            # Same advanced-indexing-with-slice trick as the per-token
            # gather above: result is [B, K-1, D]; transpose to [B, D, K-1].
            from_init = state[b_idx, :, is_slot]  # [B, K-1, D]
            from_init = jnp.transpose(from_init, (0, 2, 1))  # [B, D, K-1]
            final_state = jnp.where(take_from_x[:, None, :], from_x, from_init)
        else:
            final_state = jnp.where(
                take_from_x[:, None, :], from_x, jnp.zeros_like(from_x)
            )
    else:
        # K == 1: the conv has no left context, so the "state" is empty.
        final_state = jnp.zeros((B, D, 0), dtype=x.dtype)

    return y, final_state


def jax_causal_conv1d_update(
    x: jax.Array,  # [B, D]  one new token per batch element
    state: jax.Array,  # [B, D, kernel_size-1]
    weight: jax.Array,  # [D, kernel_size]
    bias: jax.Array | None = None,  # [D]
    activation: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single-token causal conv1d update.

    Returns ``(y [B, D], new_state [B, D, kernel_size-1])``.
    """
    assert x.ndim == 2, f"x must be [B, D], got shape {x.shape}"
    B, D = x.shape
    kernel = int(weight.shape[1])
    assert state.shape == (B, D, kernel - 1)

    # Rolling buffer: [state(kernel-1), x_new] → window of length kernel.
    window = jnp.concatenate([state, x[..., None]], axis=-1)  # [B, D, K]
    y = jnp.einsum("bdk,dk->bd", window, weight.astype(x.dtype))
    if bias is not None:
        y = y + bias.astype(x.dtype)[None, :]
    if activation == "silu":
        y = jax.nn.silu(y)
    elif activation is None:
        pass
    else:
        raise ValueError(f"Unsupported causal conv1d activation: {activation}")
    new_state = window[..., 1:]  # drop oldest
    return y, new_state


# ---------------------------------------------------------------------------
# Recurrence kernels (extend + decode reference implementations)
# ---------------------------------------------------------------------------

def ragged_gated_delta_rule_ref(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    cu_seqlens: jax.Array,
    state_indices: jax.Array,
    has_initial_state: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> tuple[jax.Array, jax.Array]:
    """Ragged gated delta-rule forward (extend / chunked-prefill).

    Token-by-token ``jax.lax.scan`` over a packed ragged batch. Boundaries
    are given by ``cu_seqlens``; ``cu_seqlens[-1]`` is the valid-token
    count (tokens past it are padding and have their state writes gated
    off by ``valid_mask``).

    Contract: the full per-layer recurrent-state table is passed in and
    the gather to per-request state happens inside the kernel. Returns
    per-request new state (matching ``RecurrentStatePool.write_layer``);
    the caller owns the scatter back into the pool.

    Args:
        mixed_qkv: Packed ``(Q | K | V)`` tokens of shape
            ``[num_tokens, 2 * n_kq * d_k + n_v * d_v]``. Q/K are stored at
            ``n_kq`` heads; expansion to ``n_v`` heads happens inside this
            function (so callers should not pre-repeat).
        b: Pre-sigmoid beta input, ``[num_tokens, n_v]``.
        a: Pre-softplus delta-t input, ``[num_tokens, n_v]``.
        recurrent_state: Full per-layer state table,
            ``[num_blocks, n_v, d_k, d_v]``.
        A_log: ``[n_v]`` log-A parameter.
        dt_bias: ``[n_v]`` delta-t bias.
        cu_seqlens: ``[B + 1]`` int32 cumulative sequence lengths in the
            packed buffer; ``cu_seqlens[-1]`` is the number of valid
            (non-padding) tokens.
        state_indices: ``[B]`` int32 mapping request index to slot in
            ``recurrent_state``.
        has_initial_state: ``[B]`` bool. ``True`` when the slot already
            holds a valid recurrent state (chunked-prefill continuation,
            prefix-cache hit, or running decode); ``False`` for brand-new
            prefills, which must start from zero regardless of stale slot
            contents. Mirrors GPU's
            ``initial_state[~has_initial_state, ...] = 0``.
        n_kq: Number of key/query heads (per-shard).
        n_v: Number of value heads (per-shard). Must be a multiple of n_kq.
        d_k: Per-head key/query dim.
        d_v: Per-head value dim.

    Returns:
        ``(new_recurrent_state, output)`` where ``new_recurrent_state`` is
        per-request, shape ``[B, n_v, d_k, d_v]`` (fp32), and ``output``
        has shape ``[num_tokens, n_v, d_v]`` in ``mixed_qkv.dtype``.
    """
    num_tokens = mixed_qkv.shape[0]
    B = state_indices.shape[0]
    key_dim = n_kq * d_k

    # Slice + reshape + (optional) GQA expand ONCE outside the scan. Keeping
    # these out of the per-token body keeps the sharding inference on stable
    # ground: `query`/`key`/`value` arrive at the scan already shaped
    # ``[T, n_v, d_k]`` / ``[T, n_v, d_v]`` with the tensor axis pinned to
    # the head dim. Doing the reshape per step under explicit sharding lets
    # JAX place ``"tensor"`` on the wrong axis when ``n_kq == 1`` (a 1-of-N
    # split), which then breaks the outer-product step inside
    # ``_gated_delta_step``.
    query = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
    key = mixed_qkv[..., key_dim : 2 * key_dim].reshape(num_tokens, n_kq, d_k)
    value = mixed_qkv[..., 2 * key_dim :].reshape(num_tokens, n_v, d_v)
    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
        query = jnp.repeat(query, repeat_factor, axis=1)
        key = jnp.repeat(key, repeat_factor, axis=1)

    last_valid_loc = cu_seqlens[-1]
    token_idx = jnp.arange(num_tokens)
    req_indices = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    # Padding tokens (idx >= last_valid_loc) get clamped to a valid local seq
    # id; their writes are gated off via `valid_mask` so the slot they "read"
    # is irrelevant.
    req_indices = jnp.clip(req_indices, 0, B - 1)
    valid_mask = token_idx < last_valid_loc

    # Gather per-seq initial state once, then mask brand-new prefills to
    # zero. Mirrors GPU's `initial_state[~has_initial_state, ...] = 0`.
    init_state = recurrent_state[state_indices].astype(jnp.float32)
    init_state = jnp.where(
        has_initial_state[:, None, None, None],
        init_state,
        jnp.zeros_like(init_state),
    )

    A = jnp.exp(A_log.astype(jnp.float32))
    dt_bias_f32 = dt_bias.astype(jnp.float32)
    scale = d_k**-0.5

    def scan_fn(state_buf, xs):
        # state_buf: [B, n_v, d_k, d_v]
        q_h, k_h, v_h, b_t, a_t, req_idx, is_valid = xs

        state = state_buf[req_idx]  # [n_v, d_k, d_v]

        # Cast to fp32 inside the kernel to match GPU's
        # `fused_gdn_gating_kernel`.
        q_h = _l2norm(q_h.astype(jnp.float32)) * scale
        k_h = _l2norm(k_h.astype(jnp.float32))
        v_h = v_h.astype(jnp.float32)
        beta = jax.nn.sigmoid(b_t.astype(jnp.float32))
        g = -A * jax.nn.softplus(a_t.astype(jnp.float32) + dt_bias_f32)

        # _gated_delta_step uses `...` and negative axes throughout, so it
        # accepts any leading-dim shape — including no batch dim.
        new_state, out = _gated_delta_step(state, q_h, k_h, v_h, g, beta)

        new_state_buf = jnp.where(
            is_valid,
            state_buf.at[req_idx].set(new_state),
            state_buf,
        )
        return new_state_buf, out.astype(mixed_qkv.dtype)

    new_state_buf, output = jax.lax.scan(
        scan_fn,
        init_state,
        (query, key, value, b, a, req_indices, valid_mask),
    )
    return new_state_buf, output


def decode_gated_delta_rule_ref(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> tuple[jax.Array, jax.Array]:
    """Decode-only gated delta-rule (parallel single-step across the batch).

    One token per request, no cross-token dependencies — so we run a
    single ``_gated_delta_step`` parallelised across the batch axis
    instead of feeding :func:`ragged_gated_delta_rule_ref` with
    ``cu_seqlens=arange(B+1)`` (which would serialise B independent steps
    as a ``T=B`` scan). Numerically equivalent to that path; just faster.

    Decode always runs after at least one extend, so every slot already
    holds valid recurrent state — there is no ``has_initial_state`` mask
    here (the equivalent argument would always be all-``True``).

    Args:
        mixed_qkv: Post-conv tokens of shape
            ``[B, 2 * n_kq * d_k + n_v * d_v]`` (one token per request).
        b: Pre-sigmoid beta input, ``[B, n_v]``.
        a: Pre-softplus delta-t input, ``[B, n_v]``.
        recurrent_state: Full per-layer state table,
            ``[num_blocks, n_v, d_k, d_v]``.
        A_log: ``[n_v]`` log-A parameter.
        dt_bias: ``[n_v]`` delta-t bias.
        state_indices: ``[B]`` int32 mapping request index to slot.
        n_kq, n_v, d_k, d_v: head/dim configuration (see
            :func:`ragged_gated_delta_rule_ref`).

    Returns:
        ``(new_recurrent_state, output)`` where ``new_recurrent_state`` is
        per-request, shape ``[B, n_v, d_k, d_v]`` (fp32), and ``output``
        has shape ``[B, n_v, d_v]`` in ``mixed_qkv.dtype``.
    """
    B = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    q = mixed_qkv[:, :key_dim].reshape(B, n_kq, d_k)
    k = mixed_qkv[:, key_dim : 2 * key_dim].reshape(B, n_kq, d_k)
    v = mixed_qkv[:, 2 * key_dim :].reshape(B, n_v, d_v)
    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
        q = jnp.repeat(q, repeat_factor, axis=1)
        k = jnp.repeat(k, repeat_factor, axis=1)

    # Cast to fp32 inside the kernel to match GPU's
    # `fused_gdn_gating_kernel` and the ragged-ref path.
    scale = d_k**-0.5
    q_h = _l2norm(q.astype(jnp.float32)) * scale
    k_h = _l2norm(k.astype(jnp.float32))
    v_h = v.astype(jnp.float32)
    A = jnp.exp(A_log.astype(jnp.float32))
    dt_bias_f32 = dt_bias.astype(jnp.float32)
    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    g = -A * jax.nn.softplus(a.astype(jnp.float32) + dt_bias_f32)

    state = recurrent_state[state_indices].astype(jnp.float32)
    new_state, out = _gated_delta_step(state, q_h, k_h, v_h, g, beta)
    return new_state, out.astype(mixed_qkv.dtype)
