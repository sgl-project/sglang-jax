"""Reference for QSA's sparse GQA attention.

Gathers the selected micro-blocks naively and runs dense attention over them.
Shares no index arithmetic with the Pallas kernel: the point of the reference is
to reach the same answer by a different route, so a mistake in the kernel's lane
masking or page walk does not reproduce here.

The reference takes logical ``k_cache`` / ``v_cache``. The kernel reads the
pool's packed layout, where K and V for one head sit in the two halves of a
32-bit word; building both views from one source in the parity test is what
checks that unpacking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST


def selected_tokens_ref(
    block_ids: jax.Array,
    position: int,
    *,
    compress_ratio: int,
) -> list[int]:
    """Token positions one query attends to: its blocks, then the open group.

    Blocks expand to ``compress_ratio`` consecutive tokens each. The open
    group -- the tokens after the last complete block -- never entered the
    compressed cache and so never competed in the top-k, but it is causally
    visible and has to be attended.
    """
    ratio = compress_ratio
    tokens = []
    for b in block_ids:
        b = int(b)
        if b < 0:
            continue
        tokens.extend(range(b * ratio, (b + 1) * ratio))

    tail_start = ((position + 1) // ratio) * ratio
    tokens.extend(range(tail_start, position + 1))

    # Blocks are causal by construction; filter anyway so a bad selection shows
    # up as a wrong answer here rather than as an out-of-range gather.
    return [t for t in tokens if 0 <= t <= position]


def sparse_gqa_attention_ref(
    q: jax.Array,
    block_ids: jax.Array,
    positions: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    page_table: jax.Array,
    token_to_req: jax.Array,
    *,
    compress_ratio: int,
    sm_scale: float,
) -> jax.Array:
    """Dense attention over each query's selected tokens.

    Args:
      q:            f[T, H, D]              query heads, already scaled by nothing
      block_ids:    i32[T, K]               seq-local block ids, -1 padded
      positions:    i32[T]                  each query's position in its request
      k_cache:      f[P, page_size, KVH, D] logical paged keys
      v_cache:      f[P, page_size, KVH, D] logical paged values
      page_table:   i32[S, pages_per_seq]   logical page -> physical page
      token_to_req: i32[T]                  query token -> request id
      compress_ratio: tokens per micro-block
      sm_scale:     1 / sqrt(head_dim)

    Returns:
      f32[T, H, D]
    """
    t_count, n_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    n_kv_heads = k_cache.shape[2]
    if n_heads % n_kv_heads:
        raise ValueError(
            f"num_q_heads ({n_heads}) must be a multiple of num_kv_heads ({n_kv_heads})"
        )
    heads_per_kv = n_heads // n_kv_heads

    block_ids_np = jnp.asarray(block_ids)
    out = []
    for t in range(t_count):
        req = int(token_to_req[t])
        pos = int(positions[t])
        tokens = selected_tokens_ref(block_ids_np[t], pos, compress_ratio=compress_ratio)
        if not tokens:
            out.append(jnp.zeros((n_heads, head_dim), jnp.float32))
            continue

        pages = jnp.asarray([page_table[req, tok // page_size] for tok in tokens])
        offsets = jnp.asarray([tok % page_size for tok in tokens])
        k = k_cache[pages, offsets].astype(jnp.float32)  # [N, KVH, D]
        v = v_cache[pages, offsets].astype(jnp.float32)

        # Each KV head serves heads_per_kv query heads; repeat rather than
        # reshape so the mapping is the obvious one.
        k = jnp.repeat(k, heads_per_kv, axis=1)  # [N, H, D]
        v = jnp.repeat(v, heads_per_kv, axis=1)

        # Ask for the fp32 contraction explicitly. A backend is free to run an
        # unannotated f32 matmul at bf16 -- the TPU MXU does, but only once the
        # shapes are big enough to be worth the MXU, so an unannotated reference
        # is accurate on small cases and not on large ones.
        scores = jnp.einsum("hd,nhd->hn", q[t].astype(jnp.float32), k, precision=HIGHEST) * sm_scale
        probs = jax.nn.softmax(scores, axis=-1)
        out.append(jnp.einsum("hn,nhd->hd", probs, v, precision=HIGHEST))

    return jnp.stack(out, axis=0)
