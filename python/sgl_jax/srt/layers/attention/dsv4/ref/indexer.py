"""Native JAX CSA indexer scoring and completed-group top-k reference."""

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.attention.dsv4.indexer import (
    INVALID_ENTRY,
    visible_entries_for_query,
)

_NEG_INF = jnp.finfo(jnp.float32).min


def csa_indexer_scores_ref(q, weights, keys, *, num_kv_heads: int | None = None):
    """Lightning-indexer scores for every query against every compressed entry.

    Args:
      q: ``[T, H, D]`` indexer query heads.
      weights: ``[T, H]`` per-head mixing weights.
      keys: ``[E, H_kv, D]`` or ``[E, D]`` compressed indexer keys. A 2-D array is
        treated as a single shared KV head.
      num_kv_heads: number of KV heads; inferred from `keys` when 3-D.

    Returns:
      ``[T, E]`` float32 scores.

    The ReLU sits *inside* the head sum, so the heads cannot be collapsed into one
    matmul; that is the DSA indexer's definition, not an accident.
    """
    q = jnp.asarray(q)
    weights = jnp.asarray(weights, jnp.float32)
    keys = jnp.asarray(keys)
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {q.shape}")
    if keys.ndim == 2:
        keys = keys[:, None, :]
    if keys.ndim != 3:
        raise ValueError(f"keys must be [E, H_kv, D] or [E, D], got {keys.shape}")
    num_heads, head_dim = q.shape[1], q.shape[2]
    if keys.shape[2] != head_dim:
        raise ValueError(f"key head_dim {keys.shape[2]} != query head_dim {head_dim}")
    kv_heads = keys.shape[1] if num_kv_heads is None else num_kv_heads
    if num_heads % kv_heads:
        raise ValueError(f"{num_heads} query heads do not fold onto {kv_heads} kv heads")
    if weights.shape != (q.shape[0], num_heads):
        raise ValueError(f"weights must be [T, H], got {weights.shape}")

    if kv_heads != keys.shape[1]:
        raise ValueError("num_kv_heads must match the key head dimension")
    group = num_heads // kv_heads
    if q.shape[0] == 0 or keys.shape[0] == 0:
        return jnp.zeros((q.shape[0], keys.shape[0]), jnp.float32)

    # As in the private DSA query-tile scorer, expose all heads to the MXU
    # together. Target an 8 MiB [Bq, H, E] FP32 temporary, with at least
    # one query per tile, instead of materializing all T queries at once.
    tile = min(32, max(1, (2 << 20) // max(1, num_heads * keys.shape[0])))
    padding = (-q.shape[0]) % tile
    tiled_q = jnp.pad(q, ((0, padding), (0, 0), (0, 0))).reshape(
        -1, tile, kv_heads, group, head_dim
    )
    tiled_w = jnp.pad(weights, ((0, padding), (0, 0))).reshape(-1, tile, kv_heads, group)

    def score_tile(inputs):
        queries, head_weights = inputs
        similarities = jnp.einsum(
            "tngd,end->tnge", queries, keys, preferred_element_type=jnp.float32
        )
        # Keep the FP32 weights out of a second MXU dot: its default
        # precision can round them to BF16 on TPU.
        return jnp.sum(jax.nn.relu(similarities) * head_weights[..., None], axis=(1, 2))

    scores = jax.lax.map(score_tile, (tiled_q, tiled_w))
    return scores.reshape(-1, keys.shape[0])[: q.shape[0]]


@functools.partial(jax.jit, static_argnames=("k", "ratio", "num_kv_heads"))
def csa_indexer_topk_ref(
    q,
    weights,
    keys,
    query_positions,
    query_request_ids,
    entry_request_ids,
    valid_token_mask,
    *,
    entry_group_ids,
    k: int,
    ratio: int,
    num_kv_heads: int | None = None,
):
    """Select up to `k` compressed entries per query.

    Args:
      q: ``[T, H, D]`` indexer queries.
      weights: ``[T, H]`` per-head weights.
      keys: ``[E, H_kv, D]`` compressed indexer keys, gathered for this step.
      query_positions: ``[T]`` absolute position of each query in its request.
      query_request_ids: ``[T]`` request each query belongs to.
      entry_request_ids: ``[E]`` request each key row belongs to. Rows that belong
        to no request must carry a value no query has.
      valid_token_mask: ``[T]`` padded query slots are False.
      entry_group_ids: `[E]` group number within each row's own request,
        distinct from the gathered row index returned by top-k.
      k: selection budget (`index_topk`).
      ratio: compression ratio of these layers (4 for CSA).

    Returns:
      ``[T, k]`` int32 entry ids, `INVALID_ENTRY` padded at the end of each row.

    Three masks decide legality, and all three are necessary:
      * the entry belongs to the same request as the query -- otherwise one
        request reads another's history;
      * the entry's group is complete at or before the query -- so a selection
        never depends on a future token;
      * the query slot is real -- padded queries select nothing.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    q = jnp.asarray(q)
    query_positions = jnp.asarray(query_positions)
    query_request_ids = jnp.asarray(query_request_ids)
    entry_request_ids = jnp.asarray(entry_request_ids)
    valid_token_mask = jnp.asarray(valid_token_mask, bool)
    num_entries = jnp.asarray(keys).shape[0]

    scores = csa_indexer_scores_ref(q, weights, keys, num_kv_heads=num_kv_heads)

    entry_ids = jnp.asarray(entry_group_ids, jnp.int32)[None, :]
    if entry_ids.shape[1] != num_entries:
        raise ValueError("entry_group_ids must describe every gathered key row")
    same_request = query_request_ids[:, None] == entry_request_ids[None, :]
    complete = (entry_ids >= 0) & (
        entry_ids < visible_entries_for_query(query_positions, ratio)[:, None]
    )
    legal = same_request & complete & valid_token_mask[:, None]

    scores = jnp.where(legal, scores, _NEG_INF)

    # Ask for min(k, E) so a history shorter than the budget does not force the
    # selector to invent entries; the row is then padded out to k.
    take = min(k, int(num_entries))
    values, indices = jax.lax.top_k(scores, take)
    legal_count = legal.sum(-1, keepdims=True)
    rank = jnp.arange(take, dtype=jnp.int32)[None, :]
    # Guard by rank *and* by score: an all-illegal row still yields `take`
    # indices from top_k, and they must not be mistaken for selections.
    keep = (rank < legal_count) & (values > _NEG_INF)
    selected = jnp.where(keep, indices.astype(jnp.int32), INVALID_ENTRY)

    if take < k:
        selected = jnp.pad(selected, ((0, 0), (0, k - take)), constant_values=INVALID_ENTRY)
    return selected
