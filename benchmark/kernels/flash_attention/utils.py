import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention.util import cdiv, get_dtype_packing


def create_kv_cache_data(
    max_kv_cache_tokens, head_num, head_dim, page_size=128, dtype=jnp.bfloat16, seed=42
):
    packing = get_dtype_packing(dtype)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    total_num_pages = cdiv(max_kv_cache_tokens, page_size)
    kv_cache = jax.random.normal(
        keys[1],
        (total_num_pages, page_size, head_num * 2 // packing, packing, head_dim),
        dtype=dtype,
    )
    return kv_cache


def create_qkv_data(total_tokens, q_head_num, kv_head_num, head_dim, dtype=jnp.bfloat16, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    q = jax.random.normal(keys[0], (total_tokens, q_head_num, head_dim), dtype=dtype)
    k = jax.random.normal(keys[1], (total_tokens, kv_head_num, head_dim), dtype=dtype)
    v = jax.random.normal(keys[2], (total_tokens, kv_head_num, head_dim), dtype=dtype)
    return q, k, v


def create_page_indices_data(num_seqs, total_kv_tokens, seq_lens, max_context_len, page_size=128):
    cache_loc = jnp.arange(0, total_kv_tokens, dtype=jnp.int32)

    cache_start_idx = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens)])

    cache_loc_list = []
    for i in range(num_seqs):
        start = cache_start_idx[i]
        end = start + seq_lens[i]
        _cache_loc = cache_loc[start:end]
        padded_cache_loc = jnp.pad(
            _cache_loc, (0, max_context_len - seq_lens[i]), constant_values=0
        )
        cache_loc_list.append(padded_cache_loc)
    paged_cache_loc = jnp.concatenate(cache_loc_list)
    return paged_cache_loc[0::page_size] // page_size, cache_loc


def create_prefill_uniform_data(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size=128,
    dtype=jnp.bfloat16,
    seed=42,
    chunk_prefill_size: int = 4096,
):
    if max_num_batched_tokens > chunk_prefill_size:
        batch_size = cdiv(max_num_batched_tokens, chunk_prefill_size)
        seq_lens_list = [chunk_prefill_size] * (batch_size - 1) + [
            max_num_batched_tokens - chunk_prefill_size * (batch_size - 1)
        ]
    else:
        batch_size = 1
        seq_lens_list = [max_num_batched_tokens]

    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(seq_lens, dtype=jnp.int32),
        ]
    )
    kv_lens = seq_lens.copy()
    q, k, v = create_qkv_data(
        max_num_batched_tokens, q_head_num, kv_head_num, head_dim, dtype, seed
    )
    kv_cache = create_kv_cache_data(
        max_kv_cache_tokens,
        kv_head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size,
        max_num_batched_tokens,
        seq_lens,
        max_context_len,
        page_size=page_size,
    )

    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    return (
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
        distribution,
    )


def create_decode_uniform_data(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size=128,
    dtype=jnp.bfloat16,
    seed=42,
):
    batch_size = max_num_batched_tokens
    # hackly set prefix len to 2048-4096 for decode one seq in random
    random_prefix_lens = jax.random.randint(jax.random.PRNGKey(42), (batch_size,), 1024, 2048)
    seq_lens = random_prefix_lens + 1
    cu_q_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(jnp.ones(batch_size, dtype=jnp.int32), dtype=jnp.int32),
        ]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(seq_lens),
        ]
    )
    q, k, v = create_qkv_data(batch_size, q_head_num, kv_head_num, head_dim, dtype, seed)
    kv_cache = create_kv_cache_data(
        max_kv_cache_tokens,
        kv_head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    total_kv_lens = seq_lens.sum().item()
    page_indices, cache_loc = create_page_indices_data(
        batch_size, total_kv_lens, seq_lens, max_context_len, page_size=page_size
    )
    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    distribution = jnp.array([batch_size, batch_size, batch_size], dtype=jnp.int32)
    return (
        q,
        k,
        v,
        kv_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
        distribution,
    )


def create_target_verify_uniform_data(
    *,
    batch_size: int,
    draft_token_num: int,
    prefix_len: int,
    page_indices_capacity: int,
    max_kv_cache_tokens: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    page_size: int = 128,
    dtype=jnp.bfloat16,
    seed: int = 42,
):
    """Create the padded MIXED workload used by batched target verification.

    Unlike ``create_prefill_uniform_data``, this models many short query
    segments attending to long existing prefixes. ``cu_kv_lens`` follows the
    production target-verify path and advances by page-aligned KV lengths,
    while ``kv_lens`` retains the actual (prefix + draft) length.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if draft_token_num <= 0:
        raise ValueError(f"draft_token_num must be positive, got {draft_token_num}")
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")

    total_q_tokens = batch_size * draft_token_num
    kv_len = prefix_len + draft_token_num
    aligned_kv_len = cdiv(kv_len, page_size) * page_size
    pages_per_seq = cdiv(kv_len, page_size)
    required_page_indices = batch_size * pages_per_seq
    if page_indices_capacity < required_page_indices:
        raise ValueError(
            "page_indices_capacity is too small: "
            f"required={required_page_indices}, got={page_indices_capacity}"
        )

    q, k, v = create_qkv_data(
        total_q_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
        dtype,
        seed,
    )
    packing = get_dtype_packing(dtype)
    total_num_pages = cdiv(max_kv_cache_tokens, page_size)
    if total_num_pages < required_page_indices:
        raise ValueError(
            "max_kv_cache_tokens is too small: "
            f"required_pages={required_page_indices}, got_pages={total_num_pages}"
        )
    # Cache contents do not affect kernel timing. Avoid allocating random
    # generation intermediates for the multi-GB production-shaped cache.
    kv_cache = jnp.zeros(
        (total_num_pages, page_size, kv_head_num * 2 // packing, packing, head_dim),
        dtype=dtype,
    )

    kv_lens = jnp.full((batch_size,), kv_len, dtype=jnp.int32)
    cu_q_lens = jnp.arange(
        0,
        total_q_tokens + 1,
        draft_token_num,
        dtype=jnp.int32,
    )
    cu_kv_lens = jnp.arange(
        0,
        batch_size * aligned_kv_len + 1,
        aligned_kv_len,
        dtype=jnp.int32,
    )

    page_indices = jnp.pad(
        jnp.arange(required_page_indices, dtype=jnp.int32),
        (0, page_indices_capacity - required_page_indices),
        constant_values=0,
    )
    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    seq_lens = kv_lens
    cache_loc = jnp.arange(total_q_tokens, dtype=jnp.int32)
    # TARGET_VERIFY metadata starts as [0, N, N]. RPA remaps the middle
    # boundary to zero when chunk_prefill_size=None, so all rows execute in
    # the MIXED pallas_call.
    distribution = jnp.array([0, batch_size, batch_size], dtype=jnp.int32)
    return (
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
        distribution,
    )
