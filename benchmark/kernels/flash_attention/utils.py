import jax
import jax.numpy as jnp

from sgl_jax.srt.utils import cdiv


def create_kv_cache_data(
    max_kv_cache_tokens, head_num, head_dim, page_size=128, dtype=jnp.bfloat16, seed=42
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    total_num_pages = cdiv(max_kv_cache_tokens, page_size)
    k_cache = jax.random.normal(
        keys[1], (total_num_pages, page_size, head_num, head_dim), dtype=dtype
    )
    v_cache = jax.random.normal(
        keys[2], (total_num_pages, page_size, head_num, head_dim), dtype=dtype
    )
    return k_cache, v_cache


def create_qkv_data(
    total_tokens, q_head_num, kv_head_num, head_dim, dtype=jnp.bfloat16, seed=42
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 1)
    q = jax.random.normal(keys[0], (total_tokens, q_head_num, head_dim), dtype=dtype)
    k = jax.random.normal(keys[0], (total_tokens, kv_head_num, head_dim), dtype=dtype)
    v = jax.random.normal(keys[0], (total_tokens, kv_head_num, head_dim), dtype=dtype)
    return q, k, v


def create_page_indices_data(
    num_seqs, total_kv_tokens, seq_lens, max_context_len, page_size=128
):
    cache_loc = jnp.arange(0, total_kv_tokens, dtype=jnp.int32)

    cache_start_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens)]
    )

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
):
    if max_num_batched_tokens > 2048:
        batch_size = cdiv(max_num_batched_tokens, 2048)
        seq_lens_list = [2048] * (batch_size - 1) + [
            max_num_batched_tokens - 2048 * (batch_size - 1)
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
    k_cache, v_cache = create_kv_cache_data(
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
        k_cache,
        v_cache,
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
    random_prefix_lens = jax.random.randint(
        jax.random.PRNGKey(42), (batch_size,), 1024, 2048
    )
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
    q, k, v = create_qkv_data(
        batch_size, q_head_num, kv_head_num, head_dim, dtype, seed
    )
    k_cache, v_cache = create_kv_cache_data(
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
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    return (
        q,
        k,
        v,
        k_cache,
        v_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
        distribution,
    )


def check_vmem_oom(
    num_q_heads_per_blk,
    num_kv_heads_per_blk,
    num_kv_pages_per_blk,
    num_q_per_blk,
    page_size,
    head_dim,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
):
    if (
        estimate_vmem_utilization_in_bytes(
            num_q_heads_per_blk,
            num_kv_heads_per_blk,
            num_kv_pages_per_blk,
            num_q_per_blk,
            page_size,
            head_dim,
            q_dtype,
            kv_dtype,
        )
        > 32 * 1024 * 1024
    ):
        return False
    return True
