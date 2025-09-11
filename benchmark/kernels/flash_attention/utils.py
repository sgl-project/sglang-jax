import jax
import jax.numpy as jnp

from sgl_jax.srt.utils import cdiv


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    """
    Merge k and v tensors following tpu_commons v3 exact logic.

    Args:
        k: Key tensor [total_pages, page_size, num_kv_heads, head_dim]
        v: Value tensor [total_pages, page_size, num_kv_heads, head_dim]

    Returns:
        Merged tensor [total_pages, page_size, num_kv_heads * 2, head_dim] with head interleaving
        Layout: [K0,V0,K1,V1,K2,V2...] in heads dimension (tpu_commons v3 exact)
    """
    assert (
        k.shape == v.shape
    ), f"k and v must have same shape, got {k.shape} vs {v.shape}"

    total_pages, page_size, num_kv_heads, head_dim = k.shape

    # tpu_commons v3 exact logic: concat then reshape to head interleaving
    kv_concat = jnp.concatenate(
        [k, v], axis=-1
    )  # [pages, page_size, heads, head_dim*2]
    kv_fused = kv_concat.reshape(
        total_pages, page_size, num_kv_heads * 2, head_dim
    )  # Head interleaving!

    return kv_fused


def create_kv_cache_data(
    max_kv_cache_tokens, head_num, head_dim, page_size=128, dtype=jnp.bfloat16, seed=42
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    total_num_pages = cdiv(max_kv_cache_tokens, page_size)
    k = jax.random.normal(
        keys[1], (total_num_pages, page_size, head_num, head_dim), dtype=dtype
    )
    v = jax.random.normal(
        keys[2], (total_num_pages, page_size, head_num, head_dim), dtype=dtype
    )
    return k, v


def create_fused_kv_cache_data(
    max_kv_cache_tokens, head_num, head_dim, page_size=128, dtype=jnp.bfloat16, seed=42
):
    """Create fused KV cache data following tpu_commons v3 head interleaving layout."""
    k, v = create_kv_cache_data(
        max_kv_cache_tokens, head_num, head_dim, page_size, dtype, seed
    )
    return merge_kv(k, v)


def create_q_data(total_tokens, head_num, head_dim, dtype=jnp.bfloat16, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 1)
    q = jax.random.normal(keys[0], (total_tokens, head_num, head_dim), dtype=dtype)
    return q


def create_page_indices_data(num_seqs, total_kv_tokens, seq_lens, page_size=128):
    cache_loc = jnp.arange(0, total_kv_tokens, dtype=jnp.int32)

    def get_padded_size_of_one_page(seq_len, page_size=128):
        padded_size = (seq_len + page_size - 1) // page_size * page_size
        return padded_size

    cache_start_idx = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens)]
    )

    cache_loc_list = []
    for i in range(num_seqs):
        start = cache_start_idx[i]
        end = start + seq_lens[i]
        _cache_loc = cache_loc[start:end]
        padded_size = (
            get_padded_size_of_one_page(seq_lens[i], page_size=page_size) - seq_lens[i]
        )
        padded_cache_loc = jnp.pad(_cache_loc, (0, padded_size), constant_values=0)
        cache_loc_list.append(padded_cache_loc)
    paged_cache_loc = jnp.concatenate(cache_loc_list)
    return paged_cache_loc[0::page_size] // 128, cache_loc


def create_prefill_uniform_data(
    batch_size,
    uniform_q_len,
    uniform_kv_len,
    max_kv_cache_tokens,
    head_num,
    head_dim,
    page_size=128,
    dtype=jnp.bfloat16,
    seed=42,
):
    seq_lens = jnp.array([uniform_q_len] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.array([uniform_kv_len] * batch_size, dtype=jnp.int32),
                dtype=jnp.int32,
            ),
        ]
    )
    kv_lens = seq_lens.copy()
    q = create_q_data(batch_size * uniform_q_len, head_num, head_dim, dtype, seed)
    k_cache, v_cache = create_kv_cache_data(
        max_kv_cache_tokens,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_lens, page_size=page_size
    )

    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return (
        q,
        k_cache,
        v_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
    )


def create_prefill_uniform_data_fused(
    batch_size,
    uniform_q_len,
    uniform_kv_len,
    max_kv_cache_tokens,
    head_num,
    head_dim,
    page_size=128,
    dtype=jnp.bfloat16,
    seed=42,
):
    """Create prefill data with fused KV cache following tpu_commons v3 layout."""
    seq_lens = jnp.array([uniform_q_len] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.array([uniform_kv_len] * batch_size, dtype=jnp.int32),
                dtype=jnp.int32,
            ),
        ]
    )
    kv_lens = seq_lens.copy()
    q = create_q_data(batch_size * uniform_q_len, head_num, head_dim, dtype, seed)
    kv_cache_fused = create_fused_kv_cache_data(
        max_kv_cache_tokens,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_lens, page_size=page_size
    )

    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return (
        q,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
    )


def create_decode_uniform_data(
    batch_size,
    uniform_kv_len,
    max_kv_cache_tokens,
    head_num,
    head_dim,
    page_size=1,
    dtype=jnp.bfloat16,
    seed=42,
):
    seq_len_cpu = uniform_kv_len + 1
    seq_lens = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(jnp.ones(batch_size, dtype=jnp.int32), dtype=jnp.int32),
        ]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32),
                dtype=jnp.int32,
            ),
        ]
    )
    kv_lens = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    q = create_q_data(batch_size, head_num, head_dim, dtype, seed)
    k_cache, v_cache = create_kv_cache_data(
        max_kv_cache_tokens,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_lens, page_size=page_size
    )
    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return (
        q,
        k_cache,
        v_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
    )


def create_decode_uniform_data_fused(
    batch_size,
    uniform_kv_len,
    max_kv_cache_tokens,
    head_num,
    head_dim,
    page_size=1,
    dtype=jnp.bfloat16,
    seed=42,
):
    """Create decode data with fused KV cache following tpu_commons v3 layout."""
    seq_len_cpu = uniform_kv_len + 1
    seq_lens = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(jnp.ones(batch_size, dtype=jnp.int32), dtype=jnp.int32),
        ]
    )
    cu_kv_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(
                jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32),
                dtype=jnp.int32,
            ),
        ]
    )
    kv_lens = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    q = create_q_data(batch_size, head_num, head_dim, dtype, seed)
    kv_cache_fused = create_fused_kv_cache_data(
        max_kv_cache_tokens,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_lens, page_size=page_size
    )
    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return (
        q,
        kv_cache_fused,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        cache_loc,
    )
