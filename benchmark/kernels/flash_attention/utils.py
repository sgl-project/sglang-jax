import jax
import jax.numpy as jnp


def create_kv_cache_data(
    total_tokens, head_num, head_dim, page_size=1, dtype=jnp.bfloat16, seed=42
):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    k = jax.random.normal(
        keys[1], (total_tokens, page_size, head_num, head_dim), dtype=dtype
    )
    v = jax.random.normal(
        keys[2], (total_tokens, page_size, head_num, head_dim), dtype=dtype
    )
    return k, v


def create_q_data(total_tokens, head_num, head_dim, dtype=jnp.bfloat16, seed=42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 1)
    q = jax.random.normal(keys[0], (total_tokens, head_num, head_dim), dtype=dtype)
    return q


PAGE_INDICES_PADDING_BUCKETS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]


def create_page_indices_data(num_seqs, total_kv_tokens, seq_lens):
    cache_loc = jnp.arange(1, total_kv_tokens + 1, dtype=jnp.int32)
    max_seq_len_in_batch = jnp.max(seq_lens).item()

    def get_padding_size(max_seq_len_in_batch):
        for bucket in PAGE_INDICES_PADDING_BUCKETS:
            if bucket >= max_seq_len_in_batch:
                return bucket
        assert False, "No bucket is greater than max_seq_len_in_batch"

    padding_size = get_padding_size(max_seq_len_in_batch)

    cache_loc_list = []
    for i in range(num_seqs):
        start = cache_loc[i]
        end = start + seq_lens[i]
        _cache_loc = cache_loc[start:end]
        padded_size = padding_size - seq_lens[i]
        padded_cache_loc = jnp.pad(_cache_loc, (0, padded_size), constant_values=0)
        cache_loc_list.append(padded_cache_loc)
    page_indices = jnp.stack(cache_loc_list)
    return page_indices, cache_loc


def create_prefill_uniform_data(
    batch_size,
    uniform_q_len,
    uniform_kv_len,
    head_num,
    head_dim,
    page_size=1,
    dtype=jnp.bfloat16,
    seed=42,
):
    seq_lens = jnp.array([uniform_q_len] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)]
    )
    kv_lens = seq_lens.copy()
    q = create_q_data(batch_size * uniform_q_len, head_num, head_dim, dtype, seed)
    k, v = create_kv_cache_data(
        batch_size * uniform_kv_len,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_lens
    )

    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return q, k, v, kv_lens, page_indices, cu_q_lens, num_seqs, seq_lens, cache_loc


def create_decode_uniform_data(
    batch_size,
    uniform_kv_len,
    head_num,
    head_dim,
    page_size=1,
    dtype=jnp.bfloat16,
    seed=42,
):
    seq_len_cpu = uniform_kv_len + 1
    seq_len = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    cu_q_lens = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(jnp.ones(batch_size, dtype=jnp.int32), dtype=jnp.int32),
        ]
    )
    kv_lens = jnp.array([seq_len_cpu] * batch_size, dtype=jnp.int32)
    q = create_q_data(batch_size, head_num, head_dim, dtype, seed)
    k, v = create_kv_cache_data(
        batch_size * seq_len_cpu,
        head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    page_indices, cache_loc = create_page_indices_data(
        batch_size, batch_size * uniform_kv_len, seq_len
    )
    num_seqs = jnp.array([batch_size], dtype=jnp.int32)
    return q, k, v, kv_lens, page_indices, cu_q_lens, num_seqs, seq_len, cache_loc
