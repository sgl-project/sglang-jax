import jax
import jax.numpy as jnp


def create_bench_data(
    cache_max_tokens, new_kv_len, kv_head_num, head_dim, page_size=1, dtype=jnp.bfloat16
):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)
    new_value = jax.random.normal(
        keys[1], (new_kv_len, kv_head_num, head_dim), dtype=dtype
    )
    cache = jax.random.normal(
        keys[2], (cache_max_tokens, kv_head_num, head_dim), dtype=dtype
    )
    return new_value, cache


def create_random_start_loc(cache_max_tokens, new_kv_len):
    key = jax.random.PRNGKey(42)
    cache_start_loc = jax.random.randint(key, (new_kv_len,), 0, cache_max_tokens - 1)
    return cache_start_loc
