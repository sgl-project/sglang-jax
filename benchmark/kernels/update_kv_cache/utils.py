import jax
import jax.numpy as jnp

from sgl_jax.srt.utils import cdiv


def create_bench_data(
    cache_max_tokens,
    new_kv_len,
    kv_head_num,
    head_dim,
    dtype=jnp.bfloat16,
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


def create_input_params(cache_max_tokens, new_kv_len, page_size=128):
    cache_start_loc = create_random_cache_start_loc(
        cache_max_tokens, new_kv_len, page_size=page_size
    )
    slice_lens = create_slice_lens(new_kv_len, page_size=page_size)
    new_kv_start_loc = create_new_kv_start_loc(new_kv_len, page_size=page_size)
    update_slices_num = create_update_slices_num(new_kv_len, page_size=page_size)
    return cache_start_loc, slice_lens, new_kv_start_loc, update_slices_num


def create_random_cache_start_loc(cache_max_tokens, new_kv_len, page_size=128):
    key = jax.random.PRNGKey(42)
    new_value_page_num = cdiv(new_kv_len, page_size)
    max_cache_page_num = cdiv(cache_max_tokens, page_size)
    cache_start_loc = (
        jax.random.randint(key, (new_value_page_num,), 0, max_cache_page_num - 1)
        * page_size
    )
    return cache_start_loc


def create_slice_lens(new_kv_len, page_size=128):
    slice_lens = []
    remaining = new_kv_len

    while remaining > 0:
        current_chunk = min(page_size, remaining)
        slice_lens.append(current_chunk)
        remaining -= current_chunk

    return jnp.array(slice_lens, dtype=jnp.int32)


def create_new_kv_start_loc(new_kv_len, page_size=128):
    page_num = cdiv(new_kv_len, page_size)
    return jnp.arange(0, page_num, dtype=jnp.int32) * page_size


def create_update_slices_num(new_kv_len, page_size=128):
    return cdiv(new_kv_len, page_size)
