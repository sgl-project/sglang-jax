import jax.numpy as jnp

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    VMEM_HEADROOM_BYTES,
    VMEM_SIZE,
    get_num_slices_per_block,
)


def test_kv_update_tile_leaves_vmem_headroom():
    new_kv = jnp.zeros((128, 1, 16, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((1, 64, 16, 2, 128), dtype=jnp.bfloat16)

    slices = get_num_slices_per_block(new_kv, kv_cache, page_size=64)
    scratch_bytes = slices * 64 * 32 * 128 * jnp.dtype(jnp.bfloat16).itemsize

    assert slices == 127
    assert scratch_bytes <= VMEM_SIZE - VMEM_HEADROOM_BYTES


def test_kv_update_tile_still_uses_all_input_tokens_when_small():
    new_kv = jnp.zeros((8, 1, 4, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((1, 16, 4, 2, 128), dtype=jnp.bfloat16)

    assert get_num_slices_per_block(new_kv, kv_cache, page_size=16) == 8
