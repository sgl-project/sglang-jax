"""Logits processing."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

FMIX32_C1 = 0x85EBCA6B
FMIX32_C2 = 0xC2B2AE35
POS_C1 = 0x27D4EB2D
POS_C2 = 0x165667B1


def _rotl32(x: jax.Array, r: int) -> jax.Array:
    """Rotate left 32-bit unsigned integer."""
    return (x << r) | (x >> (32 - r))


def _fmix32(x: jax.Array, c1: int, c2: int) -> jax.Array:
    """Final mix for 32-bit hash."""
    x = x ^ (x >> 16)
    x = x * c1
    x = x ^ (x >> 13)
    x = x * c2
    x = x ^ (x >> 16)
    return x


def hash_tiles32_kernel(
    in_ptr: jax.Array,
    out_ptr: jax.Array,
    n_u32: int,
    seed1: int,
    seed2: int,
    fm_c1: int,
    fm_c2: int,
    pos_a: int,
    pos_b: int,
    tile: int,
    block: int,
    use_cg: bool,
):
    """Pallas kernel for hashing 32-bit tiles."""
    pid = pl.program_id(0)
    base = pid * tile

    s1 = jnp.uint32(seed1)
    s2 = jnp.uint32(seed2)
    posA = jnp.uint32(pos_a)
    posB = jnp.uint32(pos_b)

    h1 = jnp.uint32(0)
    h2 = jnp.uint32(0)

    for off in range(0, tile, block):
        idx = base + off + jnp.arange(block, dtype=jnp.int32)
        mask = idx < n_u32

        # Load data with appropriate cache modifier
        if use_cg:
            v = pl.load(in_ptr + idx, mask=mask, other=0, cache="global")
        else:
            v = pl.load(in_ptr + idx, mask=mask, other=0)
        v = v.astype(jnp.uint32)

        iu = idx.astype(jnp.uint32)
        p1 = (iu * posA + s1) ^ _rotl32(iu, 15)
        p2 = (iu * posB + s2) ^ _rotl32(iu, 13)

        k1 = _fmix32(v ^ p1, fm_c1, fm_c2)
        k2 = _fmix32(v ^ p2, fm_c1, fm_c2)

        k1 = jnp.where(mask, k1, 0)
        k2 = jnp.where(mask, k2, 0)

        h1 += jnp.sum(k1)
        h2 += jnp.sum(k2)

    nbytes = jnp.uint32(n_u32 * 4)
    h1 ^= nbytes
    h2 ^= nbytes
    h1 = _fmix32(h1, fm_c1, fm_c2)
    h2 = _fmix32(h2, fm_c1, fm_c2)

    out = (h1.astype(jnp.uint64) << 32) | h2.astype(jnp.uint64)
    pl.store(out_ptr + pid, out)


def add_tree_reduce_u64_kernel(
    in_ptr: jax.Array,
    out_ptr: jax.Array,
    n_elems: int,
    chunk: int,
):
    """Pallas kernel for tree reduction of 64-bit integers."""
    pid = pl.program_id(0)
    start = pid * chunk
    h = jnp.uint64(0)

    for i in range(chunk):
        idx = start + i
        mask = idx < n_elems
        v = pl.load(in_ptr + idx, mask=mask, other=0).astype(jnp.uint64)
        h += v

    pl.store(out_ptr + pid, h)


def _as_uint32_words(t: jax.Array) -> jax.Array:
    """Convert tensor to array of 32-bit unsigned integers."""
    tb = t.astype(jnp.uint8).ravel()
    nbytes = tb.size
    pad = (4 - (nbytes & 3)) & 3
    if pad > 0:
        tb = jnp.pad(tb, (0, pad), mode="constant")
    return tb.view(jnp.uint32)


def _final_splitmix64(x: int) -> int:
    """Final splitmix64 hash step."""
    mask = (1 << 64) - 1
    x &= mask
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & mask
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    return x


def gpu_tensor_hash(
    tensor: jax.Array,
    *,
    seed: int = 0x243F6A88,
    tile_words: int = 8192,
    block_words: int = 256,
    reduce_chunk: int = 1024,
    num_warps: int = 4,
    use_cg: bool = True,
) -> int:
    """
    Compute hash of a JAX tensor on GPU.
    Args:
        tensor: Input tensor (must be on GPU)
        seed: Initial seed for hashing
        tile_words: Number of 32-bit words per tile
        block_words: Number of 32-bit words per block
        reduce_chunk: Chunk size for tree reduction
        num_warps: Number of warps to use for Pallas kernels
        use_cg: Whether to use cache global for memory access
    Returns:
        64-bit hash value as integer
    """
    # Ensure tensor is on GPU - use devices() which returns a set
    devices = tensor.devices()
    is_gpu = any(d.platform == "gpu" for d in devices)
    if not is_gpu:
        raise ValueError("Tensor must be on GPU")

    u32 = _as_uint32_words(tensor)
    n = u32.size
    if n == 0:
        return 0

    # First pass: hash tiles
    grid1 = ((n + tile_words - 1) // tile_words,)
    partials = jnp.empty(grid1[0], dtype=jnp.uint64)

    pl.pallas_call(
        hash_tiles32_kernel,
        grid=grid1,
        out_shape=partials.shape,
        compiler_params=dict(num_warps=num_warps),
    )(
        u32,
        partials,
        n,
        seed1=seed & 0xFFFFFFFF,
        seed2=((seed * 0x9E3779B1) ^ 0xDEADBEEF) & 0xFFFFFFFF,
        fm_c1=FMIX32_C1,
        fm_c2=FMIX32_C2,
        pos_a=POS_C1,
        pos_b=POS_C2,
        tile=tile_words,
        block=block_words,
        use_cg=use_cg,
    )

    # Tree reduction
    cur = partials
    while cur.size > 1:
        n_elems = cur.size
        grid2 = ((n_elems + reduce_chunk - 1) // reduce_chunk,)
        nxt = jnp.empty(grid2[0], dtype=jnp.uint64)

        pl.pallas_call(
            add_tree_reduce_u64_kernel,
            grid=grid2,
            out_shape=nxt.shape,
            compiler_params=dict(num_warps=num_warps),
        )(
            cur,
            nxt,
            n_elems,
            chunk=reduce_chunk,
        )
        cur = nxt

    # Final hash step
    return _final_splitmix64(int(cur.item()))
