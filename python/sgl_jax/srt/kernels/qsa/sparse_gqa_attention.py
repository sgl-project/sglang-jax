"""Sparse GQA attention over QSA's selected micro-blocks (per-query form).

One program per query token. Each gathers the ``ratio`` tokens of every selected
block, plus the open group's causal tail, and runs a flash-softmax over them.

Three things measured on v6e shape this kernel (probes under
``jax_learning/pr/qwen3.8-model-support/``):

* A 4-token dynamic slice of the pool's token axis lowers. ``sparse_mla_prefill``
  needs ``read_block % 16 == 0`` because the MLA cache splits the token axis into
  ``(page_size // packing, packing)`` and a dynamic offset into a tiled axis must
  be provably divisible; the GQA cache keeps token as its own major axis, so the
  selected blocks are read at their native width with no over-fetch.
* ``sflag`` holds 2048 bytes at 4 bytes per semaphore, so at most ~384 unrolled
  DMAs fit in one program (384 passes, 448 fails). 512 blocks are therefore
  gathered in chunks of ``block_units``.
* K and V for one head share a 32-bit word in the pool, so one copy brings both
  and a shift splits them -- the idiom in ``ragged_paged_attention_v3.load_bkv``.
  An fp32 cache stores them unpacked instead, and takes the fp32 contraction.

The layout is head-minor, following ``sparse_mla_prefill``: selected keys on the
128-wide lane axis, the few heads on the sublane axis.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _kernel(
    # scalar prefetch (SMEM)
    blk_ref,  # i32[T, K]  selected block ids, -1 padded
    pos_ref,  # i32[T]     query position within its request
    req_ref,  # i32[T]     query -> request
    pt_ref,  # i32[S * pages_per_seq]  logical page -> physical page, flattened
    # inputs
    q_ref,  # [1, H, D]
    cache_hbm,  # [num_pages * page_size, head_axis, packing, D]
    # output
    o_ref,  # [1, H, D]
    # scratch
    kv_scratch,  # [CBR, head_axis, packing, D]
    sem,  # DMA semaphores (G,)
    *,
    sm_scale: float,
    precision,
    ratio: int,
    K: int,
    G: int,
    CBR: int,
    packing: int,
    page_size: int,
    pages_per_seq: int,
):
    t = pl.program_id(0)
    heads, head_dim = q_ref.shape[1], q_ref.shape[2]
    q = q_ref[0]  # [H, D]
    qpos = pos_ref[t]
    pt_base = req_ref[t] * pages_per_seq
    lane = jnp.arange(CBR, dtype=jnp.int32)

    def _src(unit):
        """HBM rows of one micro-block: ``ratio`` tokens starting at unit*ratio.

        ``ratio`` divides ``page_size``, so a block never straddles a page and one
        page-table lookup places it. The physical page id bounds the address, so
        a padding lane clamped to unit 0 still reads somewhere legal.
        """
        first = unit * ratio
        page = first // page_size
        offset = first - page * page_size
        phys = pt_ref[jnp.minimum(pt_base + page, pt_ref.shape[0] - 1)]
        return cache_hbm.at[pl.ds(phys * page_size + offset, ratio)]

    def _copy(unit, slot):
        return pltpu.make_async_copy(
            _src(unit), kv_scratch.at[pl.ds(slot * ratio, ratio)], sem.at[slot]
        )

    def _split_kv():
        """K and V out of the gathered rows.

        bf16 packs them into the two halves of one 32-bit word, so a widening
        bitcast (which halves the sublane axis) plus a shift separates them
        without a strided sublane read. fp32 stores them as separate head-axis
        entries, where a plain slice is already contiguous.
        """
        if packing == 1:
            return kv_scratch[:, 0, 0, :], kv_scratch[:, 1, 0, :]
        dtype = kv_scratch.dtype
        packed = kv_scratch.bitcast(jnp.uint32).reshape(CBR, head_dim)[...]
        k = pltpu.bitcast(packed.astype(jnp.uint16), dtype)
        v = pltpu.bitcast((packed >> 16).astype(jnp.uint16), dtype)
        return k, v

    def _attend(bias, carry):
        """One flash-softmax step over whatever is currently in kv_scratch."""
        m_i, l_i, acc = carry
        k, v = _split_kv()
        s = (
            jax.lax.dot_general(
                q,
                k,
                (((1,), (1,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )
        s = s + bias[None, :]

        m_new = jnp.maximum(m_i, jnp.max(s, axis=1))
        # An all-masked step leaves m_new at -inf; correcting by 0 instead keeps
        # exp() finite and makes the step contribute nothing.
        corr = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
        p = jnp.exp(s - corr[:, None])
        alpha = jnp.where(jnp.isneginf(m_new), 1.0, jnp.exp(m_i - m_new))
        l_new = l_i * alpha + jnp.sum(p, axis=1)
        acc_new = acc * alpha[:, None] + jax.lax.dot_general(
            p.astype(k.dtype),
            v,
            (((1,), (0,)), ((), ())),
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return m_new, l_new, acc_new

    # Lanes beyond G*ratio are never written; zero them so a stale row can't
    # produce a NaN score before the -inf bias lands on it.
    if G * ratio < CBR:
        pad = CBR - G * ratio
        kv_scratch[pl.ds(G * ratio, pad)] = jnp.zeros(
            (pad,) + kv_scratch.shape[1:], kv_scratch.dtype
        )

    def chunk_body(c, carry):
        first = c * G

        def unit_at(g):
            # Clamp the column (the last chunk may run past K) and the block id
            # (top-k pads with -1); both lanes are masked out below.
            return jnp.maximum(blk_ref[t, jnp.minimum(first + g, K - 1)], 0)

        for g in range(G):
            _copy(unit_at(g), g).start()

        # The lane validity mask is built arithmetically from an iota while the
        # copies are in flight: a lane concat of ratio-wide pieces is not
        # Mosaic-lowerable, and a dynamic gather on a vector is expensive. The
        # accumulators are int32 rather than bool because Mosaic cannot
        # materialise an i1 vector from a broadcast-scalar select.
        u_vec = jnp.zeros((CBR,), jnp.int32)
        row_vec = jnp.zeros((CBR,), jnp.int32)
        in_range = jnp.zeros((CBR,), jnp.int32)
        for g in range(G):
            lo = g * ratio
            sel = (lane >= lo) & (lane < lo + ratio)
            u_g = blk_ref[t, jnp.minimum(first + g, K - 1)]
            inr = ((first + g) < K).astype(jnp.int32)
            u_vec = jnp.where(sel, u_g, u_vec)
            row_vec = jnp.where(sel, lane - lo, row_vec)
            in_range = jnp.where(sel, inr, in_range)

        for g in range(G):
            _copy(unit_at(g), g).wait()

        kpos = u_vec * ratio + row_vec
        valid = (u_vec >= 0) & (in_range > 0) & (kpos <= qpos)
        return _attend(jnp.where(valid, 0.0, float("-inf")), carry)

    carry = (
        jnp.full((heads,), float("-inf"), jnp.float32),
        jnp.zeros((heads,), jnp.float32),
        jnp.zeros((heads, head_dim), jnp.float32),
    )
    carry = jax.lax.fori_loop(0, (K + G - 1) // G, chunk_body, carry)

    # The open group: tokens after the last complete block. They never entered
    # the compressed cache, so they never competed in the top-k, but they are
    # causally visible. One more unit, with the future lanes masked off; the
    # step is inert when the group happens to be closed.
    tail_unit = (qpos + 1) // ratio
    _copy(tail_unit, 0).start()
    tail_valid = (lane < ratio) & (tail_unit * ratio + lane <= qpos)
    _copy(tail_unit, 0).wait()
    m_i, l_i, acc = _attend(jnp.where(tail_valid, 0.0, float("-inf")), carry)

    o_ref[0] = (acc / jnp.where(l_i == 0.0, 1.0, l_i)[:, None]).astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("sm_scale", "ratio", "block_units", "interpret"),
)
def sparse_gqa_attention(
    q: jax.Array,  # [T, H, D]
    block_ids: jax.Array,  # i32[T, K]
    positions: jax.Array,  # i32[T]
    token_to_req: jax.Array,  # i32[T]
    page_table: jax.Array,  # i32[S, pages_per_seq]
    cache: jax.Array,  # [num_pages, page_size, head_axis, packing, D]
    *,
    sm_scale: float,
    ratio: int = 4,
    block_units: int = 128,
    interpret: bool = False,
) -> jax.Array:
    """Attend over the micro-blocks QSA selected, one program per query token.

    ``block_units`` is how many blocks one chunk gathers before attending. Each
    costs a DMA semaphore and the sflag space holds about 384, so it is a
    ceiling, not a knob with an open top: 512 blocks at 128 per chunk is four
    chunks with room left for the query and output semaphores.

    Returns f32[T, H, D].
    """
    t_count, n_heads, head_dim = q.shape
    k_blocks = block_ids.shape[1]
    num_pages, page_size, head_axis, packing, cache_dim = cache.shape

    if page_size % ratio:
        raise ValueError(
            f"ratio ({ratio}) must divide page_size ({page_size}) so a block "
            "never straddles a page"
        )
    if cache_dim != head_dim:
        raise ValueError(f"cache head_dim {cache_dim} != query head_dim {head_dim}")
    if head_axis * packing != 2:
        raise ValueError(
            "this kernel handles one KV head per device (the TP=2 layout), so the "
            f"cache's two head axes must multiply to 2 (K and V), got "
            f"{head_axis}*{packing}"
        )
    if packing == 2 and head_dim % 128:
        raise ValueError(f"packed cache needs head_dim % 128 == 0, got {head_dim}")
    if block_units < 1:
        raise ValueError("block_units must be positive")
    if block_ids.shape[0] != t_count or positions.shape[0] != t_count:
        raise ValueError("block_ids, positions and q must agree on the token count")

    g = min(block_units, k_blocks)
    cbr = _align(g * ratio, 128)
    # Mosaic's default contraction is a single bf16 pass on the MXU, which is what
    # a bf16 cache wants. An fp32 cache was paid for to keep the precision, so ask
    # for the fp32 contraction rather than silently rounding it away.
    precision = jax.lax.Precision.HIGHEST if cache.dtype == jnp.float32 else None

    kernel = pl.pallas_call(
        functools.partial(
            _kernel,
            sm_scale=sm_scale,
            precision=precision,
            ratio=ratio,
            K=k_blocks,
            G=g,
            CBR=cbr,
            packing=packing,
            page_size=page_size,
            pages_per_seq=page_table.shape[1],
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(t_count,),
            in_specs=[
                pl.BlockSpec((1, n_heads, head_dim), lambda i, *_: (i, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((1, n_heads, head_dim), lambda i, *_: (i, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((cbr, head_axis, packing, head_dim), cache.dtype),
                pltpu.SemaphoreType.DMA((g,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((t_count, n_heads, head_dim), jnp.float32),
        interpret=interpret,
    )
    return kernel(
        block_ids,
        positions,
        token_to_req,
        page_table.reshape(-1),
        q,
        cache.reshape(num_pages * page_size, head_axis, packing, head_dim),
    )
