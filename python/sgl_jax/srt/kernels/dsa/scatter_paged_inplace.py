"""In-place paged-cache row scatter on the native 4D cache layout (Pallas).

Twin of ``dsa_sparse_backend._scatter_paged`` for the DP-decode full-T cache
pre-write, operating DIRECTLY on the pool's 4D fused kv cache
``[pages, page_size // packing, packing, D]`` (bf16 packing = 2) so the XLA
graph never reshapes the cache. Any 3D/2D view of this array has a different
physical tiling (cache is ``T(2,128)(2,1)``; ``[rows, D]`` views tile as
``T(8,128)``), so every such reshape is materialized by XLA as a full-cache
relayout copy feeding the Pallas operand (09-20 traces: ``reshape_reshape.*
.cache_in_ref`` / ``.cache_kv_hbm_ref``, ~0.95 ms x 2 x 78 layers per step).

Per entry the kernel DMAs one ``[packing, D]`` slab (one packed sublane) into
VMEM, overwrites the rows ``[lo, hi)`` of that slab from the entry's staged
``[packing, D]`` payload with an iota select, and DMAs the slab back in place
(``input_output_aliases``) — the same slab-addressing idiom the production MLA
kernel uses on this layout.

Every slab is read-modify-written exactly ONCE: tokens that share a slab
(consecutive positions of one request, e.g. the 4 verify positions per request
with bf16 packing 2) must be merged by the caller into one entry
(``dsa_sparse_backend._scatter_paged_inplace``). Two entries on the same slab
would each fetch the old slab and the later store would revert the other's
rows (10-15 TPU: 128/256 verify rows landed).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _scatter_slabs_kernel(
    slots_ref, new_ref, cache_in_ref, cache_out_ref, new_vmem, slab_vmem, sem
):
    # slots_ref (SMEM int32 [4, T]): page, sub (slab within page), lo, hi —
    # rows [lo, hi) of the slab are overwritten from new_ref[i]; lo == hi
    # keeps the slab unchanged (padding / merged-partner entries on page 0).
    # All writes go through the aliased OUTPUT ref (also required by interpret mode).
    del cache_in_ref
    T = new_vmem.shape[0]
    stage = pltpu.make_async_copy(new_ref, new_vmem, sem)
    stage.start()
    stage.wait()
    fetch = []
    for i in range(T):
        cp = pltpu.make_async_copy(
            cache_out_ref.at[slots_ref[0, i], slots_ref[1, i]],
            slab_vmem.at[i],
            sem,
        )
        cp.start()
        fetch.append(cp)
    for cp in fetch:
        cp.wait()
    for i in range(T):
        slab = slab_vmem[i]  # [packing, D]
        row_id = jax.lax.broadcasted_iota(jnp.int32, slab.shape, 0)
        which = (row_id >= slots_ref[2, i]) & (row_id < slots_ref[3, i])
        slab_vmem[i] = jnp.where(which, new_vmem[i], slab)
    store = []
    for i in range(T):
        cp = pltpu.make_async_copy(
            slab_vmem.at[i],
            cache_out_ref.at[slots_ref[0, i], slots_ref[1, i]],
            sem,
        )
        cp.start()
        store.append(cp)
    for cp in store:
        cp.wait()


def scatter_rows_inplace(slots, new_slabs, cache4d, *, interpret=None):
    """``cache4d[page_i, sub_i, lo_i:hi_i] = new_slabs[i, lo_i:hi_i]`` with the cache aliased in place.

    ``slots`` is int32 ``[4, T]`` = (page, sub, lo, hi) per entry; ``new_slabs``
    ``[T, packing, D]`` in the cache dtype (rows outside ``[lo, hi)`` are never
    read); ``cache4d`` ``[P, S, packing, D]``. Every entry performs exactly one
    read-modify-write of its slab, so slabs ``(page, sub)`` must be distinct
    across entries that write anything (``lo < hi``); entries with ``lo == hi``
    store the slab back unchanged and may share it (reserved page 0 routing of
    padding and merged partner tokens).
    """
    assert cache4d.ndim == 4 and new_slabs.ndim == 3 and slots.shape[0] == 4, (
        cache4d.shape,
        new_slabs.shape,
        slots.shape,
    )
    T, packing, D = new_slabs.shape
    assert packing == cache4d.shape[2] and cache4d.shape[3] == D, (
        new_slabs.shape,
        cache4d.shape,
    )
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    kernel = pl.pallas_call(
        _scatter_slabs_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((T, packing, D), new_slabs.dtype),
                pltpu.VMEM((T, packing, D), cache4d.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=[jax.ShapeDtypeStruct(cache4d.shape, cache4d.dtype)],
        input_output_aliases={2: 0},
        interpret=interpret,
    )
    return kernel(slots.astype(jnp.int32), new_slabs, cache4d)[0]
