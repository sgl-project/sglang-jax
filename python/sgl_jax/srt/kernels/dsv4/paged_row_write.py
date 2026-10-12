"""Run DMA for contiguous KV rows; tile read-modify-write for scattered rows.

The cache is input/output aliased, so untouched rows are not copied.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(
    dst_ref,
    loc_hbm_ref,
    valid_hbm_ref,
    values_ref,
    _,
    cache_hbm_ref,
    tiles_ref,
    sems,
    loc_ref,
    valid_ref,
    *,
    run,
    tile_rows,
    group_rows,
):
    # ``cache_hbm_ref`` is the aliased output buffer (same HBM as the input cache).
    seg = pl.program_id(0)
    dst = dst_ref[seg]

    @pl.when(dst >= 0)
    def _contiguous():
        start = pl.multiple_of(dst, tile_rows)  # the wrapper only marks tile-aligned runs
        copy = pltpu.make_async_copy(values_ref, cache_hbm_ref.at[pl.ds(start, run)], sems.at[0])
        copy.start()
        copy.wait()

    @pl.when(dst < 0)
    def _rows():
        # Only scattered segments need row metadata; keep SMEM bounded by run.
        loc_copy = pltpu.make_async_copy(loc_hbm_ref.at[seg], loc_ref, sems.at[0])
        valid_copy = pltpu.make_async_copy(valid_hbm_ref.at[seg], valid_ref, sems.at[1])
        loc_copy.start()
        valid_copy.start()
        loc_copy.wait()
        valid_copy.wait()
        # Load each distinct tile once per group and merge rows in order: last duplicate wins.
        # Groups finish their stores before the next group can reread the same tile.
        row_ids = jax.lax.broadcasted_iota(jnp.int32, (tile_rows, tiles_ref.shape[-1]), 0)

        def write_group(rows):
            valid = [valid_ref[row] != 0 for row in rows]
            locs = [loc_ref[row] for row in rows]
            # invalid rows get a tile id no valid row can share
            tiles = [jnp.where(valid[j], locs[j] // tile_rows, -1 - j) for j in range(group_rows)]
            leaders = []
            for j in range(group_rows):
                leader = jnp.int32(j)
                for i in reversed(range(j)):
                    leader = jnp.where(tiles[i] == tiles[j], jnp.int32(i), leader)
                leaders.append(leader)
            is_leader = [valid[j] & (leaders[j] == j) for j in range(group_rows)]

            def tile_start(j):
                return pl.multiple_of((locs[j] // tile_rows) * tile_rows, tile_rows)

            def load_copy(j):
                return pltpu.make_async_copy(
                    cache_hbm_ref.at[pl.ds(tile_start(j), tile_rows)], tiles_ref.at[j], sems.at[j]
                )

            def store_copy(j):
                return pltpu.make_async_copy(
                    tiles_ref.at[j], cache_hbm_ref.at[pl.ds(tile_start(j), tile_rows)], sems.at[j]
                )

            for j in range(group_rows):
                pl.when(is_leader[j])(lambda j=j: load_copy(j).start())
            for j in range(group_rows):
                pl.when(is_leader[j])(lambda j=j: load_copy(j).wait())

            def merge(j):
                target = tiles_ref.at[leaders[j]]
                new_row = jnp.broadcast_to(
                    values_ref[pl.ds(rows[j], 1), :], (tile_rows, tiles_ref.shape[-1])
                )
                target[...] = jnp.where(
                    row_ids == (locs[j] - (locs[j] // tile_rows) * tile_rows), new_row, target[...]
                )

            for j in range(group_rows):
                pl.when(valid[j])(lambda j=j: merge(j))
            for j in range(group_rows):
                pl.when(is_leader[j])(lambda j=j: store_copy(j).start())
            for j in range(group_rows):
                pl.when(is_leader[j])(lambda j=j: store_copy(j).wait())

        for group in range(run // group_rows):
            write_group(list(range(group * group_rows, (group + 1) * group_rows)))


def paged_row_write(cache, values, loc, valid, *, run: int | None = None, interpret: bool = False):
    """``cache.at[loc].set(values)`` for valid rows, by page-run DMA.

    ``cache`` [R, D] (bf16), ``values`` [T, D], ``loc`` [T] int32 destinations,
    ``valid`` [T] bool. Rows with ``valid`` False or out-of-range ``loc`` are dropped.
    """
    cache = jnp.asarray(cache)
    if cache.dtype != jnp.bfloat16:
        raise ValueError("paged_row_write requires BF16 cache")
    sublanes = pltpu.Tiling.COMPACT.shape[0]
    tile_rows = sublanes * (jnp.dtype(jnp.float32).itemsize // cache.dtype.itemsize)
    run = tile_rows if run is None else run
    group_rows = tile_rows  # Merge at most one native tile's row count per group.
    rows, dim = cache.shape
    values = jnp.asarray(values, cache.dtype)
    loc = jnp.asarray(loc, jnp.int32)
    valid = jnp.asarray(valid, bool) & (loc >= 0) & (loc < rows)
    n = values.shape[0]
    if run <= 0 or run % tile_rows:
        raise ValueError(f"run must be a positive multiple of the {tile_rows}-row tile")
    if rows % tile_rows:
        # The row fallback rewrites whole 16-row tiles, so the last tile of a cache
        # whose row count is not tile-aligned would reach past the buffer (bounds
        # checks are off). Such caches (small test pools) take the XLA scatter.
        safe = jnp.where(valid, loc, rows)
        return cache.at[safe].set(values, mode="drop")
    n_pad = -(-n // run) * run
    if n_pad != n:
        values = jnp.pad(values, ((0, n_pad - n), (0, 0)))
        loc = jnp.pad(loc, (0, n_pad - n))
        valid = jnp.pad(valid, (0, n_pad - n))
    seg = n_pad // run
    l2 = loc.reshape(seg, run)
    v2 = valid.reshape(seg, run)
    base = l2[:, 0]
    contiguous = (
        jnp.all(v2, axis=1)
        & jnp.all(l2 == base[:, None] + jnp.arange(run, dtype=jnp.int32)[None, :], axis=1)
        & (base % tile_rows == 0)
        & (base + run <= rows)
    )
    dst = jnp.where(contiguous, base, -1).astype(jnp.int32)
    # HBM-to-SMEM DMA requires a 512-byte contiguous inner slice.
    dma_elements = pltpu.Tiling.COMPACT.shape[1]  # One 32-bit lane row = 512 bytes.
    metadata_run = -(-run // dma_elements) * dma_elements
    loc = jnp.pad(l2, ((0, 0), (0, metadata_run - run)))
    valid = jnp.pad(v2.astype(jnp.int32), ((0, 0), (0, metadata_run - run)))
    return pl.pallas_call(
        functools.partial(_kernel, run=run, tile_rows=tile_rows, group_rows=group_rows),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(seg,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((run, dim), lambda i, *_: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((group_rows, tile_rows, dim), cache.dtype),
                pltpu.SemaphoreType.DMA((group_rows,)),
                pltpu.SMEM((metadata_run,), jnp.int32),
                pltpu.SMEM((metadata_run,), jnp.int32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        interpret=interpret,
        name=f"dsv4-paged-row-write-r{run}-d{dim}",
    )(dst, loc, valid, values, cache)
