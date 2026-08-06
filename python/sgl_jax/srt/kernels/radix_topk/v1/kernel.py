# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Top-K implementation supporting multiple Vector Subcores.

High-level algorithm:
The file implements a Top-K selection algorithm targeting Vector Subcores.

Large K: Distributed MSB Radix Select (`topk_multitile`)
   - Uses a radix-based selection algorithm, proceeding digit-by-digit from the
     most significant bit (MSB) down to the least significant bit (LSB).
   - Multiple vector subcores may collaborate. In this case, they will work
     independently on all windows except for the final window, during which
     they will collaborate to build a global histogram, identify a global
     threshold, postfill remaining (tied) K elements from first subcore to last,
     and DMA each subcore's results to the correct subset of locations in HBM.
   - If subcores are not collaborating, each operates independently on
     a strided slice of the batch elements. This avoids synchronization and
     communication, so can be performance beneficial for large enough batches.
   - The algorithm processes data in sequential windows and can cooperate across
     multiple subcores.
     a) Histogram: Computes a histogram of the current digit's values across all
        active candidates.
     b) Global Sync (Optional: only when cooperating across subcores): Combines
        local histograms into a global histogram.
     c) Threshold Discovery: Accumulates the histogram from the maximum digit
        down to zero to find a `target_digit`. The `target_digit` is chosen such
        that elements with digits > `target_digit` are guaranteed to be in the
        top K.
     d) Filtering:
        - Elements with digit > `target_digit` are added to the final Top-K
          output.
        - Remaining elements with digit == `target_digit` survive (if needed, to
          meet K) as candidates for the next digit iteration, or for postfilling
          of ties.
        - Elements with digit < `target_digit` are eliminated.
   - Postfill: If K elements haven't been found after all digits are processed
     (e.g., due to duplicate values tied across all bits), then the remaining
     candidates are all tied. So we can deterministically "postfill" the
     remaining spots from the pool of surviving candidates. When multiple
     subcores are collaborating, this uses a cross-subcore prefix sum to
     break ties by index (earlier windows = lower subcores = lower indices
     are preferred).
   - Emit outputs: When tiles work independently, each produces K elements and
     can DMA the output independently of other cores. When collaborating, the
     subcores compute a prefix-sum to determine the start offset at which
     to write the elements.
"""

import dataclasses
import functools
import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax import extend as jax_extend
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.interpreters import batching, mlir

# from jax._src import state as jax_state  # deps: ignore
Ref = Any  # jax_state.AbstractRef | jax_state.TransformedRef


INVALID_VALUE = np.uint32(2**32 - 1).view(np.int32)


all_reduce_popcount = plsc.all_reduce_population_count


class _DigitGetter:
    """Helper class for getting shifted/masked digits from the keys ref."""

    def __init__(
        self,
        keys: jax.Ref,
        digit_num: jax.Array,
        digit_width: int,
        return_keys: bool,
    ):
        self.keys = keys
        self.digit_num = digit_num
        self.digit_width = digit_width
        self.return_keys = return_keys

    def __getitem__(self, slc):
        keys_chunk = self.keys[slc]
        shift_amount = jnp.uint32(self.digit_width * self.digit_num)
        digit_mask = jnp.uint32((1 << self.digit_width) - 1)
        digits = keys_chunk.view(jnp.uint32)
        if self.keys.dtype == jnp.int32:
            digits ^= jnp.uint32(1 << 31)
        elif self.keys.dtype == jnp.float32:
            bits = (digits.view(jnp.int32) >> 31).view(jnp.uint32)  # sign -> 32 bits
            bits |= jnp.uint32(1 << 31)
            digits ^= bits
        digits = (digits >> shift_amount) & digit_mask
        return (keys_chunk, digits) if self.return_keys else digits


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TopKScratch:
    """Scratch buffers for Top-K."""

    sem: Ref
    num_to_process: Ref
    num_remaining_global: Ref
    top_k_found: Ref
    top_k_found_global: Ref
    cross_subcore_cumsum: Ref
    histogram_indices: Ref
    histogram: Ref
    zeros_histogram: Ref
    global_histogram: Ref
    keys: Ref
    values: Ref
    topk_keys: Ref
    topk_values: Ref
    output_iota: Ref
    output_vmshd_keys: Ref
    output_vmshd_vals: Ref

    def digits(self, digit_num: jax.Array, digit_width: int) -> _DigitGetter:
        return _DigitGetter(self.keys, digit_num, digit_width, return_keys=False)

    def keys_and_digits(
        self,
        digit_num: jax.Array,
        digit_width: int,
    ) -> _DigitGetter:
        return _DigitGetter(self.keys, digit_num, digit_width, return_keys=True)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class DigitLoopArgs:
    digit_num: jax.Array
    num_remaining: jax.Array
    num_remaining_global: jax.Array
    num_local_outputs: jax.Array
    num_global_outputs: jax.Array
    num_postfill_candidate_elements: jax.Array


def clear_histogram(histogram: jax.Ref):
    """Clears the histogram."""
    histogram_size = np.prod(histogram.shape)
    vec_dim = plsc.get_sparse_core_info().num_lanes

    @plsc.parallel_loop(lower=0, upper=histogram_size, step=vec_dim)
    def clear_chunk(chunk_idx):
        histogram[pl.ds(chunk_idx, vec_dim)] = jnp.zeros(vec_dim, histogram.dtype)


def sum_across_subcores(values: Sequence[tuple[Ref, jax.Array]]):
    """Sums the values across subcores."""
    for value_ref, _ in values:
        value_ref[0] = jnp.zeros((), value_ref.dtype)
    plsc.subcore_barrier()
    for value_ref, value in values:
        plsc.fetch_and_add(value_ref, value, subcore_id=0)  # First add.
    plsc.subcore_barrier()
    return [plsc.fetch_and_add(v[0], jnp.int32(0), subcore_id=0) for v in values]  # Now, fetch.


def preceding_subcores_sum(mesh: plsc.VectorSubcoreMesh, ref: Ref, value: jax.Array):
    """Sums values of preceding subcores."""
    subcore_id = jax.lax.axis_index("subcore")
    if ref.shape != (mesh.num_subcores,):
        raise ValueError(f"SMEM shape {ref.shape} must be ({mesh.num_subcores},).")

    @pl.when(subcore_id == 0)
    def clear_smem():
        for i in range(mesh.num_subcores):
            ref[i] = jnp.zeros((), ref.dtype)

    plsc.subcore_barrier()
    plsc.fetch_and_add(ref.at[subcore_id], value, subcore_id=0)
    plsc.subcore_barrier()

    @pl.when(subcore_id == 0)
    def compute_cumsum():  # Subcore 0 computes prefix sum.
        prev = jnp.zeros((), ref.dtype)
        for i in range(mesh.num_subcores):
            # `prev += pl.swap(ref, i, prev)` requires `masked_swap` primitive support
            tmp = ref[i]
            ref[i] = prev
            prev += tmp

    plsc.subcore_barrier()
    return plsc.fetch_and_add(ref.at[subcore_id], jnp.int32(0), subcore_id=0)


@functools.partial(
    jax.jit,
    static_argnames=[
        "k",
        "num_seq_windows",
        "digit_width",
        "num_digits",
        "poison_scratch",
        "use_tc_tiling_on_sc",
        "debug",
    ],
)
def _sc_top_k_impl(
    keys: jax.Array,
    *maybe_values: Sequence[jax.Array],
    k: int,
    num_seq_windows: int,
    digit_width: int,
    num_digits: int,
    poison_scratch: bool = False,
    use_tc_tiling_on_sc: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Top-K primitive impl."""
    vec_dim = plsc.get_sparse_core_info().num_lanes
    histogram_size = 2**digit_width
    assert histogram_size >= vec_dim, "Histogram size must be >= vec_dim."

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=1,  # We may update this to 2 below, if there's enough work.
    )
    # TODO: bjp - tune this.
    num_cooperating_tiles = 1 if keys.shape[-1] <= 4096 else mesh.num_subcores
    batch_size = math.prod(keys.shape[:-1])
    if batch_size / (mesh.num_subcores // num_cooperating_tiles) > 1:
        mesh = dataclasses.replace(mesh, num_cores=2)
    if keys.shape[-1] % (num_seq_windows * num_cooperating_tiles) != 0:
        raise NotImplementedError(
            f"keys.shape[-1] ({keys.shape[-1]}) must be divisible by "
            "(num_seq_windows * num_cooperating_tiles) "
            f"({num_seq_windows * num_cooperating_tiles})."
        )
    elements_tile_size = keys.shape[-1] // (num_seq_windows * num_cooperating_tiles)
    if num_seq_windows <= 0:
        raise ValueError(f"num_seq_windows ({num_seq_windows}) must be > 0.")
    if elements_tile_size % vec_dim != 0 or elements_tile_size == 0:
        raise NotImplementedError(
            f"elements_tile_size ({elements_tile_size}) must be >0 and divisible "
            f"by vec_dim ({vec_dim})."
        )

    carryforward_padding = 0
    if num_seq_windows > 1:
        carryforward_padding = pl.cdiv(k, vec_dim) * vec_dim

    aligned_k = pl.cdiv(k, vec_dim) * vec_dim
    pad_histogram_size = (
        pl.cdiv(histogram_size, 128) * 128 if use_tc_tiling_on_sc else histogram_size
    )
    scratch_types = TopKScratch(
        sem=pltpu.SemaphoreType.DMA,
        num_to_process=pltpu.SMEM((1,), jnp.int32),
        num_remaining_global=pltpu.SMEM((1,), jnp.int32),
        top_k_found=pltpu.SMEM((1,), jnp.int32),
        top_k_found_global=pltpu.SMEM((1,), jnp.int32),
        cross_subcore_cumsum=pltpu.SMEM((num_cooperating_tiles,), jnp.int32),
        histogram_indices=pltpu.VMEM((pad_histogram_size,), jnp.int32),
        histogram=pltpu.VMEM((pad_histogram_size,), jnp.int32),
        zeros_histogram=pltpu.VMEM((pad_histogram_size,), jnp.int32),
        global_histogram=pltpu.VMEM_SHARED((pad_histogram_size,), jnp.int32),
        keys=pltpu.VMEM((elements_tile_size + carryforward_padding,), keys.dtype),
        values=pltpu.VMEM((elements_tile_size + carryforward_padding,), jnp.int32),
        topk_keys=pltpu.VMEM((aligned_k,), keys.dtype),
        topk_values=pltpu.VMEM((aligned_k,), jnp.int32),
        output_iota=pltpu.VMEM((aligned_k,), jnp.int32),
        output_vmshd_keys=pltpu.VMEM_SHARED((aligned_k,), keys.dtype),
        output_vmshd_vals=pltpu.VMEM_SHARED((aligned_k,), jnp.int32),
    )
    if num_cooperating_tiles == 1:
        scratch_types = dataclasses.replace(
            scratch_types,
            cross_subcore_cumsum=None,
            global_histogram=None,
            output_vmshd_keys=None,
            output_vmshd_vals=None,
        )
    # TODO: Support scratch trees directly in pl.kernel.
    scratch_types, scratch_tree = jax.tree.flatten(scratch_types)

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct(shape=keys.shape[:-1] + (aligned_k,), dtype=keys.dtype),
            jax.ShapeDtypeStruct(shape=keys.shape[:-1] + (aligned_k,), dtype=jnp.int32),
        ),
        mesh=mesh,
        scratch_types=scratch_types,
        name="topk_multitile",
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=use_tc_tiling_on_sc,
            needs_layout_passes=False,
        ),
        debug=debug,
    )
    def _kernel(keys_ref, values_ref, output_keys_ref, output_values_ref, *scratch):
        scratch = scratch_tree.unflatten(scratch)
        if poison_scratch:
            # Poison all scratch buffers.
            with jax.named_scope("poison_scratch"):

                def poison_ref(ref):
                    if ref.memory_space == pltpu.MemorySpace.VMEM:

                        @plsc.parallel_loop(0, ref.size, vec_dim)
                        def poison_chunk(i):
                            ref[pl.ds(i, vec_dim)] = jnp.full(
                                vec_dim, jnp.uint32(0xDEADDEAD).view(ref.dtype)
                            )

                    elif ref.memory_space == pltpu.MemorySpace.SMEM:

                        @plsc.parallel_loop(0, ref.size)
                        def poison_smem(i):
                            ref[i] = jnp.uint32(0xDEADDEAD).view(ref.dtype)

                jax.tree.map(poison_ref, scratch)
                # Poison global histogram.
                if scratch.global_histogram is not None:
                    pltpu.sync_copy(scratch.zeros_histogram, scratch.global_histogram)

        subcore_id = jax.lax.axis_index("subcore")
        core_id = jax.lax.axis_index("core")
        batch_elements_executing_in_parallel_per_core = mesh.num_subcores // num_cooperating_tiles
        batch_index_begin = (
            subcore_id % batch_elements_executing_in_parallel_per_core
            + core_id * batch_elements_executing_in_parallel_per_core
        )

        # The kernel is written with the assumption that we have no cooperation or
        # that all cores participate.
        assert num_cooperating_tiles in {1, mesh.num_subcores}
        coop_subcore_id = 0 if num_cooperating_tiles == 1 else subcore_id
        is_coop_leader = num_cooperating_tiles > 1 and coop_subcore_id == 0

        batch_elements_executing_in_parallel = (
            batch_elements_executing_in_parallel_per_core * mesh.num_cores
        )

        @pl.loop(batch_index_begin, batch_size, step=batch_elements_executing_in_parallel)
        def batch_loop(batch_idx):
            batch_indices = jnp.unravel_index(batch_idx, keys_ref.shape[:-1])

            @pl.when(is_coop_leader)
            def clear_zeros_histogram():
                clear_histogram(scratch.zeros_histogram)

            @plsc.parallel_loop(0, histogram_size, vec_dim)
            def fill_histogram_indices(i):
                scratch.histogram_indices[pl.ds(i, vec_dim)] = i + jnp.arange(vec_dim)

            # TODO: Double buffer scratch.keys + write_outputs
            @pl.loop(0, num_seq_windows)
            def window_loop(window_id):
                if num_seq_windows == 1:
                    window_id = 0  # For better constant propagation below
                is_last_coop_window = num_cooperating_tiles > 1 and window_id == num_seq_windows - 1

                with jax.named_scope("load input window"):
                    coop_subcore_first_window = coop_subcore_id * num_seq_windows
                    offset = elements_tile_size * (coop_subcore_first_window + window_id)
                    src_keys = keys_ref.at[*batch_indices, pl.ds(offset, elements_tile_size)]
                    # Slice out the padding, if any is present.
                    dst_keys = scratch.keys.at[:elements_tile_size]
                    keys_window_copy = pltpu.async_copy(src_keys, dst_keys, scratch.sem)
                    if values_ref is not None:
                        src_vals = values_ref.at[*batch_indices, pl.ds(offset, elements_tile_size)]
                        # As above, slice out the padding, if any is present.
                        dst_vals = scratch.values.at[:elements_tile_size]
                        val_window_copy = pltpu.async_copy(src_vals, dst_vals, scratch.sem)
                    else:
                        val_window_copy = None

                        @plsc.parallel_loop(0, elements_tile_size, vec_dim)
                        def fill_indices(i):
                            scratch.values[pl.ds(i, vec_dim)] = offset + i + jnp.arange(vec_dim)

                    @pl.when(window_id > 0)
                    def copy_previous_windows_topk_to_scratch():
                        @plsc.parallel_loop(0, k, vec_dim)
                        def copy_topk_to_scratch(i):
                            scratch.keys[pl.ds(elements_tile_size + i, vec_dim)] = (
                                scratch.topk_keys[pl.ds(i, vec_dim)]
                            )
                            scratch.values[pl.ds(elements_tile_size + i, vec_dim)] = (
                                scratch.topk_values[pl.ds(i, vec_dim)]
                            )

                        # TODO: Seems like we could enforce padding when writing?
                        if num_unaligned := k % vec_dim:
                            mask = jnp.arange(vec_dim) < num_unaligned
                            first_unaligned = (k // vec_dim) * vec_dim
                            for buf, pad_val in [
                                (scratch.keys, 0),
                                (scratch.values, INVALID_VALUE),
                            ]:
                                slc = pl.ds(elements_tile_size + first_unaligned, vec_dim)
                                buf[slc] = jnp.where(mask, buf[slc], pad_val)

                    keys_window_copy.wait()
                    del keys_window_copy
                    if val_window_copy is not None:
                        val_window_copy.wait()
                    del val_window_copy

                active_scratch = jnp.where(window_id > 0, aligned_k, 0)
                num_remaining = elements_tile_size + active_scratch
                args = DigitLoopArgs(
                    digit_num=jnp.int32(num_digits - 1),
                    num_remaining=num_remaining,
                    num_remaining_global=num_remaining * num_cooperating_tiles,
                    num_local_outputs=jnp.int32(0),
                    num_global_outputs=jnp.int32(0),
                    num_postfill_candidate_elements=jnp.int32(0),
                )

                def cond(args: DigitLoopArgs):
                    return (args.digit_num >= 0) & (args.num_global_outputs < k)

                def body(args: DigitLoopArgs):
                    with jax.named_scope("clear_histogram"):
                        # TODO: Alternatively to the loop in clear_histogram, use
                        # async DMA to zero out the histogram. Whether this is faster
                        # depends on how much other work is available to hide the latency
                        # behind. For small inputs, it can be slower.
                        clear_histogram(scratch.histogram)

                        # On last window, coop leader starts DMA to clear global histogram.
                        @pl.when(is_coop_leader & is_last_coop_window)
                        def start_clearing_global_histogram():
                            pltpu.async_copy(
                                scratch.zeros_histogram, scratch.global_histogram, scratch.sem
                            )

                    with jax.named_scope("fill_histogram"):

                        @plsc.parallel_loop(0, args.num_remaining, vec_dim)
                        def fill_histogram(i):
                            digits = scratch.digits(args.digit_num, digit_width)[pl.ds(i, vec_dim)]
                            counts, mask = plsc.scan_count(digits)
                            plsc.addupdate_scatter(scratch.histogram, [digits], counts, mask=mask)

                    @pl.when(is_last_coop_window)
                    @jax.named_scope("sync global histogram")
                    def sync_global_histogram():
                        @pl.when(is_coop_leader)
                        def wait_clearing_global_histogram():
                            pltpu.make_async_copy(
                                scratch.zeros_histogram, scratch.global_histogram, scratch.sem
                            ).wait()

                        plsc.subcore_barrier()
                        pltpu.sync_copy(
                            scratch.histogram,
                            scratch.global_histogram.at[scratch.histogram_indices],
                            add=True,
                        )
                        plsc.subcore_barrier()
                        pltpu.sync_copy(scratch.global_histogram, scratch.histogram)

                    with jax.named_scope("find_target_digit"):
                        n_minus_k = args.num_remaining_global - (k - args.num_global_outputs)
                        init = [jnp.zeros(vec_dim, jnp.int32)] * 2

                        @plsc.parallel_loop(0, histogram_size, vec_dim, carry=init)
                        def find_target_digit(i, carry):
                            [num_less, inclusive_sums] = carry
                            histogram_chunk = scratch.histogram[pl.ds(i, vec_dim)]
                            chunk_cumsum = jnp.cumsum(histogram_chunk)
                            digits_below_n_minus_k_this_chunk = plsc.all_reduce_ffs(
                                (chunk_cumsum + inclusive_sums) >= n_minus_k
                            )
                            return [
                                num_less + digits_below_n_minus_k_this_chunk,
                                inclusive_sums + chunk_cumsum[vec_dim - 1],
                            ]

                        [target_digit, _] = (
                            find_target_digit  # pylint: disable=unpacking-non-sequence
                        )
                        # target_digit is the largest digit such that all entries with digit
                        # smaller can be safely discarded. The vector is a splat.

                    # TODO: DMA could start async zeroing out the histogram here.

                    with jax.named_scope("filter_top_k_elements"):
                        # Entries with digit > target_digit are surely in the output
                        # for this window.
                        # Entries with digit == target_digit are candidates and go to
                        # the next digit step or to local/global postfill.
                        init = (jnp.int32(0), args.num_local_outputs)

                        @plsc.parallel_loop(0, args.num_remaining, vec_dim, carry=init)
                        def filter_top_k_elements(i, carry):
                            num_survivors, num_local_outputs = carry
                            keys_chunk, digits = scratch.keys_and_digits(
                                args.digit_num, digit_width
                            )[pl.ds(i, vec_dim)]
                            values_chunk = scratch.values[pl.ds(i, vec_dim)]
                            is_valid = values_chunk != INVALID_VALUE
                            is_top_k = is_valid & (digits > target_digit)
                            survive_to_next_iter = is_valid & (digits == target_digit)
                            plsc.store_compressed(
                                scratch.topk_keys.at[pl.ds(num_local_outputs, vec_dim)],
                                keys_chunk,
                                mask=is_top_k,
                            )
                            plsc.store_compressed(
                                scratch.topk_values.at[pl.ds(num_local_outputs, vec_dim)],
                                values_chunk,
                                mask=is_top_k,
                            )
                            plsc.store_compressed(
                                scratch.keys.at[pl.ds(num_survivors, vec_dim)],
                                keys_chunk,
                                mask=survive_to_next_iter,
                            )
                            plsc.store_compressed(
                                scratch.values.at[pl.ds(num_survivors, vec_dim)],
                                values_chunk,
                                mask=survive_to_next_iter,
                            )
                            return (
                                num_survivors + all_reduce_popcount(survive_to_next_iter)[0],
                                num_local_outputs + all_reduce_popcount(is_top_k)[0],
                            )

                        num_survivors, num_local_outputs = (
                            filter_top_k_elements  # pylint: disable=unpacking-non-sequence
                        )

                    # Pad the last chunk of survivors with 0, INVALID_VALUE.
                    num_unaligned = num_survivors % vec_dim

                    @pl.when(num_unaligned > 0)
                    def pad_last_chunk():
                        slc = pl.ds(num_survivors - num_unaligned, vec_dim)
                        mask = jnp.arange(vec_dim) < num_unaligned
                        for buf, pad_val in [(scratch.keys, 0), (scratch.values, INVALID_VALUE)]:
                            buf[slc] = jnp.where(mask, buf[slc], pad_val)

                    # Give the next iter chunk-aligned num_remaining.
                    pad_amount = (vec_dim - num_unaligned) % vec_dim
                    num_remaining = num_survivors + pad_amount
                    # Sync across subcores on last window (if cooperating).
                    num_remaining_global, num_global_outputs = jax.lax.cond(
                        is_last_coop_window,
                        lambda: sum_across_subcores(
                            [
                                (scratch.num_remaining_global, num_remaining),
                                (scratch.top_k_found_global, num_local_outputs),
                            ]
                        ),
                        lambda: [num_remaining, num_local_outputs],
                    )

                    return dataclasses.replace(
                        args,
                        digit_num=args.digit_num - 1,
                        num_remaining=num_remaining,
                        num_remaining_global=num_remaining_global,
                        num_local_outputs=num_local_outputs,
                        num_global_outputs=num_global_outputs,
                        num_postfill_candidate_elements=num_survivors,
                    )

                with jax.named_scope("digit_loop"):
                    digit_loop_result = jax.lax.while_loop(cond, body, args)
                num_local_outputs = digit_loop_result.num_local_outputs
                num_global_outputs = digit_loop_result.num_global_outputs
                num_postfill_candidate_elements = digit_loop_result.num_postfill_candidate_elements

                num_top_k_remaining = k - num_global_outputs
                with jax.named_scope("local_postfill"):

                    def local_postfill():
                        num_valid_postfill = jnp.minimum(
                            num_top_k_remaining, num_postfill_candidate_elements
                        )
                        num_invalid_postfill = num_top_k_remaining - num_valid_postfill
                        keys_chunk = jnp.full(vec_dim, scratch.keys[:vec_dim][0])
                        # pylint: disable=cell-var-from-loop
                        for num_to_postfill, offset, is_valid in [
                            (num_valid_postfill, num_local_outputs, True),
                            (num_invalid_postfill, num_local_outputs + num_valid_postfill, False),
                        ]:
                            num_unaligned = num_to_postfill % vec_dim
                            num_aligned = num_to_postfill - num_unaligned

                            @plsc.parallel_loop(0, num_aligned, vec_dim)
                            def pad_buffers(i):
                                scratch.topk_keys[pl.ds(offset + i, vec_dim)] = jnp.where(
                                    is_valid, keys_chunk, 0
                                )
                                scratch.topk_values[pl.ds(offset + i, vec_dim)] = jnp.where(
                                    is_valid, scratch.values[pl.ds(i, vec_dim)], INVALID_VALUE
                                )

                            @pl.when(num_unaligned > 0)
                            def pad_unaligned():
                                mask = jnp.arange(vec_dim) < num_unaligned
                                plsc.store_compressed(
                                    scratch.topk_keys.at[pl.ds(offset + num_aligned, vec_dim)],
                                    jnp.where(is_valid, keys_chunk, 0),
                                    mask=mask,
                                )
                                plsc.store_compressed(
                                    scratch.topk_values.at[pl.ds(offset + num_aligned, vec_dim)],
                                    jnp.where(
                                        is_valid,
                                        scratch.values[pl.ds(num_aligned, vec_dim)],
                                        INVALID_VALUE,
                                    ),
                                    mask=mask,
                                )

                        # pylint: enable=cell-var-from-loop
                        return num_top_k_remaining

                    num_local_outputs += jax.lax.cond(
                        (num_top_k_remaining > 0)
                        & ((window_id < num_seq_windows - 1) | (num_cooperating_tiles == 1)),
                        local_postfill,
                        lambda: 0,
                    )

                with jax.named_scope("global_postfill"):

                    def global_postfill():
                        # At this point, we are on the last window (see the cond which
                        # calls global_postfill), so all Vector Subcores are synced and agree on
                        # num_top_k_remaining.
                        #
                        # So we will cumsum the valid postfill candidate elements across
                        # Vector Subcores (each subcore only sees the sum of its predecessors).
                        # Then we'll subtract the # postfill candidates we expect to be
                        # written by preceding Vector Subcores (up to a max of num_top_k_remaining).
                        # Then we'll write the lesser of [how many are left to write], or
                        # [how many valid candidates we have].
                        # And we'll write those at an offset corresponding to the number of
                        # elements we expect preceding Vector Subcores to have written.
                        num_predecessor_postfill = preceding_subcores_sum(
                            mesh, scratch.cross_subcore_cumsum, num_postfill_candidate_elements
                        )
                        num_predecessor_postfill = jnp.minimum(
                            num_top_k_remaining, num_predecessor_postfill
                        )
                        num_to_postfill = jnp.minimum(
                            num_top_k_remaining - num_predecessor_postfill,
                            num_postfill_candidate_elements,
                        )
                        keys_chunk = jnp.full(vec_dim, scratch.keys[:vec_dim][0])
                        offset = num_local_outputs
                        num_unaligned = num_to_postfill % vec_dim
                        num_aligned = num_to_postfill - num_unaligned

                        @plsc.parallel_loop(0, num_aligned, vec_dim)
                        def copy_to_topk_buffers(i):
                            scratch.topk_keys[pl.ds(offset + i, vec_dim)] = keys_chunk
                            scratch.topk_values[pl.ds(offset + i, vec_dim)] = scratch.values[
                                pl.ds(i, vec_dim)
                            ]

                        @pl.when(num_unaligned > 0)
                        def copy_unaligned():
                            mask = jnp.arange(vec_dim) < num_unaligned
                            plsc.store_compressed(
                                scratch.topk_keys.at[pl.ds(offset + num_aligned, vec_dim)],
                                keys_chunk,
                                mask=mask,
                            )
                            plsc.store_compressed(
                                scratch.topk_values.at[pl.ds(offset + num_aligned, vec_dim)],
                                scratch.values[pl.ds(num_aligned, vec_dim)],
                                mask=mask,
                            )

                        return num_to_postfill

                    if num_cooperating_tiles > 1:
                        num_local_outputs += jax.lax.cond(
                            (num_top_k_remaining > 0) & (window_id == num_seq_windows - 1),
                            global_postfill,
                            lambda: 0,
                        )

                with jax.named_scope("write_outputs"):

                    @pl.when(window_id == num_seq_windows - 1)
                    def write_outputs():
                        if num_cooperating_tiles > 1:
                            preceding_subcore_outputs = preceding_subcores_sum(
                                mesh, scratch.cross_subcore_cumsum, num_local_outputs
                            )
                        else:
                            preceding_subcore_outputs = 0

                        @plsc.parallel_loop(0, num_local_outputs, vec_dim)
                        def fill_output_offsets(i):
                            scratch.output_iota[pl.ds(i, vec_dim)] = (
                                preceding_subcore_outputs + i + jnp.arange(vec_dim)
                            )

                        num_unaligned = num_local_outputs % vec_dim

                        @pl.when(num_unaligned > 0)
                        def write_unaligned_offsets_chunk():
                            # Can write a full chunk because output_iota is aligned_k size.
                            i = num_local_outputs - num_unaligned
                            scratch.output_iota[pl.ds(i, vec_dim)] = (
                                preceding_subcore_outputs + i + jnp.arange(vec_dim)
                            )

                        hbm_offsets = scratch.output_iota.at[pl.ds(0, num_local_outputs)]
                        if num_cooperating_tiles > 1:
                            pltpu.sync_copy(
                                (
                                    scratch.topk_keys.at[pl.ds(num_local_outputs)],
                                    scratch.topk_values.at[pl.ds(num_local_outputs)],
                                ),
                                (
                                    scratch.output_vmshd_keys.at[hbm_offsets],
                                    scratch.output_vmshd_vals.at[hbm_offsets],
                                ),
                            )
                            plsc.subcore_barrier()

                            @pl.when(is_coop_leader)
                            def _():
                                pltpu.sync_copy(
                                    (
                                        scratch.output_vmshd_keys.at[:aligned_k],
                                        scratch.output_vmshd_vals.at[:aligned_k],
                                    ),
                                    (
                                        output_keys_ref.at[*batch_indices].at[:aligned_k],
                                        output_values_ref.at[*batch_indices].at[:aligned_k],
                                    ),
                                )

                        else:
                            pltpu.sync_copy(
                                (
                                    scratch.topk_keys.at[pl.ds(num_local_outputs)],
                                    scratch.topk_values.at[pl.ds(num_local_outputs)],
                                ),
                                (
                                    output_keys_ref.at[*batch_indices].at[hbm_offsets],
                                    output_values_ref.at[*batch_indices].at[hbm_offsets],
                                ),
                            )

    return jax.tree.map(
        lambda t: t[..., :k], _kernel(keys, *(maybe_values if maybe_values else [None]))
    )


sc_top_k_p = jax_extend.core.Primitive("sc_top_k")
sc_top_k_p.multiple_results = True


@sc_top_k_p.def_impl
def _impl(*_, **__):
    raise RuntimeError("sc_top_k must be JIT'ed")


@sc_top_k_p.def_abstract_eval
def _sc_top_k_abstract_eval(keys, *maybe_values, k, **_):
    if maybe_values:
        (values,) = maybe_values
        if values.dtype != jnp.int32 or values.shape != keys.shape:
            raise ValueError(
                f"`values` has dtype {values.dtype} and shape {values.shape}, but "
                f"must have dtype int32 and shape matching `keys` {keys.shape}."
            )
        assert keys.shape == values.shape, (keys, values)
    return (
        jax.core.ShapedArray((*keys.shape[:-1], k), keys.dtype),
        jax.core.ShapedArray((*keys.shape[:-1], k), jnp.int32),
    )


def _sc_top_k_batch_rule(batched_args, batch_dims, **kwargs):
    if any(bd is None for bd in batch_dims):
        raise ValueError("sc_top_k does not support broadcasting vmap.")
    if len(set(batch_dims)) != 1:
        raise ValueError("sc_top_k does not support vmap on different k/v axes.")
    if any(bd == arg.ndim - 1 for arg, bd in zip(batched_args, batch_dims, strict=True)):
        raise ValueError("sc_top_k does not support vmap on the innermost axis.")
    outputs, indices = sc_top_k_p.bind(*batched_args, **kwargs)
    return (outputs, indices), batch_dims


batching.primitive_batchers[sc_top_k_p] = _sc_top_k_batch_rule
mlir.register_lowering(sc_top_k_p, mlir.lower_fun(_sc_top_k_impl))


@functools.partial(
    jax.jit,
    static_argnames=[
        "k",
        "use_approx_top_k",
        "num_seq_windows",
        "digit_width",
        "num_digits",
        "poison_scratch",
        "use_tc_tiling_on_sc",
        "debug",
    ],
)
def radix_topk_pallas(
    keys: jax.Array,
    values: jax.Array | None = None,
    *,
    k: int,
    use_approx_top_k: bool = False,
    num_seq_windows: int = 1,
    digit_width: int = 8,
    num_digits: int = 4,
    poison_scratch: bool = False,
    use_tc_tiling_on_sc: bool = False,
    debug: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Multi-Vector Subcore Top-K implementation.

    Args:
      keys: Input keys, shape (*batch_dims, elements_per_batch_entry).
      values: (Optional) Values to gather using the top-k indices, shape matching
        `keys`. Supports the case where we want to use an alternate set of
        indices, e.g. [shard_map(local topk)] -> [all gather] -> [global topk
        using indices derived from local topk's]. Note, 0xFFFFFFFF u32 / -1 s32 is
        reserved as a sentinel value. If passed in this argument, results are
        undefined.
      k: The number of top K elements to return.
      use_approx_top_k: Whether to pre-reduce using lax.approx_max_k.
      num_seq_windows: The number of sequential windows to process per subcore.
      digit_width: The width of the digit representation.
      num_digits: The number of digits to use.
      poison_scratch: Whether to poison the scratch buffers.
      use_tc_tiling_on_sc: Whether to use TC tiling on SC.
      debug: Whether to enable debug logging.

    Returns:
      A tuple of (topk_keys, topk_values), both of shape (*batch_dims, k).
    """
    if not use_approx_top_k and num_seq_windows != 1:
        raise ValueError("exact radix top-k currently supports only num_seq_windows=1")
    if use_tc_tiling_on_sc and digit_width == 4:
        raise ValueError("TC tiling is not supported for the 4x8 radix configuration")
    logging.info("SparseCore topk args: %s", dict(locals()))
    # TODO: We might get a speedup by using approx_max_k in the
    # values-not-None case, but would need to add support for doubly-gathered
    # indices, i.e. keys=approx_max_keys, values=values.at[approx_max_values].
    if use_approx_top_k and values is None:
        keys, values = jax.lax.approx_max_k(keys, k=k, aggregate_to_topk=False)
    return tuple(
        sc_top_k_p.bind(
            keys,
            *([] if values is None else [values]),
            k=k,
            num_seq_windows=num_seq_windows,
            digit_width=digit_width,
            num_digits=num_digits,
            poison_scratch=poison_scratch,
            use_tc_tiling_on_sc=use_tc_tiling_on_sc,
            debug=debug,
        )
    )
