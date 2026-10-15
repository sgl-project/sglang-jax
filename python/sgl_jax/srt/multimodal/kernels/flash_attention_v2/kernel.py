# Copyright 2026 The SGLang-JAX Authors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.ref import Ref

from .config import (
    DEFAULT_GUARD_LIMIT,
    DEFAULT_GUARD_MIN,
    DEFAULT_KV_LOOKAHEAD,
    FRAGMENT_ROWS,
    GUARD_GROUPS,
    GUARD_ROWS,
    HEAD_DIM,
    KV_FRAGMENTS,
    KV_PAIR_TILES,
    KV_ROWS,
    LOG2_E,
    MAX_IO_BUFFERS,
    NATIVE_CHANNELS,
    NATIVE_HEAD_ALIGNMENT,
    PROBABILITY_SLOTS,
    STATE_COLS,
    AttentionOptions,
    BufferPlan,
    IOPlan,
    ProbabilityCarry,
    RecoveryCarry,
    ScalarIndex,
)
from .io import BlockIO, _read_native_head
from .pipeline import FragmentLayout, Pipeline


@dataclass
class AttentionOps:
    """Numerical operations and reference ownership for one resident Q."""

    options: AttentionOptions
    q_ref: Ref
    k_ref: Ref
    v_ref: Ref
    anchor_ref: Ref  # FP32 log2-domain max from the first KV tile, held fixed.
    denom_ref: Ref  # FP32 softmax denominator, broadcast across 128 columns.
    acc_ref: Ref  # FP32 weighted-value accumulator; PV inputs remain BF16.
    output_ref: Ref
    scale_refs: tuple[Ref, Ref, Ref]
    sink_ref: Ref
    q_start: ScalarIndex
    q_length: ScalarIndex
    kv_length: ScalarIndex
    head_start: ScalarIndex
    kv_head_start: ScalarIndex
    gqa_ratio: int
    query_heads: int
    kv_plan: BufferPlan = BufferPlan()  # noqa: RUF009 -- Immutable plan.
    native_layout: bool = False
    reload_kv: Callable[[ScalarIndex], None] = field(init=False, repr=False)
    mask_ref: Ref = field(init=False, repr=False)

    def kv_head(self, head_index: ScalarIndex) -> ScalarIndex:
        head = self.head_start + head_index
        if self.query_heads % self.q_ref.shape[0]:
            head = jnp.minimum(head, self.query_heads - 1)
        return head // self.gqa_ratio - self.kv_head_start

    def load_rhs(
        self, kv: ScalarIndex, head_index: ScalarIndex, mxu: int, rhs_slot: int | None
    ) -> None:
        ref = self.k_ref if mxu == 0 else self.v_ref
        if self.native_layout:
            if (
                self.gqa_ratio == 1
                and self.query_heads % self.q_ref.shape[0] == 0
                and self.q_ref.shape[0] % NATIVE_HEAD_ALIGNMENT == 0
            ):
                head = head_index
            else:
                head = self.kv_head_start % NATIVE_HEAD_ALIGNMENT + self.kv_head(head_index)
            rhs = _read_native_head(ref.at[self.kv_plan.slot(kv)], head, 0, KV_ROWS)
            pltpu.matmul_push_rhs(rhs, rhs_slot, mxu, transpose=(mxu == 0))
            return
        pltpu.matmul_push_rhs(
            ref[self.kv_plan.slot(kv), self.kv_head(head_index), :, :],
            cast(int, rhs_slot),
            mxu,
            transpose=(mxu == 0),
        )

    def issue_qk(
        self,
        head_index: ScalarIndex,
        row: ScalarIndex,
        mrb_address: int,
        rhs_slot: int | None,
    ) -> None:
        if self.native_layout:
            lhs = self.q_ref[head_index, pl.ds(row, FRAGMENT_ROWS), :]
            if lhs.shape[-1] == NATIVE_CHANNELS:
                lhs = jnp.concatenate((lhs, jnp.zeros_like(lhs)), axis=1)
            pltpu.matmul_acc_lhs(mrb_address, lhs, 0, load_staged_rhs=rhs_slot)
            return
        pltpu.matmul_acc_lhs(
            mrb_address,
            self.q_ref[head_index, pl.ds(row, FRAGMENT_ROWS), :],
            0,
            load_staged_rhs=rhs_slot,
        )

    def issue_pv(self, p: jax.Array, mrb_address: int, rhs_slot: int | None) -> None:
        pltpu.matmul_acc_lhs(mrb_address, p, 1, load_staged_rhs=rhs_slot)

    def logits(
        self,
        scores: jax.Array,
        kv: ScalarIndex,
        head_index: ScalarIndex,
        row: ScalarIndex,
        masked: bool = True,
        base2: bool = False,
    ) -> jax.Array:
        opt = self.options
        global_head = self.head_start + head_index
        scale = opt.sm_scale
        if not opt.identity_qk_scales:
            scale *= self.scale_refs[0][global_head] * self.scale_refs[1][global_head]
        if base2 and not opt.logits_soft_cap:
            scale *= LOG2_E
        scores = scores * scale
        if opt.logits_soft_cap:
            scores = opt.logits_soft_cap * jnp.tanh(scores / opt.logits_soft_cap)
            if base2:
                scores *= LOG2_E
        if not masked:
            return scores
        if (
            opt.causal
            and opt.mask_value == -math.inf
            and opt.window_left < 0
            and opt.window_right < 0
        ):
            # Cache diagonal masks, including bottom-right offsets.

            first = jnp.maximum((self.q_start + self.kv_length - self.q_length) // KV_ROWS, 0)
            slot = jnp.where(kv < first, 0, kv - first + 1)
            visible = self.mask_ref[slot, pl.ds(row, FRAGMENT_ROWS), :] != 0
            return jnp.where(visible, scores, -jnp.inf)
        qr = self.q_start + row + lax.broadcasted_iota(jnp.int32, (FRAGMENT_ROWS, KV_ROWS), 0)
        kr = lax.broadcasted_iota(jnp.int32, (FRAGMENT_ROWS, KV_ROWS), 1)
        center = qr + self.kv_length - self.q_length - kv * KV_ROWS
        key_limit = self.kv_length - kv * KV_ROWS
        if opt.mask_value == -math.inf:
            upper = key_limit - 1
            if opt.causal:
                upper = jnp.minimum(upper, center)
            if opt.window_right >= 0:
                upper = jnp.minimum(upper, center + opt.window_right)
            visible = kr <= upper
            if opt.window_left >= 0:
                visible &= kr >= center - opt.window_left
            return jnp.where(visible, scores, -jnp.inf)
        visible = jnp.ones((FRAGMENT_ROWS, KV_ROWS), dtype=jnp.bool_)
        if opt.causal:
            visible &= kr <= center
        if opt.window_left >= 0:
            visible &= kr >= center - opt.window_left
        if opt.window_right >= 0:
            visible &= kr <= center + opt.window_right
        scores = jnp.where(visible, scores, opt.mask_value * (LOG2_E if base2 else 1.0))
        # Finite masks never relax packed-sequence boundaries.

        valid = kr < key_limit
        return jnp.where(valid, scores, -jnp.inf)

    def compute_probabilities(
        self,
        kv: ScalarIndex,
        head_index: ScalarIndex,
        row: ScalarIndex,
        mrb_address: int,
        initialize: bool,
        masked: bool = True,
    ) -> jax.Array:
        scores = self.logits(
            pltpu.matmul_pop(mrb_address, (FRAGMENT_ROWS, KV_ROWS), jnp.float32, 0),
            kv,
            head_index,
            row,
            masked,
            base2=True,
        )
        fragment_slice = (head_index, pl.ds(row, FRAGMENT_ROWS), slice(None))
        if initialize:
            anchor = jnp.max(scores, axis=1, keepdims=True)
            if self.options.has_sink:
                anchor = jnp.maximum(anchor, self.sink_ref[self.head_start + head_index] * LOG2_E)
            anchor = jnp.where(anchor == -jnp.inf, 0.0, anchor)
            anchor = jnp.broadcast_to(anchor, (FRAGMENT_ROWS, STATE_COLS))
            self.anchor_ref[fragment_slice] = anchor
        else:
            anchor = self.anchor_ref[fragment_slice]
        # Keep broadcast state to avoid scalar-column gathers.

        p = jnp.exp2(scores - jnp.concatenate((anchor, anchor), axis=1))
        denom = jnp.sum(p, axis=1, keepdims=True)
        denom = jnp.broadcast_to(denom, (FRAGMENT_ROWS, STATE_COLS))
        if initialize and self.options.has_sink:
            denom += jnp.exp2(self.sink_ref[self.head_start + head_index] * LOG2_E - anchor)
        elif not initialize:
            denom += self.denom_ref[fragment_slice]
        self.denom_ref[fragment_slice] = denom
        return p.astype(jnp.bfloat16)

    def accumulate_output(
        self,
        head_index: ScalarIndex,
        row: ScalarIndex,
        mrb_address: int,
        initialize: bool,
    ) -> None:
        value = pltpu.matmul_pop(mrb_address, (FRAGMENT_ROWS, HEAD_DIM), jnp.float32, 1)
        if not self.options.identity_v_scale:
            value *= self.scale_refs[2][self.head_start + head_index]
        fragment_slice = (head_index, pl.ds(row, FRAGMENT_ROWS), slice(None))
        if not initialize:
            value += self.acc_ref[fragment_slice]
        self.acc_ref[fragment_slice] = value

    def recover_fragment(
        self, head_index: ScalarIndex, row: ScalarIndex, tiles: ScalarIndex
    ) -> None:
        """Stable online max/rescale replay; called only after normal drain."""
        sink = self.sink_ref[self.head_start + head_index] if self.options.has_sink else -jnp.inf
        initial_max = jnp.full((FRAGMENT_ROWS, 1), sink, jnp.float32)
        initial_denom = jnp.full(
            (FRAGMENT_ROWS, 1), jnp.where(sink == -jnp.inf, 0.0, 1.0), jnp.float32
        )
        initial_acc = jnp.zeros((FRAGMENT_ROWS, HEAD_DIM), jnp.float32)

        def step(kv: ScalarIndex, carry: RecoveryCarry) -> RecoveryCarry:
            running_max, denom, accumulator = carry
            self.reload_kv(kv)
            self.load_rhs(kv, head_index, 0, 0)
            self.issue_qk(head_index, row, 0, 0)
            scores = self.logits(
                pltpu.matmul_pop(0, (FRAGMENT_ROWS, KV_ROWS), jnp.float32, 0),
                kv,
                head_index,
                row,
            )
            updated_max = jnp.maximum(running_max, jnp.max(scores, axis=1, keepdims=True))
            safe_max = jnp.where(updated_max == -jnp.inf, 0.0, updated_max)
            alpha = jnp.where(running_max == -jnp.inf, 0.0, jnp.exp(running_max - safe_max))
            p = jnp.exp(scores - safe_max)
            self.load_rhs(kv, head_index, 1, 0)
            self.issue_pv(p.astype(jnp.bfloat16), 0, 0)
            value = pltpu.matmul_pop(0, (FRAGMENT_ROWS, HEAD_DIM), jnp.float32, 1)
            if not self.options.identity_v_scale:
                value *= self.scale_refs[2][self.head_start + head_index]
            return (
                updated_max,
                denom * alpha + jnp.sum(p, axis=1, keepdims=True),
                accumulator * alpha + value,
            )

        _, denom, accumulator = lax.fori_loop(
            0, tiles, step, (initial_max, initial_denom, initial_acc)
        )
        result = jnp.where(denom > 0, accumulator / jnp.where(denom > 0, denom, 1.0), 0.0)
        self.output_ref[head_index, pl.ds(row, FRAGMENT_ROWS), :] = result[
            :, : self.output_ref.shape[-1]
        ].astype(jnp.bfloat16)

    def _unsafe(
        self,
        denom: jax.Array,
        value: jax.Array,
        head_index: ScalarIndex,
        row: ScalarIndex,
        all_nonempty: bool = False,
    ) -> jax.Array:
        """Share the unchanged numerical guard with selective recovery."""
        if all_nonempty:
            if self.options.guard_limit == DEFAULT_GUARD_LIMIT:
                # Positive FP32 bit ordering permits an unsigned interval test.

                bits = lax.bitcast_convert_type(denom, jnp.uint32)
                lower = lax.bitcast_convert_type(jnp.float32(DEFAULT_GUARD_MIN), jnp.uint32)
                upper = lax.bitcast_convert_type(jnp.float32(DEFAULT_GUARD_LIMIT), jnp.uint32)
                unsafe = ((bits - lower) > (upper - lower)) | ~jnp.isfinite(value)
            else:
                unsafe = (
                    ~jnp.isfinite(denom)
                    | (denom > self.options.guard_limit)
                    | (denom < 1 / self.options.guard_limit)
                    | ~jnp.isfinite(value)
                )
        else:
            qr = self.q_start + row + lax.broadcasted_iota(jnp.int32, (GUARD_ROWS, HEAD_DIM), 0)
            center = qr + self.kv_length - self.q_length
            lo = jnp.zeros_like(center)
            hi = jnp.full_like(center, self.kv_length - 1)
            if self.options.mask_value == -math.inf:
                if self.options.causal:
                    hi = jnp.minimum(hi, center)
                if self.options.window_left >= 0:
                    lo = jnp.maximum(lo, center - self.options.window_left)
                if self.options.window_right >= 0:
                    hi = jnp.minimum(hi, center + self.options.window_right)
            nonempty = lo <= hi
            if self.options.has_sink:
                nonempty |= jnp.isfinite(self.sink_ref[self.head_start + head_index])
            valid = qr < self.q_length
            unsafe = (
                ~jnp.isfinite(denom)
                | (denom > self.options.guard_limit)
                | ((denom < 1 / self.options.guard_limit) & nonempty)
            )
            unsafe = (unsafe | ~jnp.isfinite(value)) & valid
        return unsafe

    def head_needs_recovery(self, head_index: ScalarIndex) -> jax.Array:
        """Inspect original FP32 state only after the aggregate guard failed."""

        def chunk(i, flags):
            row = i * GUARD_ROWS
            denom = self.denom_ref[head_index, pl.ds(row, GUARD_ROWS), :]
            denom = jnp.concatenate((denom, denom), axis=1)
            value = self.acc_ref[head_index, pl.ds(row, GUARD_ROWS), :]
            unsafe = self._unsafe(denom, value, head_index, row)
            return jnp.maximum(
                flags,
                jnp.max(
                    unsafe.astype(jnp.float32).reshape(GUARD_GROUPS, FRAGMENT_ROWS, HEAD_DIM),
                    axis=0,
                ),
            )

        flags = lax.fori_loop(
            0,
            self.q_ref.shape[1] // GUARD_ROWS,
            chunk,
            jnp.zeros((FRAGMENT_ROWS, HEAD_DIM), jnp.float32),
        )
        return jnp.max(flags) > 0

    def finalize(self, force: Ref, all_nonempty: bool = False) -> jax.Array:
        """Return an aggregate guard; leave valid normal results in output."""

        def head(head_index: ScalarIndex, guard_flags: jax.Array) -> jax.Array:
            # Q128 chunks bound vector liveness.
            def chunk(i: ScalarIndex, guard_flags: jax.Array) -> jax.Array:
                row = i * GUARD_ROWS
                denom = self.denom_ref[head_index, pl.ds(row, GUARD_ROWS), :]
                denom = jnp.concatenate((denom, denom), axis=1)
                value = self.acc_ref[head_index, pl.ds(row, GUARD_ROWS), :]
                unsafe = self._unsafe(denom, value, head_index, row, all_nonempty)
                # Fold vector tiles before the scalar reduction.

                guard_flags = jnp.maximum(
                    guard_flags,
                    jnp.max(
                        unsafe.astype(jnp.float32).reshape(GUARD_GROUPS, FRAGMENT_ROWS, HEAD_DIM),
                        axis=0,
                    ),
                )
                guard_flags = jnp.maximum(
                    guard_flags,
                    (force[self.head_start + head_index] != 0).astype(jnp.float32),
                )
                if all_nonempty:
                    result = value / denom
                else:
                    result = jnp.where(denom > 0, value / jnp.where(denom > 0, denom, 1.0), 0.0)
                self.output_ref[head_index, pl.ds(row, GUARD_ROWS), :] = result[
                    :, : self.output_ref.shape[-1]
                ].astype(jnp.bfloat16)
                return guard_flags

            for i in range(self.q_ref.shape[1] // GUARD_ROWS):
                guard_flags = chunk(i, guard_flags)
            return guard_flags

        guard_flags = lax.fori_loop(
            0,
            self.q_ref.shape[0],
            head,
            jnp.zeros((FRAGMENT_ROWS, HEAD_DIM), jnp.float32),
        )
        return jnp.max(guard_flags) > 0


def _kernel(
    q_metadata_ref: Ref,
    kv_metadata_ref: Ref,
    q_scale_ref: Ref,
    k_scale_ref: Ref,
    v_scale_ref: Ref,
    sink_ref: Ref,
    force_ref: Ref,
    q_hbm: Ref,
    k_hbm: Ref,
    v_hbm: Ref,
    output_hbm: Ref,
    q_ref: Ref,
    k_ref: Ref,
    v_ref: Ref,
    anchor_ref: Ref,
    denom_ref: Ref,
    acc_ref: Ref,
    output_ref: Ref,
    mask_ref: Ref,
    q_sem: Ref,
    k_sem: Ref,
    v_sem: Ref,
    output_sem: Ref,
    *,
    options: AttentionOptions,
    gqa_ratio: int,
    kv_group_heads: int,
    pipeline: Pipeline,
    query_heads: int,
    io_plan: IOPlan,
    kv_lookahead: int = DEFAULT_KV_LOOKAHEAD,
    io_factory=BlockIO,
    native_layout: bool = False,
    sequence_offset: ScalarIndex = 0,
) -> None:
    group_heads, q_rows = q_ref.shape[-3:-1]
    sequence_index = pl.program_id(0)
    head_start = pl.program_id(1) * group_heads
    kv_head_start = head_start // gqa_ratio

    def read_boundary(ref, row, column):
        return ref[sequence_offset + row] if native_layout else ref[row, column]

    # Packing guarantees fragment-aligned non-native DMA.
    q_base = pl.multiple_of(
        read_boundary(q_metadata_ref, sequence_index, 0), 1 if native_layout else q_rows
    )
    kv_base = pl.multiple_of(
        read_boundary(kv_metadata_ref, sequence_index, 0),
        1 if native_layout else KV_ROWS,
    )
    q_length = read_boundary(q_metadata_ref, sequence_index + 1, 1) - read_boundary(
        q_metadata_ref, sequence_index, 1
    )
    kv_length = read_boundary(kv_metadata_ref, sequence_index + 1, 1) - read_boundary(
        kv_metadata_ref, sequence_index, 1
    )

    q_blocks = (q_length + (q_rows - 1)) // q_rows

    q_io = io_factory(io_plan.q, ((q_hbm, q_ref, q_sem),), q_base, head_start, group_heads)
    output_io = io_factory(
        io_plan.output,
        ((output_hbm, output_ref, output_sem),),
        q_base,
        head_start,
        group_heads,
    )
    q_transfer = q_io.transfer

    def output_transfer(q_index: ScalarIndex, start: bool) -> None:
        alignment = 1 if native_layout else FRAGMENT_ROWS
        valid_rows = pl.multiple_of(
            (jnp.minimum(q_rows, q_length - q_index * q_rows) + (alignment - 1))
            // alignment
            * alignment,
            alignment,
        )
        output_io.transfer(q_index, start, rows=valid_rows)

    @pl.when(q_blocks > 0)
    def load_first_q() -> None:
        q_transfer(0, True)

    def run_q(q_index: ScalarIndex, loaded_kv_tiles: ScalarIndex) -> jax.Array:
        q_start = q_index * q_rows
        output_buffer = output_io.buffer(q_index)

        def release_output_slot() -> None:
            @pl.when(q_index >= output_ref.shape[0])
            def release() -> None:
                output_transfer(q_index - output_ref.shape[0], False)

        if output_ref.shape[0] == MAX_IO_BUFFERS:
            release_output_slot()

        def prefetch_q() -> None:
            @pl.when(q_index + 1 < q_blocks)
            def prefetch():
                q_transfer(q_index + 1, True)

        kv_io = io_factory(
            io_plan.kv,
            ((k_hbm, k_ref, k_sem), (v_hbm, v_ref, v_sem)),
            kv_base,
            kv_head_start,
            kv_group_heads,
            loaded_kv_tiles,
        )

        # Finite masks retain masked keys in normalization.

        causal_prune = options.causal and options.mask_value == -math.inf
        key_end = kv_length
        if causal_prune:
            q_end = jnp.minimum(q_start + q_rows, q_length)
            key_end = jnp.clip(q_end + kv_length - q_length, 0, kv_length)
        tiles = (key_end + (KV_ROWS - 1)) // KV_ROWS
        attention_ops = AttentionOps(
            options=options,
            q_ref=q_io.buffer(q_index),
            k_ref=k_ref,
            v_ref=v_ref,
            anchor_ref=anchor_ref,
            denom_ref=denom_ref,
            acc_ref=acc_ref,
            output_ref=output_buffer,
            scale_refs=(q_scale_ref, k_scale_ref, v_scale_ref),
            sink_ref=sink_ref,
            q_start=q_start,
            q_length=q_length,
            kv_length=kv_length,
            head_start=head_start,
            kv_head_start=kv_head_start,
            gqa_ratio=gqa_ratio,
            query_heads=query_heads,
            kv_plan=io_plan.kv,
            native_layout=native_layout,
        )
        attention_ops.mask_ref = mask_ref

        def prepare_masks():
            if (
                options.causal
                and options.mask_value == -math.inf
                and options.window_left < 0
                and options.window_right < 0
            ):
                first = jnp.maximum((q_start + kv_length - q_length) // KV_ROWS, 0)
                mask_ref[0, :, :] = jnp.ones((q_rows, KV_ROWS), jnp.int32)
                qr = q_start + lax.broadcasted_iota(jnp.int32, (q_rows, KV_ROWS), 0)
                kr = lax.broadcasted_iota(jnp.int32, (q_rows, KV_ROWS), 1)
                center = qr + kv_length - q_length
                for offset in range(q_rows // KV_ROWS + 1):
                    kv = first + offset
                    upper = jnp.minimum(center - kv * KV_ROWS, kv_length - kv * KV_ROWS - 1)
                    mask_ref[offset + 1, :, :] = (kr <= upper).astype(jnp.int32)

        init_layout = FragmentLayout(
            fragments=q_rows // FRAGMENT_ROWS, heads=group_heads, initialize=True
        )
        full_layout = FragmentLayout(fragments=q_rows // FRAGMENT_ROWS, heads=group_heads)
        tail_half_layout = FragmentLayout(
            fragments=KV_FRAGMENTS, row_base=q_rows - KV_ROWS, heads=group_heads
        )

        pending_probs = tuple(
            jnp.zeros((FRAGMENT_ROWS, KV_ROWS), jnp.bfloat16) for _ in range(PROBABILITY_SLOTS)
        )

        def nonempty(_: None) -> jax.Array:
            kv_io.transfer(0, True)

            for ahead in range(1, kv_lookahead + 1):

                @pl.when(tiles > ahead)
                def prefetch_initial() -> None:
                    kv_io.transfer(ahead, True)  # noqa: B023

            # Equal-length full Q blocks share the diagonal mask table.

            reusable_mask = (q_length == kv_length) & (q_length % q_rows == 0)

            @pl.when((q_index == 0) | ~reusable_mask)
            def fill_masks():
                prepare_masks()

            q_transfer(q_index, False)
            kv_io.transfer(0, False)
            state = pipeline.emit_tile(attention_ops, init_layout, None, 0, 0, pending_probs)

            def append(
                kv: ScalarIndex,
                state: ProbabilityCarry,
                *,
                previous: FragmentLayout,
                current: FragmentLayout,
            ) -> ProbabilityCarry:

                @pl.when(kv + kv_lookahead < tiles)
                def prefetch_next():
                    kv_io.transfer(kv + kv_lookahead, True)

                kv_io.transfer(kv, False)

                return pipeline.emit_tile(attention_ops, current, previous, kv, kv - 1, state)

            # The final causal tile touches only the last KV_ROWS query rows.

            tail_short = jnp.bool_(False)
            if causal_prune:
                tail_short = (tiles - 1) * KV_ROWS >= q_start + (
                    q_rows - KV_ROWS
                ) + kv_length - q_length
            full_tiles = tiles - tail_short.astype(jnp.int32)

            def full_tail(state: ProbabilityCarry) -> ProbabilityCarry:
                state = append(1, state, previous=init_layout, current=full_layout)

                def kv_loop(kv: ScalarIndex, state: ProbabilityCarry) -> ProbabilityCarry:
                    return append(kv, state, previous=full_layout, current=full_layout)

                if causal_prune and options.window_left < 0 and options.window_right < 0:
                    interior_end = jnp.maximum(
                        KV_PAIR_TILES,
                        jnp.minimum(
                            full_tiles,
                            jnp.minimum(
                                kv_length // KV_ROWS,
                                (q_start + kv_length - q_length + 1) // KV_ROWS,
                            ),
                        ),
                    )
                    pairs = (interior_end - KV_PAIR_TILES) // KV_PAIR_TILES

                    def pair(i: ScalarIndex, x: ProbabilityCarry) -> ProbabilityCarry:
                        kv = KV_PAIR_TILES + KV_PAIR_TILES * i

                        @pl.when(kv + kv_lookahead < tiles)
                        def prefetch_even():
                            kv_io.transfer(kv + kv_lookahead, True)

                        @pl.when(kv + (kv_lookahead + 1) < tiles)
                        def prefetch_odd():
                            kv_io.transfer(kv + (kv_lookahead + 1), True)

                        kv_io.transfer(kv, False)
                        kv_io.transfer(kv + 1, False)
                        interior_layout = FragmentLayout(
                            fragments=q_rows // FRAGMENT_ROWS,
                            heads=group_heads,
                            masked=False,
                        )
                        for block in range(KV_PAIR_TILES):
                            for p in range(full_layout.periods):
                                old_layout = (
                                    full_layout if block == 0 and p == 0 else interior_layout
                                )
                                x = pipeline.emit(
                                    attention_ops=attention_ops,
                                    current=interior_layout,
                                    previous=old_layout,
                                    kv=kv + block,
                                    previous_kv=kv + block - (p == 0),
                                    period=p,
                                    previous_period=(p - 1) % full_layout.periods,
                                    pending_probs=x,
                                )
                        return x

                    state = lax.fori_loop(0, pairs, pair, state)
                    leftover = KV_PAIR_TILES + KV_PAIR_TILES * pairs
                    # Share one remainder body for interior and causal boundary tiles.

                    return lax.fori_loop(leftover, full_tiles, kv_loop, state)
                return lax.fori_loop(KV_PAIR_TILES, full_tiles, kv_loop, state)

            state = lax.cond(full_tiles > 1, full_tail, lambda x: x, state)

            def end_short(state: ProbabilityCarry) -> ProbabilityCarry:

                state = lax.cond(
                    full_tiles == 1,
                    lambda x: append(full_tiles, x, previous=init_layout, current=tail_half_layout),
                    lambda x: append(full_tiles, x, previous=full_layout, current=tail_half_layout),
                    state,
                )
                # QK reads are complete; prefetch Q while draining PV.

                prefetch_q()
                return pipeline.drain(attention_ops, tail_half_layout, tiles, state)

            def end_full(state: ProbabilityCarry) -> ProbabilityCarry:
                prefetch_q()
                return lax.cond(
                    tiles == 1,
                    lambda x: pipeline.drain(attention_ops, init_layout, tiles, x),
                    lambda x: pipeline.drain(attention_ops, full_layout, tiles, x),
                    state,
                )

            lax.cond(tail_short & (tiles > 1), end_short, end_full, state)
            if output_ref.shape[0] == 1:
                release_output_slot()
            if (
                options.causal
                and options.window_left < 0
                and options.window_right < 0
                and options.mask_value == -math.inf
            ):
                all_nonempty = (
                    (q_start + q_rows <= q_length)
                    & (kv_length > 0)
                    & (q_start + kv_length - q_length >= 0)
                )
                return lax.cond(
                    all_nonempty,
                    lambda _: attention_ops.finalize(force=force_ref, all_nonempty=True),
                    lambda _: attention_ops.finalize(force=force_ref),
                    operand=None,
                )
            return attention_ops.finalize(force=force_ref)

        def empty(_: None) -> jax.Array:
            q_transfer(q_index, False)
            if output_ref.shape[0] == 1:
                release_output_slot()
            output_buffer[...] = jnp.zeros(output_buffer.shape, jnp.bfloat16)
            prefetch_q()
            return jnp.bool_(False)

        needs_recovery = lax.cond(tiles > 0, nonempty, empty, operand=None)

        @pl.when(needs_recovery)
        def recover() -> None:
            # Reclaim prefetched Q before replay, then rebuild its prefetch.

            @pl.when(q_index + 1 < q_blocks)
            def restore_current_q() -> None:
                q_transfer(q_index + 1, False)
                q_transfer(q_index, True)
                q_transfer(q_index, False)

            attention_ops.reload_kv = kv_io.reload

            def head(head_index: ScalarIndex, _: None) -> None:
                def frag(f: ScalarIndex, _: None) -> None:
                    attention_ops.recover_fragment(
                        head_index=head_index, row=f * FRAGMENT_ROWS, tiles=tiles
                    )

                unsafe = lax.cond(
                    force_ref[head_start + head_index] != 0,
                    lambda _: jnp.bool_(True),
                    lambda _: attention_ops.head_needs_recovery(head_index),
                    operand=None,
                )

                @pl.when(unsafe)
                def failed_head():
                    lax.fori_loop(0, q_rows // FRAGMENT_ROWS, frag, None)

            lax.fori_loop(0, group_heads, head, None)

            prefetch_q()

        output_transfer(q_index, True)
        return jnp.maximum(loaded_kv_tiles, tiles)

    lax.fori_loop(0, q_blocks, run_q, jnp.int32(0))
    for lag in range(output_ref.shape[0], 0, -1):

        @pl.when(q_blocks >= lag)
        def drain_output() -> None:
            output_transfer(q_blocks - lag, False)  # noqa: B023
