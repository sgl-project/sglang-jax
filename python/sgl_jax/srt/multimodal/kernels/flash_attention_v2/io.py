# Copyright 2026 The SGLang-JAX Authors.
# SPDX-License-Identifier: Apache-2.0

"""Blocked and native Q/K/V transfers and output writes."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.ref import Ref

from .config import (
    BF16_BITS,
    BF16_PER_WORD,
    FRAGMENT_ROWS,
    HEAD_DIM,
    KV_ROWS,
    MAX_HEAD_GROUP,
    NATIVE_CHANNELS,
    NATIVE_HEAD_ALIGNMENT,
    BufferPlan,
    ScalarIndex,
)


@dataclass
class BlockIO:
    """Shared Q/KV/output DMA. All policy and tuple iteration are trace-time."""

    plan: BufferPlan
    tensor_refs: tuple[tuple[Ref, Ref, Ref], ...]  # HBM, VMEM, semaphore per tensor
    token_base: ScalarIndex
    head_start: ScalarIndex
    head_count: int
    loaded_tiles: ScalarIndex = 0  # Loaded prefix; only cache_tiles remain reusable.

    def buffer(self, block: ScalarIndex, tensor: int = 0) -> Ref:
        ref = self.tensor_refs[tensor][1]
        return ref if len(ref.shape) == 3 else ref.at[self.plan.slot(block)]

    def transfer(self, block: ScalarIndex, start: bool, *, rows=None) -> None:
        def copy():
            slot = self.plan.slot(block)
            for tensor, (hbm, _, sem) in enumerate(self.tensor_refs):
                external = hbm.at[
                    self.token_base // hbm.shape[2] + block,
                    pl.ds(self.head_start, self.head_count),
                    :,
                    :,
                ]
                local = self.buffer(block, tensor)
                if rows is not None:  # output: only the valid, 16-aligned rows
                    local, external = (
                        external.at[:, pl.ds(0, rows), :],
                        local.at[:, pl.ds(0, rows), :],
                    )
                dma = pltpu.make_async_copy(external, local, sem if not sem.shape else sem.at[slot])
                dma.start() if start else dma.wait()

        if self.plan.cache_tiles:
            missing = block >= self.loaded_tiles
            if self.plan.ring_slots:
                missing = (block >= self.plan.cache_tiles) | missing
            pl.when(missing)(copy)
        else:
            copy()

    def reload(self, block: ScalarIndex) -> None:
        if self.plan.ring_slots:

            def copy():
                self.transfer(block, True)
                self.transfer(block, False)

            if self.plan.cache_tiles:
                pl.when(block >= self.plan.cache_tiles)(copy)
            else:
                copy()


def _align(value, alignment):
    return pl.cdiv(value, alignment) * alignment


def _read_native_head(raw, head, row_start, row_count, *, pad_to=HEAD_DIM):
    """Extract one head from BF16 pairs in token-major VMEM."""
    stride = raw.shape[2] // BF16_PER_WORD
    pieces = []
    for fold in range(raw.shape[0]):
        words = raw.bitcast(jnp.uint32).at[fold].reshape(raw.shape[1] * stride, NATIVE_CHANNELS)
        if isinstance(head, int):
            value = words[pl.ds(row_start * stride + head // BF16_PER_WORD, row_count, stride)]
        else:
            # Recovery and irregular GQA use a dynamic head. Constant slices
            # avoid an unsupported sub-tile dynamic VMEM address.
            base = row_start * stride
            pair = head // BF16_PER_WORD
            value = words[pl.ds(base, row_count, stride)]
            for lane in range(1, stride):
                other = words[pl.ds(base + lane, row_count, stride)]
                value = jnp.where(pair == lane, other, value)
        bits = value >> ((head % BF16_PER_WORD) * BF16_BITS)
        pieces.append(pltpu.bitcast(bits.astype(jnp.uint16), jnp.bfloat16))
    if len(pieces) == 1 and pad_to == HEAD_DIM:
        pieces.append(jnp.zeros((row_count, NATIVE_CHANNELS), jnp.bfloat16))
    return jnp.concatenate(pieces, axis=1)


@dataclass
class TokenMajorBlockIO(BlockIO):
    """Read token-major Q/K/V and write output using the shared buffer plan."""

    raw_refs: tuple = ()
    role: str = "kv"
    raw_heads: int = MAX_HEAD_GROUP
    sequence_length: object = None

    def transfer(self, block, start, *, rows=None):
        def copy():
            slot = self.plan.slot(block)
            for tensor, (hbm, compute, sem) in enumerate(self.tensor_refs):
                raw = self.raw_refs[tensor].at[slot]
                tile_rows = raw.shape[1]
                origin = self.token_base + block * tile_rows
                count = jnp.minimum(tile_rows, hbm.shape[0] - origin) if rows is None else rows
                head_base = (
                    self.head_start
                    if self.role != "kv"
                    else self.head_start // NATIVE_HEAD_ALIGNMENT * NATIVE_HEAD_ALIGNMENT
                )
                if self.role == "output" and start:
                    values = self.buffer(block, tensor)
                    for fold in range(raw.shape[0]):
                        for row in range(0, tile_rows, NATIVE_CHANNELS):
                            raw[fold, pl.ds(row, NATIVE_CHANNELS), : self.head_count, :] = (
                                jnp.transpose(
                                    values[
                                        :,
                                        pl.ds(row, NATIVE_CHANNELS),
                                        pl.ds(fold * NATIVE_CHANNELS, NATIVE_CHANNELS),
                                    ],
                                    (1, 0, 2),
                                )
                            )
                for fold in range(raw.shape[0]):
                    external = hbm.at[
                        pl.ds(origin, count),
                        pl.ds(head_base, self.raw_heads),
                        pl.ds(fold * NATIVE_CHANNELS, NATIVE_CHANNELS),
                    ]
                    local = raw.at[fold, pl.ds(0, count), : self.raw_heads, :]
                    src, dst = (local, external) if self.role == "output" else (external, local)
                    dma = pltpu.make_async_copy(src, dst, sem.at[slot, fold])
                    dma.start() if start else dma.wait()
                if self.role != "output" and not start:
                    remaining = self.sequence_length - block * tile_rows

                    @pl.when(remaining < tile_rows)
                    def clear_tail(raw=raw, remaining=remaining, tile_rows=tile_rows):
                        # A cached/ring tile is sanitized once on arrival.
                        # Complete tiles keep the hot RHS path free of masks.
                        def clear(row_block, _):
                            row = row_block * FRAGMENT_ROWS
                            shape = (FRAGMENT_ROWS, self.raw_heads, NATIVE_CHANNELS)
                            valid = row + lax.broadcasted_iota(jnp.int32, shape, 0) < remaining
                            for fold in range(raw.shape[0]):
                                view = raw.at[fold, pl.ds(row, FRAGMENT_ROWS), : self.raw_heads, :]
                                view[...] = jnp.where(valid, view[...], 0)

                        lax.fori_loop(0, tile_rows // FRAGMENT_ROWS, clear, None)

                if self.role == "query" and not start:
                    destination = self.buffer(block, tensor)

                    for index in range(self.head_count):
                        for row in range(0, tile_rows, KV_ROWS):
                            value = _read_native_head(
                                raw, index, row, KV_ROWS, pad_to=destination.shape[-1]
                            )
                            destination[index, pl.ds(row, KV_ROWS), :] = value

        if self.plan.cache_tiles:
            missing = block >= self.loaded_tiles
            if self.plan.ring_slots:
                missing = (block >= self.plan.cache_tiles) | missing
            pl.when(missing)(copy)
        else:
            copy()


def _pack_sequences(
    x: jax.Array, cu: jax.Array, padded_heads: int, guard_rows: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Align segments and guard tails for contiguous block/head/row/channel DMA."""
    sequences = cu.shape[0] - 1
    block = guard_rows
    aligned = ((cu + block - 1) // block) * block + jnp.arange(
        sequences + 1, dtype=jnp.int32
    ) * block
    size = (pl.cdiv(x.shape[0], block) + sequences + 1) * block
    position = jnp.arange(size, dtype=jnp.int32)
    seq = jnp.minimum(jnp.searchsorted(aligned[1:], position, side="right"), sequences - 1)
    offset = position - aligned[seq]
    origin = cu[seq] + offset
    valid = (offset >= 0) & (offset < cu[seq + 1] - cu[seq])
    if x.shape[0]:
        packed = jnp.where(valid[:, None, None], x[jnp.clip(origin, 0, x.shape[0] - 1)], 0)
    else:
        packed = jnp.zeros((size, x.shape[1], HEAD_DIM), x.dtype)
    packed = jnp.pad(packed.transpose(1, 0, 2), ((0, padded_heads - x.shape[1]), (0, 0), (0, 0)))
    packed = packed.reshape(padded_heads, size // block, block, HEAD_DIM).transpose(1, 0, 2, 3)
    metadata = jnp.stack((aligned, cu), axis=1)
    original = jnp.arange(x.shape[0], dtype=jnp.int32)
    owners = jnp.minimum(jnp.searchsorted(cu[1:], original, side="right"), sequences - 1)
    restore = aligned[owners] + original - cu[owners]
    return packed, metadata, restore
