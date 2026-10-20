# Copyright 2026 The SGLang-JAX Authors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import jax
import jax.numpy as jnp

HEAD_DIM: Final = 256
MAX_HEAD_DIM: Final = HEAD_DIM
KV_ROWS: Final = 256
FRAGMENT_ROWS: Final = 16
STATE_COLS: Final = 128
GUARD_ROWS: Final = 128
GUARD_GROUPS: Final = GUARD_ROWS // FRAGMENT_ROWS
DEFAULT_Q_ROWS: Final = 512
MAX_Q_ROWS: Final = 1024
RESIDENT_HEAD_ROWS: Final = 8192
MIN_HEAD_GROUP: Final = RESIDENT_HEAD_ROWS // MAX_Q_ROWS
MAX_HEAD_GROUP: Final = RESIDENT_HEAD_ROWS // DEFAULT_Q_ROWS
KV_FRAGMENTS: Final = KV_ROWS // FRAGMENT_ROWS
DEFAULT_Q_FRAGMENTS: Final = DEFAULT_Q_ROWS // FRAGMENT_ROWS
MAX_Q_FRAGMENTS: Final = MAX_Q_ROWS // FRAGMENT_ROWS

PIPELINE_PERIOD: Final = 64
MRB_SLOTS: Final = 64
MRB_ADDRESS_STRIDE: Final = 4
QK_POP_DELAY: Final = 28
PV_ISSUE_DELAY: Final = 32
PV_POP_DELAY: Final = 60
RHS_PREFETCH_LEAD: Final = 16
PROBABILITY_SLOTS: Final = PV_ISSUE_DELAY - QK_POP_DELAY
RHS_SLOTS: Final = 2
TEMPLATE_CACHE_SIZE: Final = 32

DEFAULT_KV_LOOKAHEAD: Final = 2
SUPPORTED_KV_LOOKAHEAD: Final = (2, 4, 6)
SUPPORTED_IO_BUFFERS: Final = (1, 2)
MAX_IO_BUFFERS: Final = 2
MIN_KV_TILES: Final = 2
KV_PAIR_TILES: Final = 2
KV_STREAMS: Final = 2
KV_RING_MARGIN: Final = 3
MASK_EXTRA_TILES: Final = 2
WIDE_Q_THRESHOLD: Final = 32768
BF16_BYTES: Final = 2
WORD_BYTES: Final = 4
# Two half-width FP32 states plus the full-width FP32 accumulator, in BF16 bytes.
FP32_STATE_FACTOR: Final = 4
MIB: Final = 1024 * 1024
EXPLICIT_VMEM_BYTES: Final = 62 * MIB
VMEM_LIMIT_BYTES: Final = 63 * MIB
NATIVE_CHANNELS: Final = 128
NATIVE_HEAD_ALIGNMENT: Final = 8
NATIVE_TOKEN_ALIGNMENT: Final = 32
BF16_BITS: Final = 16
BF16_PER_WORD: Final = WORD_BYTES // BF16_BYTES
QKV_INPUTS: Final = 3
SCALAR_PREFETCH_COUNT: Final = 7

DEFAULT_SM_SCALE: Final = HEAD_DIM**-0.5
DEFAULT_GUARD_LIMIT: Final = 1e20
DEFAULT_GUARD_MIN: Final = 1e-20
LOG2_E: Final = math.log2(math.e)

type ScalarIndex = int | jax.Array
type HeadValue = float | int | jax.Array

type ProbabilityCarry = tuple[jax.Array, ...]

type RecoveryCarry = tuple[jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class AttentionOptions:
    causal: bool = True
    window_left: int = -1
    window_right: int = -1
    sm_scale: float = DEFAULT_SM_SCALE
    logits_soft_cap: float = 0.0
    mask_value: float = -math.inf
    guard_limit: float = DEFAULT_GUARD_LIMIT
    identity_qk_scales: bool = True
    identity_v_scale: bool = True
    has_sink: bool = False

    def __post_init__(self) -> None:
        if self.window_left < -1 or self.window_right < -1:
            raise ValueError("Window bounds must be -1 or nonnegative")
        if not math.isfinite(self.sm_scale) or self.sm_scale <= 0:
            raise ValueError("sm_scale must be finite and positive")
        if not math.isfinite(self.logits_soft_cap) or self.logits_soft_cap < 0:
            raise ValueError("logits_soft_cap must be finite and nonnegative")
        if math.isnan(self.mask_value) or self.mask_value == math.inf:
            raise ValueError("mask_value must be finite or -inf")


@dataclass(frozen=True)
class BufferPlan:
    """Cached tile prefix plus rotating slots; ring_slots=0 means full residency."""

    cache_tiles: int = 0
    ring_slots: int = 1

    @property
    def slot_count(self) -> int:
        return self.cache_tiles + self.ring_slots

    def slot(self, block: ScalarIndex) -> ScalarIndex:
        if self.cache_tiles and self.ring_slots:
            return jnp.where(
                block < self.cache_tiles,
                block,
                self.cache_tiles + (block - self.cache_tiles) % self.ring_slots,
            )
        return block % self.slot_count


@dataclass(frozen=True)
class TuningConfig:
    """Static IO tuning; auto preserves blocked D256 and native narrower IO."""

    kv_lookahead: int = DEFAULT_KV_LOOKAHEAD
    kv_cache_tiles: int | None = None
    output_buffers: int | None = None
    q_residency: int = DEFAULT_Q_ROWS
    q_buffers: int = 1
    kv_ring_buffers: int | None = None
    io_layout: Literal["auto", "native", "blocked"] = "auto"

    @property
    def head_group(self) -> int:
        return RESIDENT_HEAD_ROWS // self.q_residency

    def __post_init__(self) -> None:
        if self.io_layout not in ("auto", "native", "blocked"):
            raise ValueError("io_layout must be auto, native, or blocked")
        if type(self.q_buffers) is not int or self.q_buffers not in SUPPORTED_IO_BUFFERS:
            raise ValueError("q_buffers must be 1 or 2")
        if self.kv_ring_buffers is not None and (
            type(self.kv_ring_buffers) is not int
            or (
                self.kv_ring_buffers != 0
                and self.kv_ring_buffers < self.kv_lookahead + KV_RING_MARGIN
            )
        ):
            raise ValueError("kv_ring_buffers must be None, 0, or at least kv_lookahead + 3")
        if type(self.q_residency) is not int or self.q_residency not in (
            DEFAULT_Q_ROWS,
            MAX_Q_ROWS,
        ):
            raise ValueError("q_residency must be 512 or 1024")
        if type(self.kv_lookahead) is not int or self.kv_lookahead not in SUPPORTED_KV_LOOKAHEAD:
            raise ValueError("kv_lookahead must be 2, 4, or 6")
        if self.kv_cache_tiles is not None and (
            type(self.kv_cache_tiles) is not int or self.kv_cache_tiles < 0
        ):
            raise ValueError("kv_cache_tiles must be None or a nonnegative integer")
        if self.output_buffers is not None and (
            type(self.output_buffers) is not int or self.output_buffers not in SUPPORTED_IO_BUFFERS
        ):
            raise ValueError("output_buffers must be None, 1, or 2")


DEFAULT_TUNING: Final = TuningConfig()


def select_tuning(
    q_tokens: int,
    kv_tokens: int,
    sequences: int,
    heads: int,
    kv_heads: int,
    *,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    mask_value: float = -math.inf,
) -> TuningConfig:
    """Select Q1024/H8 for long single-sequence causal MHA; otherwise Q512/H16."""
    wide = (
        min(q_tokens, kv_tokens) >= WIDE_Q_THRESHOLD
        and sequences == 1
        and heads == kv_heads
        and causal
        and window_size == (-1, -1)
        and mask_value == -math.inf
    )
    return TuningConfig(q_residency=MAX_Q_ROWS if wide else DEFAULT_Q_ROWS)


@dataclass(frozen=True)
class IOPlan:
    """Capacity allocation in the 62 MiB explicit VMEM budget."""

    q: BufferPlan
    kv: BufferPlan
    output: BufferPlan

    @classmethod
    def create(
        cls,
        packed_kv_tokens: int,
        kv_group_heads: int,
        tuning: TuningConfig = DEFAULT_TUNING,
        *,
        q_staging_bytes: int = 0,
        kv_tile_bytes: int | None = None,
        output_staging_bytes: int = 0,
    ) -> IOPlan:
        if packed_kv_tokens < 0 or not 1 <= kv_group_heads <= MAX_HEAD_GROUP:
            raise ValueError("Expected nonnegative KV capacity and 1..16 KV heads/program")
        rows, heads = tuning.q_residency, tuning.head_group
        q_bytes = heads * rows * HEAD_DIM * BF16_BYTES
        kv_bytes = kv_tile_bytes or KV_STREAMS * kv_group_heads * KV_ROWS * HEAD_DIM * BF16_BYTES

        fixed = (
            q_bytes * (FP32_STATE_FACTOR + tuning.q_buffers)
            + (rows // KV_ROWS + MASK_EXTRA_TILES) * rows * KV_ROWS * WORD_BYTES
        )
        available = EXPLICIT_VMEM_BYTES - fixed - q_staging_bytes * tuning.q_buffers
        output_bytes = q_bytes + output_staging_bytes
        tiles = max(MIN_KV_TILES, (packed_kv_tokens + (KV_ROWS - 1)) // KV_ROWS)
        outputs = tuning.output_buffers or MAX_IO_BUFFERS
        q = BufferPlan(ring_slots=tuning.q_buffers)
        automatic = tuning.kv_cache_tiles is None and tuning.kv_ring_buffers is None
        capacity = (available - output_bytes * outputs) // kv_bytes
        if tuning.kv_ring_buffers == 0 or (automatic and tiles <= capacity):
            cache_tiles = tiles if tuning.kv_cache_tiles is None else tuning.kv_cache_tiles
            if not tiles <= cache_tiles <= capacity:
                raise ValueError("ring=0 requires enough cache for all KV tiles within VMEM")
            return cls(q, BufferPlan(cache_tiles, 0), BufferPlan(ring_slots=outputs))
        outputs = tuning.output_buffers or 1
        capacity = (available - output_bytes * outputs) // kv_bytes
        ring_slots = tuning.kv_ring_buffers or tuning.kv_lookahead + KV_RING_MARGIN
        cache_tiles = (
            capacity - ring_slots if tuning.kv_cache_tiles is None else tuning.kv_cache_tiles
        )
        if cache_tiles < 0 or cache_tiles + ring_slots > capacity:
            raise ValueError("Tuning configuration exceeds the 62 MiB explicit VMEM budget")
        return cls(q, BufferPlan(cache_tiles, ring_slots), BufferPlan(ring_slots=outputs))
