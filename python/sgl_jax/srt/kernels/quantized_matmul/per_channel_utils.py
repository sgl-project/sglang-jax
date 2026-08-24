# SPDX-License-Identifier: Apache-2.0
"""Exact tuning registry and lazy loader for per-channel quantized matmul."""

import functools
import importlib
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PerChannelTunedKey(NamedTuple):
    """Identity of one validated per-channel local matmul workload."""

    tpu_version: int
    n_batch: int
    n_out: int
    n_in: int
    x_dtype: str
    x_q_dtype: str
    w_q_dtype: str


class PerChannelTunedValue(NamedTuple):
    """Hashable tile value without eagerly importing the Pallas package."""

    batch_block_size: int
    out_block_size: int
    in_block_size: int
    n_lane_multiplier: int = 1


class PerChannelTunedEntry(NamedTuple):
    """Validated Pallas tile for one exact workload."""

    tuned_value: PerChannelTunedValue


# Entries are added only after correctness and same-boundary DOT comparison on
# the exact local shape. Production promotion additionally requires repeat and
# whole-model validation. This table is not seeded from the shared block-wise
# tuning table and has no nearest/default fallback. W8A8 entries are kept
# separate from W8A16 by ``x_q_dtype``.
#
# The TPU7 W8A16 entries below were selected by exact-shape searches on
# GLM-5.2 TP=1 local workloads. The M=2048 entries and the two M=2 Full
# Indexer additions come from a constrained Cartesian ring-16/XProf sweep;
# end-to-end serving validation is tracked separately. The TPU7 W8A8 entries
# cover the exact M=2/2048 GLM-5.2 workloads selected by the corrected-numerics
# E1 screen plus the E3/E5 three-seed repeats. The dense_gate_up M=2048 entry
# deliberately uses the VMEM-safe runner-up instead of the isolated-op winner.
# Backend selection is global and lives in QuantizationConfig. The registry
# only maps exact workloads to validated Pallas tiles.
PER_CHANNEL_TUNED_ENTRIES: dict[PerChannelTunedKey, PerChannelTunedEntry] = {
    # M=2 attention projections.
    PerChannelTunedKey(7, 2, 2048, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 2048, 3072))
    ),
    PerChannelTunedKey(7, 2, 16384, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 4096, 2048))
    ),
    PerChannelTunedKey(7, 2, 576, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 576, 3072))
    ),
    PerChannelTunedKey(7, 2, 6144, 16384, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 2048, 8192))
    ),
    # M=2 dense-prefix MLP. gate_proj and up_proj share one physical key.
    PerChannelTunedKey(7, 2, 12288, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 4096, 3072))
    ),
    PerChannelTunedKey(7, 2, 6144, 12288, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 2048, 6144))
    ),
    # M=2 Full Indexer projections, validated by the M=2048 all-workload plus
    # missing-M=2-indexer Cartesian sweep.
    PerChannelTunedKey(7, 2, 4096, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 4096, 1024))
    ),
    PerChannelTunedKey(7, 2, 128, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 128, 3072))
    ),
    # M=8 evaluation workloads. Selected by the exact-shape TPU7 XProf sweep
    # exp-gh4so4dlad; all entries passed correctness and three-run CV <= 2%.
    PerChannelTunedKey(7, 8, 2048, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 512, 6144))
    ),
    PerChannelTunedKey(7, 8, 16384, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 2048, 2048))
    ),
    PerChannelTunedKey(7, 8, 576, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 256, 6144))
    ),
    PerChannelTunedKey(7, 8, 6144, 16384, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 512, 16384))
    ),
    PerChannelTunedKey(7, 8, 12288, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 4096, 3072))
    ),
    PerChannelTunedKey(7, 8, 6144, 12288, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 512, 12288))
    ),
    PerChannelTunedKey(7, 8, 4096, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 4096, 2048))
    ),
    PerChannelTunedKey(7, 8, 128, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 128, 6144))
    ),
    # M=1024 attention projections.
    PerChannelTunedKey(7, 1024, 2048, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 1024, 3072))
    ),
    PerChannelTunedKey(7, 1024, 16384, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 2048))
    ),
    PerChannelTunedKey(7, 1024, 576, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 576, 1536))
    ),
    PerChannelTunedKey(7, 1024, 6144, 16384, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 1536, 4096))
    ),
    # M=1024 dense-prefix MLP.
    PerChannelTunedKey(7, 1024, 12288, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    PerChannelTunedKey(7, 1024, 6144, 12288, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    # M=1024 Full Indexer launch coverage. See the M=2 note above.
    PerChannelTunedKey(7, 1024, 4096, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 2048))
    ),
    PerChannelTunedKey(7, 1024, 128, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 128, 1536))
    ),
    # M=2048 attention projections.
    PerChannelTunedKey(7, 2048, 2048, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    PerChannelTunedKey(7, 2048, 16384, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 4096, 2048))
    ),
    PerChannelTunedKey(7, 2048, 576, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 576, 2048))
    ),
    PerChannelTunedKey(7, 2048, 6144, 16384, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 1536, 4096))
    ),
    # M=2048 dense-prefix MLP. gate_proj and up_proj share one physical key.
    PerChannelTunedKey(7, 2048, 12288, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        # The isolated-op winner (1024, 1024, 6144) needs 51.85 MiB scoped
        # VMEM in the full model and exceeds TPU7's 48 MiB limit.  This sweep's
        # rank-3 candidate is only 0.07% slower and is selected for full-model
        # serving validation with half the BK.
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    PerChannelTunedKey(7, 2048, 6144, 12288, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    # M=2048 Full Indexer projections.
    PerChannelTunedKey(7, 2048, 4096, 2048, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 2048))
    ),
    PerChannelTunedKey(7, 2048, 128, 6144, "bfloat16", "bfloat16", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(512, 128, 6144))
    ),
    # M=2 W8A8 workloads. q_a/q_b use the E3 repeat winners; the remaining
    # entries are corrected-numerics E1 candidates for whole-model validation.
    PerChannelTunedKey(7, 2, 2048, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 2048, 3072))
    ),
    PerChannelTunedKey(7, 2, 16384, 2048, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 4096, 2048))
    ),
    PerChannelTunedKey(7, 2, 576, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 576, 6144))
    ),
    PerChannelTunedKey(7, 2, 6144, 16384, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 1536, 8192))
    ),
    PerChannelTunedKey(7, 2, 12288, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 2048, 6144))
    ),
    PerChannelTunedKey(7, 2, 6144, 12288, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2, 2048, 6144))
    ),
    PerChannelTunedKey(7, 2, 4096, 2048, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(8, 2048, 2048))
    ),
    PerChannelTunedKey(7, 2, 128, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(4, 128, 6144))
    ),
    # M=2048 W8A8 workloads. o_proj/dense_down use the E5 repeat winners.
    PerChannelTunedKey(7, 2048, 2048, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(512, 2048, 6144))
    ),
    PerChannelTunedKey(7, 2048, 16384, 2048, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 4096, 2048))
    ),
    PerChannelTunedKey(7, 2048, 576, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 576, 6144))
    ),
    PerChannelTunedKey(7, 2048, 6144, 16384, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 1536, 4096))
    ),
    PerChannelTunedKey(7, 2048, 12288, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 3072))
    ),
    PerChannelTunedKey(7, 2048, 6144, 12288, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(512, 3072, 4096))
    ),
    PerChannelTunedKey(7, 2048, 4096, 2048, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(1024, 2048, 2048))
    ),
    PerChannelTunedKey(7, 2048, 128, 6144, "bfloat16", "float8_e4m3fn", "float8_e4m3fn"): (
        PerChannelTunedEntry(PerChannelTunedValue(2048, 128, 3072))
    ),
}


@functools.lru_cache(maxsize=1)
def get_current_tpu_version() -> int:
    """Return the current TPU major version, or ``-1`` off TPU."""
    try:
        device_kind = jax.devices()[0].device_kind
    except Exception:
        return -1
    match = re.match(r"^TPU[^\d]*(\d+)", device_kind)
    return int(match.group(1)) if match is not None else -1


def get_exact_per_channel_tuned_entry(
    *,
    n_batch: int,
    n_out: int,
    n_in: int,
    x_dtype: jnp.dtype,
    x_q_dtype: jnp.dtype,
    w_q_dtype: jnp.dtype,
    tpu_version: int | None = None,
) -> PerChannelTunedEntry | None:
    """Return only an exact validated entry; never infer or default a tile."""
    key = PerChannelTunedKey(
        tpu_version=get_current_tpu_version() if tpu_version is None else tpu_version,
        n_batch=int(n_batch),
        n_out=int(n_out),
        n_in=int(n_in),
        x_dtype=jnp.dtype(x_dtype).name,
        x_q_dtype=jnp.dtype(x_q_dtype).name,
        w_q_dtype=jnp.dtype(w_q_dtype).name,
    )
    return PER_CHANNEL_TUNED_ENTRIES.get(key)


@functools.lru_cache(maxsize=1)
def get_per_channel_kernel():
    """Lazily import the TPU-only Pallas implementation."""
    package = __package__ or "sgl_jax.srt.kernels.quantized_matmul"
    module = importlib.import_module(f"{package}.quantized_matmul_kernels.kernel")
    return module.quantized_matmul_kernel
