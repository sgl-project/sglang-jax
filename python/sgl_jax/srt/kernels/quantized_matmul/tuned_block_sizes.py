# SPDX-License-Identifier: Apache-2.0
"""Tuned block sizes for quantized matmul kernel."""

import logging
import re
from typing import NamedTuple

import jax

logger = logging.getLogger(__name__)


class TunedKey(NamedTuple):
    tpu_version: int
    n_batch: int
    n_out: int
    n_in: int
    x_q_dtype: str
    w_q_dtype: str


class TunedValue(NamedTuple):
    batch_block_size: int
    out_block_size: int
    in_block_size: int
    n_lane_multiplier: int = 1


TUNED_BLOCK_SIZES_RAW = {
    # go/keep-sorted start
    (7, 1, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (1, 128, 8192),
    (7, 1, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (1, 1024, 4096),
    (7, 1, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (1, 4096, 4096),
    (7, 1, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (1, 8192, 512),
    (7, 1, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (1, 512, 4096),
    (7, 2, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (2, 128, 8192),
    (7, 2, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (2, 256, 2048),
    (7, 2, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (2, 4096, 2048),
    (7, 2, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (2, 4096, 1024),
    (7, 2, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (2, 512, 4096),
    (7, 4, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (4, 128, 4096),
    (7, 4, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (4, 1024, 1024),
    (7, 4, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (4, 4096, 2048),
    (7, 4, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (4, 4096, 256),
    (7, 4, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (4, 4096, 512),
    (7, 8, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (8, 128, 2048),
    (7, 8, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (8, 256, 2048),
    (7, 8, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (8, 512, 8192),
    (7, 8, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (8, 8192, 1024),
    (7, 8, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (8, 8192, 2048),
    (7, 16, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (16, 128, 1024),
    (7, 16, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (16, 1024, 8192),
    (7, 16, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (16, 2048, 2048),
    (7, 16, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (16, 2048, 256),
    (7, 16, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (16, 256, 4096),
    (7, 32, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (32, 128, 2048),
    (7, 32, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (32, 1024, 8192),
    (7, 32, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (32, 2048, 2048),
    (7, 32, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (32, 2048, 1024),
    (7, 32, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (32, 8192, 2048),
    (7, 64, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (64, 128, 8192),
    (7, 64, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (64, 1024, 4096),
    (7, 64, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (64, 4096, 4096),
    (7, 64, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (64, 2048, 256),
    (7, 64, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (64, 2048, 4096),
    (7, 8192, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (512, 128, 8192),
    (7, 8192, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (256, 1024, 8192),
    (7, 8192, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (1024, 2048, 2048),
    (7, 8192, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (1024, 4096, 1024),
    (7, 8192, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (1024, 2048, 4096),
    (7, 16384, 128, 8192, "float8_e4m3fn", "float8_e4m3fn"): (1024, 128, 8192),
    (7, 16384, 1024, 8192, "float8_e4m3fn", "float8_e4m3fn"): (512, 1024, 8192),
    (7, 16384, 4096, 8192, "float8_e4m3fn", "float8_e4m3fn"): (256, 4096, 8192),
    (7, 16384, 8192, 1024, "float8_e4m3fn", "float8_e4m3fn"): (1024, 4096, 1024),
    (7, 16384, 8192, 4096, "float8_e4m3fn", "float8_e4m3fn"): (256, 8192, 4096),
    # go/keep-sorted end
}

TUNED_BLOCK_SIZES: dict[TunedKey, TunedValue] = {
    TunedKey(*key): TunedValue(*value) for key, value in TUNED_BLOCK_SIZES_RAW.items()
}

DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: 48 * 1024 * 1024}


def get_device_vmem_limit() -> int:
    tpu_version = get_tpu_version()
    if tpu_version not in DEVICE_VMEM_LIMIT:
        logger.warning(
            "VMEM limit for TPU version %d not found. Using default VMEM limit " "of 96MiB",
            tpu_version,
        )
        return 96 * 1024 * 1024
    return DEVICE_VMEM_LIMIT[tpu_version]


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    match = re.match(r"^TPU[^\d]*(\d+)", kind)
    if match is None:
        return -1
    return int(match.group(1))


def get_key(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
) -> TunedKey:
    """Returns the key for the given parameters."""
    return TunedKey(
        get_tpu_version(),
        n_batch,
        n_out,
        n_in,
        x_q_dtype,
        w_q_dtype,
    )


def get_tuned_block_sizes(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: str,
    w_q_dtype: str,
) -> TunedValue:
    """Retrieve the tuned block sizes for the given parameters.

    Args:
        n_batch: The batch size.
        n_out: The number of output features.
        n_in: The number of input features.
        x_q_dtype: The data type of the activation ('int8' or 'float8_e4m3fn').
        w_q_dtype: The data type of the weight ('int8' or 'float8_e4m3fn').

    Returns:
        tuple: A tuple containing the batch_block_size, out_block_size, and
        in_block_size.
    """
    key = get_key(
        n_batch,
        n_out,
        n_in,
        x_q_dtype,
        w_q_dtype,
    )
    tuned_value = TUNED_BLOCK_SIZES.get(key)
    if tuned_value is None:
        logger.warning("Couldn't find tuned sizes for the quantized matmul kernel with %s", key)
        return TunedValue(128, 128, 128)
    else:
        return tuned_value
