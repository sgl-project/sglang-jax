from __future__ import annotations

import math
from collections.abc import Sequence

import jax.numpy as jnp

TPU_TILE_SHAPE = (8, 128)
ENCODER_PAGE_SIZE = 128


def encoder_pool_block_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Return the physical shape of one page registered with Raiden."""

    if len(shape) != 2:
        raise ValueError("encoder embedding must be a matrix")
    rows, width = (int(dim) for dim in shape)
    if rows <= 0 or width <= 0:
        raise ValueError("encoder embedding dimensions must be positive")
    tile_width = math.prod(TPU_TILE_SHAPE)
    return max(2, rows), max(2, math.ceil(width / tile_width)), *TPU_TILE_SHAPE


def encoder_transfer_nbytes(shape: Sequence[int], dtype: object) -> int:
    """Return bytes transferred for all pages covering an embedding."""

    rows, width = shape
    pages = math.ceil(rows / ENCODER_PAGE_SIZE)
    return (
        pages
        * math.prod(encoder_pool_block_shape((ENCODER_PAGE_SIZE, width)))
        * jnp.dtype(dtype).itemsize
    )
