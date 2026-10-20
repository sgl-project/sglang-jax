"""The small-batch routing-index kernel (``permute_indices_small``) is an integer function of the
[tokens, topk] expert-id matrix; it must equal the jnp reference (stable argsort / bincount /
inverse) bit for bit, including duplicate ids, unused experts and N not a multiple of 128.
Runs in interpret mode on CPU; on a TPU host the real kernel is exercised."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.moe_permute_small.kernel import (
    PERMUTE_KERNEL_MAX_ROWS,
    permute_indices_reference,
    permute_indices_small,
)


def _check(ids, num_experts):
    got = permute_indices_small(ids, num_experts)
    ref = permute_indices_reference(ids, num_experts)
    assert len(got) == len(ref) == 4
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(r))


@pytest.mark.parametrize(
    "tokens,topk,num_experts", [(1, 8, 256), (3, 8, 64), (7, 4, 16), (32, 8, 256), (41, 6, 256)]
)
def test_random_routing_matches_reference(tokens, topk, num_experts):
    key = jax.random.PRNGKey(tokens * 131 + topk)
    ids = jax.random.randint(key, (tokens, topk), 0, num_experts, dtype=jnp.int32)
    assert tokens * topk <= PERMUTE_KERNEL_MAX_ROWS
    _check(ids, num_experts)


def test_duplicate_and_all_equal_ids():
    ids = jnp.full((16, 8), 5, dtype=jnp.int32)
    _check(ids, 64)
    ids = jnp.tile(
        jnp.arange(8, dtype=jnp.int32)[None, :], (16, 1)
    )  # every token the same 8 experts
    _check(ids, 8)


def test_max_rows_boundary():
    tokens, topk = PERMUTE_KERNEL_MAX_ROWS // 8, 8
    ids = jax.random.randint(jax.random.PRNGKey(0), (tokens, topk), 0, 256, dtype=jnp.int32)
    _check(ids, 256)
