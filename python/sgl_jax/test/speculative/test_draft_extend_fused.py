from functools import partial

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sgl_jax.srt.speculative.draft_extend_fused import (
    _make_draft_extend_metadata,
    _make_target_verify_metadata,
    _per_dp_cumsum_device,
)


def _explicit_mesh(dp_size):
    if jax.device_count() < dp_size:
        pytest.skip(f"requires {dp_size} devices")
    return Mesh(
        np.asarray(jax.devices()).reshape(dp_size, -1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize(
    "dp_size,lens,expected",
    [
        (1, [4, 0], [0, 4, 4]),
        (2, [4, 0, 2, 3], [0, 4, 4, 0, 2, 5]),
        (2, [0, 0, 2, 0], [0, 0, 0, 0, 2, 2]),
    ],
)
def test_per_dp_cumsum_preserves_rank_boundaries_and_sharding(dp_size, lens, expected, use_jit):
    mesh = _explicit_mesh(dp_size)
    sharding = NamedSharding(mesh, P("data"))
    cumsum = partial(_per_dp_cumsum_device, dp_size=dp_size)
    if use_jit:
        cumsum = jax.jit(cumsum)

    with jax.set_mesh(mesh):
        values = jax.device_put(np.asarray(lens, dtype=np.int32), sharding)
        result = cumsum(values)

    # Each rank starts at zero; padding must not consume query/KV offsets.
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert result.dtype == np.int32
    assert result.sharding == sharding


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("target_verify", [False, True])
def test_fused_metadata_builds_sharded_query_and_kv_offsets(dp_size, target_verify):
    mesh = _explicit_mesh(dp_size)
    sharding = NamedSharding(mesh, P("data"))

    with jax.set_mesh(mesh):
        seq_lens = jax.device_put(np.asarray([5, 0, 3, 0][: dp_size * 2], np.int32), sharding)
        query_lens = jax.device_put(np.asarray([2, 0, 1, 0][: dp_size * 2], np.int32), sharding)
        allocated_lens = jax.device_put(np.asarray([8, 0, 4, 0][: dp_size * 2], np.int32), sharding)
        pages = jax.device_put(
            np.asarray([10, 11, 0, 0, 20, 0, 0, 0][: dp_size * 4], np.int32), sharding
        )
        old_metadata = FlashAttentionMetadata(page_indices=pages)
        if target_verify:
            make_metadata = jax.jit(
                partial(
                    _make_target_verify_metadata,
                    speculative_num_draft_tokens=2,
                    page_size=4,
                    dp_size=dp_size,
                )
            )
            verify_lens = jax.device_put(
                np.asarray([3, 0, 1, 0][: dp_size * 2], np.int32), sharding
            )
            result = make_metadata(old_metadata, verify_lens, allocated_lens)
        else:
            make_metadata = jax.jit(
                partial(_make_draft_extend_metadata, page_size=4, dp_size=dp_size)
            )
            result = make_metadata(old_metadata, seq_lens, allocated_lens, query_lens=query_lens)

    # Exercise fused EAGLE3 callers with padded slots and page-aligned KV lengths.
    expected_cu_q = [0, 2, 2, 0, 2, 2] if target_verify else [0, 2, 2, 0, 1, 1]
    np.testing.assert_array_equal(np.asarray(result.cu_q_lens), expected_cu_q[: dp_size * 3])
    np.testing.assert_array_equal(np.asarray(result.cu_kv_lens), [0, 8, 8, 0, 4, 4][: dp_size * 3])
    np.testing.assert_array_equal(np.asarray(result.seq_lens), np.asarray(seq_lens))
    np.testing.assert_array_equal(np.asarray(result.page_indices), np.asarray(pages))
    np.testing.assert_array_equal(np.asarray(result.distribution), [0, 1, 1] * dp_size)
    for value in jax.tree.leaves(result):
        assert value.sharding == sharding
