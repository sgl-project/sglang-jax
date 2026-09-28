"""Metadata transfer parity, including DP layout and overlapping host writes."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.jax_utils import (
    _metadata_unpacker,
    device_array,
    packed_device_array,
)


@pytest.fixture(params=[1, 2, 4])
def sharding(request):
    dp = request.param
    if len(jax.devices()) < dp:
        pytest.skip(
            f"Requires {dp} devices (use XLA_FLAGS=--xla_force_host_platform_device_count=4)"
        )
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(dp, -1), ("data", "tensor"))
    return NamedSharding(mesh, PartitionSpec("data"))


def assert_same(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
        if not isinstance(a, jax.Array):
            continue
        assert a.dtype == b.dtype
        assert a.sharding.is_equivalent_to(b.sharding, a.ndim)
        for shard in a.addressable_shards:
            np.testing.assert_array_equal(shard.data, np.asarray(b)[shard.index])


def test_dtypes_shapes_and_transfer_count(sharding):
    values = {
        "tokens": np.arange(32, dtype=np.int32),
        "cache": np.arange(256, dtype=np.int32),
        "strided": np.arange(64, dtype=np.int32).reshape(8, 8)[:, ::2],
        "scales": np.arange(8, dtype=np.float32),
        "mask": np.array([True, False] * 4),
        "empty": np.empty(0, dtype=np.int32),
        "absent": None,
    }
    expected = device_array(values, sharding)
    with patch.object(
        jax, "make_array_from_callback", wraps=jax.make_array_from_callback
    ) as upload:
        actual = packed_device_array(values, sharding)
    assert upload.call_count == 3  # int32, float32, bool; no dtype promotion
    assert_same(actual, expected)
    again = packed_device_array(values, sharding)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(again)):
        assert a.sharding is b.sharding


def test_overlapping_batches_own_their_host_storage(sharding):
    values = [np.arange(4096, dtype=np.int32), np.arange(8, dtype=np.int32)]
    expected = [value.copy() for value in values]
    first = packed_device_array(values, sharding)
    for value in values:
        value.fill(-1)
    second = packed_device_array(values, sharding)
    del values
    for actual, reference in zip(first, expected):
        np.testing.assert_array_equal(actual, reference)
    for actual in second:
        np.testing.assert_array_equal(actual, -np.ones(actual.shape, dtype=np.int32))


def test_unpack_is_local_to_each_dp_rank(sharding):
    shapes = ((32,), (128,), (8, 4))
    ranks, sizes, buffer_sharding, unpack = _metadata_unpacker(shapes, sharding)
    buffer = device_array(np.zeros((ranks, sum(sizes)), np.int32), buffer_sharding)
    hlo = unpack.lower(buffer).compile().as_text().lower()
    for collective in ("all-gather(", "all-to-all(", "collective-permute(", "all-reduce("):
        assert collective not in hlo


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.IDLE])
def test_forward_batch_optional_metadata(sharding, mode):
    token_count, batch_size = 32, 8
    values = dict(
        input_ids=np.arange(token_count, dtype=np.int32),
        seq_lens=np.array([4, 0, 2, 0, 0, 0, 1, 0], dtype=np.int32),
        out_cache_loc=np.arange(token_count, dtype=np.int32),
        positions=np.arange(token_count, dtype=np.int32),
        req_pool_indices=np.arange(batch_size, dtype=np.int32),
        cache_loc=np.arange(256, dtype=np.int32),
        extend_prefix_lens=np.zeros(batch_size, dtype=np.int32) if mode.is_extend() else None,
        extend_seq_lens=np.ones(batch_size, dtype=np.int32) if mode.is_extend() else None,
        lora_scalings=np.arange(token_count, dtype=np.float32),
        lora_token_indices=np.arange(token_count, dtype=np.int32),
        lora_ranks=np.full(token_count, 8, dtype=np.int32),
        recurrent_indices=np.arange(batch_size, dtype=np.int32),
        recurrent_cow_src_indices=np.arange(batch_size, dtype=np.int32),
        recurrent_track_indices=np.arange(batch_size, dtype=np.int32),
        recurrent_track_mask=np.arange(batch_size, dtype=np.int32) % 2,
    )
    batch = SimpleNamespace(
        **values,
        bid=0,
        forward_mode=mode,
        mrope_positions=np.arange(3 * token_count, dtype=np.int32).reshape(3, -1),
        input_embedding=np.ones((token_count, 4), dtype=np.float32),
        apply_for_deepstack=True,
        deepstack_visual_embedding=np.ones((2, token_count, 4), dtype=np.float32),
        lora_ids=None,
        spec_info_padded=None,
        spec_algorithm=None,
        capture_hidden_mode=None,
    )
    runner = SimpleNamespace(
        mesh=sharding.mesh,
        attn_backend=None,
        model_config=SimpleNamespace(is_embedding=False, hf_config=SimpleNamespace()),
    )
    with patch("sgl_jax.srt.model_executor.forward_batch_info.packed_device_array", device_array):
        expected = ForwardBatch.init_new(batch, runner)
    actual = ForwardBatch.init_new(batch, runner)
    assert_same(actual, expected)
    # Absent optional fields must stay absent, including an all-None pytree.
    for name in values:
        if name.startswith(("lora_", "recurrent_")):
            setattr(batch, name, None)
    batch.apply_for_deepstack = False
    batch.input_embedding = batch.mrope_positions = None
    with patch.object(
        jax, "make_array_from_callback", wraps=jax.make_array_from_callback
    ) as upload:
        plain = ForwardBatch.init_new(batch, runner)
    assert upload.call_count == 2  # packed small metadata + the un-copied cache_loc
    assert plain.lora_scalings is plain.recurrent_indices is None
    assert packed_device_array((None, {"absent": None}), sharding) == (None, {"absent": None})
