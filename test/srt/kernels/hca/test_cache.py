"""HCA allocation, page-table, reuse, and recurrent copy-on-write tests."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from .common import HCATestFactory

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="HCA physical pools require TPU"
)
HCA_TEST = HCATestFactory()


def test_hca_allocator_lifecycle_and_page_tables():
    mesh, _, state_pool, request_pool, allocator = HCA_TEST.runtime(
        requests=2, max_context_len=512
    )
    requests = [HCA_TEST.request(), HCA_TEST.request()]
    with jax.set_mesh(mesh):
        req_indices = np.asarray(allocator.alloc(requests), np.int32)
        seq_lens = np.asarray([128, 257], np.int32)
        with pytest.raises(RuntimeError):
            allocator.page_tables(req_indices, seq_lens)  # capacity not grown yet
        allocator.ensure_compressed_capacity(req_indices, seq_lens)
        (
            window_page_indices,
            window_cu_kv_lens,
            compressed_page_indices,
            compressed_cu_kv_lens,
            compressed_kv_lens,
        ) = allocator.page_tables(req_indices, seq_lens)

        np.testing.assert_array_equal(compressed_kv_lens, [1, 2])
        assert np.all(window_page_indices > 0)
        assert compressed_page_indices[0] > 0
        assert window_cu_kv_lens[-1] == window_page_indices.size * allocator.page_size
        assert (
            compressed_cu_kv_lens[-1]
            == compressed_page_indices.size * allocator.page_size
        )

        src = request_pool.get_linear_recurrent_indices(req_indices)
        state_pool.state_buffers[0] = (
            state_pool.state_buffers[0]
            .at[src[0], 0]
            .set(7, out_sharding=P("data", None, None, None))
        )
        copied, conv = state_pool.copy_slots(
            np.asarray([src[0]], np.int32), np.asarray([src[1]], np.int32)
        )
        assert not conv
        np.testing.assert_array_equal(
            np.asarray(copied[0][src[1], 0]), np.asarray(copied[0][src[0], 0])
        )

        allocator.free(requests[0])
        replacement = HCA_TEST.request()
        assert allocator.alloc([replacement]) == [0]
        replacement_slot = request_pool.get_linear_recurrent_indices(
            np.asarray([replacement.req_pool_idx], np.int32)
        )[0]
        reset = np.asarray(state_pool.state_buffers[0])[replacement_slot]
        assert np.all(reset[:, 0] == 0)
        assert np.all(np.isneginf(reset[:, 1]))
