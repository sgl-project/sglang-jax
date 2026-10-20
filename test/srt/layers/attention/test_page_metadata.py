"""Compact and token-based FA/MLA metadata must agree on values and sharding."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

PAGE_SIZE = 4
FIELDS = ("cu_q_lens", "cu_kv_lens", "page_indices", "seq_lens", "distribution")


def _batches(dp, mode, *, capacity=4, empty_last_rank=False, all_empty=False):
    pages = np.zeros((dp, capacity), dtype=np.int32)
    pages[0, :3] = [8, 3, 11]
    if dp == 2:
        pages[1, 0] = 5
    seq_lens = np.array([[6, 3], [1, 0]], dtype=np.int32)[:dp].copy()
    query_lens = np.array([[2, 1], [1, 0]], dtype=np.int32)[:dp].copy()
    if empty_last_rank:
        pages[-1] = seq_lens[-1] = query_lens[-1] = 0
    if all_empty:
        pages[:] = seq_lens[:] = query_lens[:] = 0
    token_locations = np.where(
        pages[:, :, None] > 0,
        pages[:, :, None] * PAGE_SIZE + np.arange(PAGE_SIZE, dtype=np.int32),
        0,
    )
    legacy = SimpleNamespace(
        dp_size=dp,
        per_dp_bs_size=2,
        forward_mode=mode,
        seq_lens=seq_lens.ravel(),
        extend_seq_lens=query_lens.ravel(),
        cache_loc=token_locations.ravel(),
    )
    compact = SimpleNamespace(**vars(legacy), allocated_page_indices=pages.ravel())
    compact.cache_loc = None  # The compact consumer must not read token locations.
    compact.allocated_page_indices.setflags(write=False)
    return legacy, compact, pages


def _backend(dp, *, decode_page_limit=None, swa_mapping=None):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(dp, 4 // dp),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    return SimpleNamespace(
        mesh=mesh,
        page_size=PAGE_SIZE,
        attention_data_partition_axis="data",
        _decode_page_limit=lambda batch: decode_page_limit,
        swa_index_mapping=swa_mapping,
    )


class PageMetadataTest(unittest.TestCase):
    def _assert_equal(self, expected, actual):
        for name in FIELDS:
            reference, got = getattr(expected, name), getattr(actual, name)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(reference), err_msg=name)
            self.assertEqual(got.dtype, reference.dtype, name)
            self.assertEqual(got.sharding, reference.sharding, name)

    def test_fa_and_mla_match_token_metadata(self):
        for backend_type in (FlashAttention, MLAAttentionBackend):
            for dp, empty_rank in ((1, False), (2, False), (2, True)):
                for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                    with self.subTest(
                        backend=backend_type, dp=dp, empty_rank=empty_rank, mode=mode
                    ):
                        legacy, compact, pages = _batches(dp, mode, empty_last_rank=empty_rank)
                        backend = _backend(dp)
                        reference = backend_type.get_forward_metadata(backend, legacy)
                        actual = backend_type.get_forward_metadata(backend, compact)
                        self._assert_equal(reference, actual)
                        np.testing.assert_array_equal(
                            np.asarray(actual.page_indices), pages.ravel()
                        )
                        np.testing.assert_array_equal(np.asarray(actual.cu_kv_lens)[:3], [0, 8, 12])

    def test_all_empty(self):
        for backend_type in (FlashAttention, MLAAttentionBackend):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(backend=backend_type, mode=mode):
                    legacy, compact, _ = _batches(2, mode, all_empty=True)
                    backend = _backend(2)
                    reference = backend_type.get_forward_metadata(backend, legacy)
                    actual = backend_type.get_forward_metadata(backend, compact)
                    self._assert_equal(reference, actual)
                    np.testing.assert_array_equal(np.asarray(actual.page_indices), np.zeros(8))
                    np.testing.assert_array_equal(np.asarray(actual.cu_kv_lens), np.zeros(6))

    def test_swa_uses_page_start_token_slots(self):
        slots = np.arange(16 * PAGE_SIZE, dtype=np.int32)
        for per_rank_mapping in (False, True):
            shifts = np.array([1, 2] if per_rank_mapping else [1, 1], dtype=np.int32)
            mappings = [
                ((slots // PAGE_SIZE + shift) % 16) * PAGE_SIZE + slots % PAGE_SIZE
                for shift in shifts
            ]
            for mapping in mappings:
                mapping[0] = 0
            backend = _backend(2, swa_mapping=mappings if per_rank_mapping else mappings[0])
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(per_rank_mapping=per_rank_mapping, mode=mode):
                    legacy, compact, pages = _batches(2, mode)
                    reference = FlashAttention.get_forward_metadata(backend, legacy)
                    actual = FlashAttention.get_forward_metadata(backend, compact)
                    self._assert_equal(reference, actual)
                    expected_swa = np.where(pages > 0, (pages + shifts[:, None]) % 16, 0)
                    np.testing.assert_array_equal(
                        np.asarray(actual.swa_page_indices), expected_swa.ravel()
                    )
                    np.testing.assert_array_equal(
                        np.asarray(actual.swa_page_indices), np.asarray(reference.swa_page_indices)
                    )
                    self.assertEqual(
                        actual.swa_page_indices.sharding, reference.swa_page_indices.sharding
                    )

    def test_fa_page_limit_only_bounds_decode(self):
        for limit in (2, 8):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(limit=limit, mode=mode):
                    legacy, compact, pages = _batches(2, mode, capacity=8)
                    backend = _backend(2, decode_page_limit=limit)
                    reference = FlashAttention.get_forward_metadata(backend, legacy)
                    actual = FlashAttention.get_forward_metadata(backend, compact)
                    self._assert_equal(reference, actual)
                    expected_pages = pages[:, : 2 * limit] if mode == ForwardMode.DECODE else pages
                    np.testing.assert_array_equal(
                        np.asarray(actual.page_indices), expected_pages.ravel()
                    )


if __name__ == "__main__":
    unittest.main()
