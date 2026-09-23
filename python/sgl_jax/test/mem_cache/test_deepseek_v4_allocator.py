"""Resource-level contracts for V4 history and SWA page ownership."""

import unittest

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)


def make_allocator(page_size=128, history_pages=4, swa_pages=4, dp_size=1):
    mesh = Mesh(np.asarray(jax.devices()[:dp_size]), ("data",))
    spec = DeepseekV4CacheSpec((0, 4, 128), head_dim=8, index_head_dim=4)
    pool = DeepseekV4TokenToKVPool(
        page_size * history_pages * dp_size,
        page_size * swa_pages * dp_size,
        page_size,
        spec,
        mesh,
        dp_size,
    )
    return DeepseekV4TokenToKVPoolAllocator(pool)


def assert_snapshot_equal(test, actual, expected):
    for current_family, saved_family in zip(actual[:6], expected[:6]):
        for current, saved in zip(current_family, saved_family):
            np.testing.assert_array_equal(current, saved)
    test.assertEqual(actual[6], expected[6])


class TestDeepseekV4Allocator(unittest.TestCase):
    def test_demand_and_failed_allocation_are_atomic(self):
        for page_size in (128, 256):
            with self.subTest(page_size=page_size):
                allocator = make_allocator(page_size, history_pages=1, swa_pages=2)
                first = allocator.alloc_extend([0], [page_size - 1], [-1], page_size - 1)
                self.assertEqual(allocator.full_available_size(), 0)
                self.assertEqual(allocator.swa_available_size(), page_size)
                demand = allocator.estimate_extend(
                    [page_size - 1], [page_size + 1], [int(first[-1])], 2
                )
                self.assertEqual((demand.history_pages, demand.swa_pages), (1, 1))
                before = allocator.backup_state()
                self.assertIsNone(
                    allocator.alloc_extend([page_size - 1], [page_size + 1], [int(first[-1])], 2)
                )
                assert_snapshot_equal(self, allocator.backup_state(), before)

    def test_partial_page_release_rejected_without_mutation(self):
        allocator = make_allocator()
        indices = allocator.alloc_extend([0], [129], [-1], 129)
        before = allocator.backup_state()
        with self.assertRaisesRegex(ValueError, "partial page"):
            allocator.free(indices[:127])
        assert_snapshot_equal(self, allocator.backup_state(), before)
        allocator.free(indices)
        self.assertEqual(allocator.full_available_size(), 4 * 128)

    def test_swa_reclamation_keeps_history(self):
        allocator = make_allocator()
        indices = allocator.alloc_extend([0], [256], [-1], 256)
        allocator.free_swa(indices[:128])
        self.assertEqual(allocator.full_available_size(), 2 * 128)
        self.assertEqual(allocator.swa_available_size(), 3 * 128)
        self.assertEqual(allocator.count_swa_mapped(indices[:128]), 0)
        self.assertEqual(allocator.count_swa_mapped(indices[128:]), 128)
        allocator.free(indices)
        self.assertEqual(allocator.full_available_size(), 4 * 128)
        self.assertEqual(allocator.swa_available_size(), 4 * 128)

    def test_prefill_to_decode_reuses_partial_page(self):
        allocator = make_allocator()
        prefix = allocator.alloc_extend([0], [127], [-1], 127)
        demand = allocator.estimate_decode([128], [int(prefix[-1])])
        self.assertEqual((demand.history_pages, demand.swa_pages), (0, 0))
        decoded = allocator.alloc_decode([128], [int(prefix[-1])])
        self.assertEqual(int(decoded[0]), int(prefix[-1]) + 1)
        allocator.free(np.concatenate((prefix, decoded)))
        self.assertEqual(allocator.full_available_size(), 4 * 128)

    def test_variable_length_batch_keeps_request_pages_isolated(self):
        for page_size in (128, 256):
            with self.subTest(page_size=page_size):
                allocator = make_allocator(page_size, history_pages=8, swa_pages=8)
                lengths = np.array([1, page_size + 1, 2 * page_size + 1])
                demand = allocator.estimate_extend([0, 0, 0], lengths, [-1, -1, -1], sum(lengths))
                self.assertEqual((demand.history_pages, demand.swa_pages), (6, 6))
                initial = allocator.alloc_extend([0, 0, 0], lengths, [-1, -1, -1], sum(lengths))
                per_request = np.split(initial, np.cumsum(lengths)[:-1])

                next_lengths = lengths + np.array([3, 3, 1])
                tails = [int(loc[-1]) for loc in per_request]
                demand = allocator.estimate_extend(lengths, next_lengths, tails, 7)
                self.assertEqual((demand.history_pages, demand.swa_pages), (0, 0))
                extension = allocator.alloc_extend(lengths, next_lengths, tails, 7)
                extra = np.split(extension, [3, 6])
                owned = [
                    np.concatenate((prefix, suffix)) for prefix, suffix in zip(per_request, extra)
                ]
                page_sets = [set(np.unique(loc // page_size)) for loc in owned]
                swa_mapping = allocator.full_to_swa_index_mapping
                swa_page_sets = [set(np.unique(swa_mapping[loc] // page_size)) for loc in owned]
                for i in range(len(owned)):
                    self.assertEqual(allocator.count_swa_mapped(owned[i]), len(owned[i]))
                    for j in range(i + 1, len(owned)):
                        self.assertTrue(page_sets[i].isdisjoint(page_sets[j]))
                        self.assertTrue(swa_page_sets[i].isdisjoint(swa_page_sets[j]))

                remaining_swa = [swa_mapping[loc].copy() for loc in owned[1:]]
                allocator.free(owned[0])
                self.assertEqual(allocator.count_swa_mapped(owned[0]), 0)
                for loc, expected in zip(owned[1:], remaining_swa):
                    np.testing.assert_array_equal(swa_mapping[loc], expected)
                    self.assertEqual(allocator.count_swa_mapped(loc), len(loc))
                    allocator.free(loc)
                self.assertEqual(allocator.full_available_size(), 8 * page_size)
                self.assertEqual(allocator.swa_available_size(), 8 * page_size)

    def test_grouped_release_and_rollback(self):
        allocator = make_allocator()
        first = allocator.alloc_extend([0], [128], [-1], 128)
        second = allocator.alloc_extend([0], [128], [-1], 128)
        original_mapping = allocator.full_to_swa_index_mapping
        before = allocator.backup_state()
        allocator.free_group_begin()
        allocator.free(first)
        allocator.free(second)
        self.assertEqual(allocator.full_available_size(), 2 * 128)
        allocator.free_group_end()
        self.assertEqual(allocator.full_available_size(), 4 * 128)
        allocator.restore_state(before)
        self.assertIs(allocator.full_to_swa_index_mapping, original_mapping)
        assert_snapshot_equal(self, allocator.backup_state(), before)

    def test_dp_rank_ledgers_are_isolated(self):
        if len(jax.devices()) < 2:
            self.skipTest("requires two JAX devices")
        allocator = make_allocator(dp_size=2)
        first = allocator.alloc_extend([0], [128], [-1], 128, dp_rank=0)
        self.assertEqual(allocator.full_available_size(0), 3 * 128)
        self.assertEqual(allocator.full_available_size(1), 4 * 128)
        second = allocator.alloc_extend([0], [128], [-1], 128, dp_rank=1)
        self.assertEqual(int(first[0]), int(second[0]))
        allocator.free(first, dp_rank=0)
        self.assertEqual(allocator.full_available_size(0), 4 * 128)
        self.assertEqual(allocator.full_available_size(1), 3 * 128)

    def test_swa_exhaustion_preserves_history_and_mapping(self):
        allocator = make_allocator(history_pages=3, swa_pages=1)
        first = allocator.alloc_extend([0], [128], [-1], 128)
        before = allocator.backup_state()
        self.assertEqual(allocator.full_available_size(), 2 * 128)
        self.assertEqual(allocator.swa_available_size(), 0)
        demand = allocator.estimate_extend([0], [1], [-1], 1)
        self.assertEqual((demand.history_pages, demand.swa_pages), (1, 1))
        self.assertFalse(allocator.can_allocate(demand))
        self.assertIsNone(allocator.alloc_extend([0], [1], [-1], 1))
        assert_snapshot_equal(self, allocator.backup_state(), before)
        allocator.free_swa(first)
        self.assertEqual(allocator.count_swa_mapped(first), 0)
        second = allocator.alloc_extend([0], [1], [-1], 1)
        self.assertEqual(len(second), 1)
        allocator.free(first)
        allocator.free(second)

    def test_invalid_extension_and_partial_swa_release_are_atomic(self):
        allocator = make_allocator()
        first = allocator.alloc_extend([0], [127], [-1], 127)
        before = allocator.backup_state()
        for prefix, seq, tail, count in (
            ([127], [129], [int(first[-1]) - 1], 2),
            ([127], [129], [int(first[-1])], 1),
            ([127, 127], [128, 128], [int(first[-1])] * 2, 2),
            ([0], [1], [128], 1),
        ):
            with self.subTest(prefix=prefix, seq=seq, tail=tail):
                with self.assertRaises(ValueError):
                    allocator.alloc_extend(prefix, seq, tail, count)
                assert_snapshot_equal(self, allocator.backup_state(), before)
        with self.assertRaisesRegex(ValueError, "partial page"):
            allocator.free_swa(first[:-1])
        assert_snapshot_equal(self, allocator.backup_state(), before)
        allocator.free(first)

    def test_invalid_grouped_release_does_not_commit(self):
        allocator = make_allocator()
        first = allocator.alloc_extend([0], [128], [-1], 128)
        allocator.free_group_begin()
        allocator.free(first[:-1])
        before = allocator.backup_state()
        with self.assertRaisesRegex(ValueError, "partial page"):
            allocator.free_group_end()
        assert_snapshot_equal(self, allocator.backup_state(), before)
        allocator.restore_state(before)
        allocator.free(first[-1:])
        allocator.free_group_end()
        self.assertEqual(allocator.full_available_size(), 4 * 128)


if __name__ == "__main__":
    unittest.main()
