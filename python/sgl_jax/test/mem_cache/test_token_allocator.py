# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_token_allocator.py -v
"""TokenToKVPoolAllocator (page_size 1): sorted free list and first-fit runs.

The free list is kept sorted, and alloc() takes the first run of at least
need_size consecutive slots when the front of the list is not contiguous, so
prefill allocations are physically contiguous whenever such a run exists.
"""

import os
import unittest
from unittest import mock

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from sgl_jax.srt.mem_cache import allocator as allocator_module
from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.test.test_utils import CustomTestCase


def _first_fit_start(free, need_size):
    """Reference: start of the slots alloc() takes from the sorted free list."""
    if need_size <= 1 or free[need_size - 1] - free[0] == need_size - 1:
        return 0
    run_start = 0
    for i in range(1, len(free) + 1):
        if i == len(free) or free[i] != free[i - 1] + 1:
            if i - run_start >= need_size:
                return run_start
            run_start = i
    return 0


class TestTokenToKVPoolAllocator(CustomTestCase):
    def test_alloc_takes_first_contiguous_run(self):
        alloc = TokenToKVPoolAllocator(size=32, kvcache=None)
        first, second, third = (alloc.alloc(8) for _ in range(3))
        np.testing.assert_array_equal(second, np.arange(9, 17))
        alloc.free(first[2:4])
        alloc.free(second[1:6])
        np.testing.assert_array_equal(
            alloc.free_slots[0], [3, 4, 10, 11, 12, 13, 14] + list(range(25, 33))
        )
        # The front [3, 4, 10, ...] is not contiguous: take the first run of 5.
        np.testing.assert_array_equal(alloc.alloc(5), [10, 11, 12, 13, 14])
        # No run of 9 is left: fall back to the front of the list.
        np.testing.assert_array_equal(alloc.alloc(9), [3, 4] + list(range(25, 32)))
        np.testing.assert_array_equal(alloc.free_slots[0], [32])
        self.assertIsNone(alloc.alloc(2))
        self.assertEqual(alloc.available_size(), 1)

    def test_random_ops_match_reference(self):
        """Every alloc result and free list matches the sorted-list first-fit model."""
        for chunk in (1, 3, 4096):
            for seed in range(20):
                with (
                    self.subTest(chunk=chunk, seed=seed),
                    mock.patch.object(
                        allocator_module, "_FIRST_FIT_SCAN_CHUNK", chunk, create=True
                    ),
                ):
                    self._run_random_ops(seed)

    def _run_random_ops(self, seed):
        rng = np.random.default_rng(seed)
        dp_size = int(rng.integers(1, 3))
        alloc = TokenToKVPoolAllocator(size=600 * dp_size, kvcache=None, dp_size=dp_size)
        expected = [alloc.free_slots[rank].copy() for rank in range(dp_size)]
        live = [[] for _ in range(dp_size)]
        for _ in range(200):
            rank = int(rng.integers(dp_size))
            op = rng.random()
            if op < 0.5 or not live[rank]:
                need = int(rng.choice([1, 2, 3, rng.integers(1, 40), rng.integers(1, 300)]))
                out = alloc.alloc(need, dp_rank=rank)
                if need > len(expected[rank]):
                    self.assertIsNone(out)
                    continue
                start = _first_fit_start(expected[rank], need)
                np.testing.assert_array_equal(out, expected[rank][start : start + need])
                expected[rank] = np.delete(expected[rank], np.s_[start : start + need])
                live[rank].append(out)
            elif op < 0.8:
                # Free part of a request, in any order and either index dtype.
                slots = live[rank].pop(int(rng.integers(len(live[rank]))))
                slots = rng.permutation(slots).astype(rng.choice([np.int32, np.int64]))
                keep = int(rng.integers(len(slots)))
                if keep:
                    live[rank].append(slots[:keep])
                alloc.free(slots[keep:], dp_rank=rank)
                expected[rank] = np.sort(np.concatenate([expected[rank], slots[keep:]]))
            else:
                alloc.free_group_begin()
                freed = [[] for _ in range(dp_size)]
                for _ in range(int(rng.integers(1, 4))):
                    r = int(rng.integers(dp_size))
                    if live[r]:
                        slots = live[r].pop(int(rng.integers(len(live[r]))))
                        alloc.free(slots, dp_rank=r)
                        freed[r].append(slots)
                alloc.free_group_end()
                for r in range(dp_size):
                    if freed[r]:
                        expected[r] = np.sort(np.concatenate([expected[r]] + freed[r]))
            for r in range(dp_size):
                self.assertEqual(alloc.free_slots[r].dtype, expected[r].dtype)
                np.testing.assert_array_equal(alloc.free_slots[r], expected[r])

    def test_free_does_not_resort_the_free_list(self):
        """free() merges the freed slots into the sorted list; it must not sort
        the whole list, which costs O(N log N) on every finished request."""
        size = 100_000
        for use_group in (False, True):
            alloc = TokenToKVPoolAllocator(size=size, kvcache=None)
            requests = [alloc.alloc(1000) for _ in range(50)]
            freed = np.concatenate([requests[3], requests[17][::-1], requests[40][:7]])
            with mock.patch.object(np, "sort", wraps=np.sort) as sort_spy:
                if use_group:
                    alloc.free_group_begin()
                    for part in np.split(freed, [1000, 2000]):
                        alloc.free(part)
                    alloc.free_group_end()
                else:
                    alloc.free(freed)
            sorted_sizes = [np.size(call.args[0]) for call in sort_spy.call_args_list]
            self.assertLessEqual(max(sorted_sizes, default=0), len(freed), sorted_sizes)
            remaining = np.setdiff1d(np.arange(1, size + 1), np.concatenate(requests))
            np.testing.assert_array_equal(
                alloc.free_slots[0], np.sort(np.concatenate([remaining, freed]))
            )


if __name__ == "__main__":
    unittest.main()
