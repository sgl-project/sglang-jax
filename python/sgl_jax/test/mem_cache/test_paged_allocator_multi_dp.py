# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_paged_allocator_multi_dp.py -v

import os
import unittest

# Set up multi-device simulation for testing
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestPagedAllocatorMultiDP(unittest.TestCase):
    """Test PagedTokenToKVPoolAllocator with multiple DP ranks (dp_size > 1)."""

    def setUp(self):
        """Setup test parameters."""
        self.pool_size = 1024  # Total pool size
        self.page_size = 16  # Page size
        self.kv_head_num = 8
        self.head_dim = 128
        self.layer_num = 2
        self.dtype = jnp.bfloat16

    def _create_allocator(self, dp_size=1, page_size=None):
        """Helper to create allocator with specified dp_size."""
        if page_size is None:
            page_size = self.page_size

        # Create KV cache
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
        )

        # Create allocator
        allocator = PagedTokenToKVPoolAllocator(
            size=self.pool_size,
            page_size=page_size,
            kvcache=kv_cache,
            dp_size=dp_size,
        )

        return allocator

    def test_multi_dp_basic_allocation(self):
        """Test basic allocation and free for multiple DP ranks."""
        dp_size = 3
        allocator = self._create_allocator(dp_size=dp_size)

        # Calculate expected pages per rank
        num_pages = self.pool_size // self.page_size
        pages_per_rank = num_pages // dp_size

        # Test allocation and free for each rank
        for rank in range(dp_size):
            # Allocate 2 pages (32 tokens)
            alloc_size = 2 * self.page_size
            indices = allocator.alloc(alloc_size, dp_rank=rank)

            # Verify allocation succeeded
            self.assertIsNotNone(indices, f"Allocation for rank {rank} should succeed")
            self.assertEqual(len(indices), alloc_size)

            # Verify available size decreased for this rank
            expected_free_pages = pages_per_rank - 2
            self.assertEqual(
                len(allocator.free_pages[rank]),
                expected_free_pages,
                f"Rank {rank} should have {expected_free_pages} free pages",
            )

            # Free the pages
            allocator.free(indices, dp_rank=rank)

            # Merge released pages back to free pool
            allocator.merge_and_sort_free(dp_rank=rank)

            # Verify pages returned to correct rank's pool
            self.assertEqual(
                len(allocator.free_pages[rank]),
                pages_per_rank,
                f"Rank {rank} should have all pages back after free",
            )

    def test_free_group_batching_single_rank(self):
        """Test free_group_begin/end batching for single DP rank (dp_size=1)."""
        allocator = self._create_allocator(dp_size=1)

        # Allocate 3 chunks
        alloc_size = 2 * self.page_size
        indices1 = allocator.alloc(alloc_size, dp_rank=0)
        indices2 = allocator.alloc(alloc_size, dp_rank=0)
        indices3 = allocator.alloc(alloc_size, dp_rank=0)

        initial_free_pages = len(allocator.free_pages[0])

        # Begin batching
        allocator.free_group_begin()

        # Verify free_group is a list of lists
        self.assertIsInstance(allocator.free_group, list)
        self.assertEqual(len(allocator.free_group), 1)
        self.assertEqual(allocator.free_group[0], [])

        # Free multiple chunks
        allocator.free(indices1, dp_rank=0)
        allocator.free(indices2, dp_rank=0)
        allocator.free(indices3, dp_rank=0)

        # Verify indices NOT immediately freed (accumulated in list)
        self.assertEqual(len(allocator.free_group[0]), 3)
        self.assertEqual(len(allocator.free_pages[0]), initial_free_pages)

        # End batching
        allocator.free_group_end()

        # Merge released pages back to free pool
        allocator.merge_and_sort_free(dp_rank=0)

        # Verify all indices freed in batch
        expected_free_pages = initial_free_pages + 6  # 3 allocations × 2 pages each
        self.assertEqual(len(allocator.free_pages[0]), expected_free_pages)

        # Verify free_group reset to []
        self.assertEqual(allocator.free_group[0], [])

    def test_free_group_batching_multi_rank(self):
        """Test free_group_begin/end batching for multiple DP ranks (dp_size>1)."""
        dp_size = 3
        allocator = self._create_allocator(dp_size=dp_size)

        # Allocate pages for each rank
        alloc_size = 2 * self.page_size
        allocated_indices = {}
        for rank in range(dp_size):
            allocated_indices[rank] = [
                allocator.alloc(alloc_size, dp_rank=rank),
                allocator.alloc(alloc_size, dp_rank=rank),
            ]

        # Record initial free page counts
        initial_free_pages = {rank: len(allocator.free_pages[rank]) for rank in range(dp_size)}

        # Begin batching
        allocator.free_group_begin()

        # Verify free_group initialized as list of lists
        self.assertIsInstance(allocator.free_group, list)
        self.assertEqual(len(allocator.free_group), dp_size)
        for rank in range(dp_size):
            self.assertEqual(allocator.free_group[rank], [])

        # Free for ranks in mixed order
        allocator.free(allocated_indices[1][0], dp_rank=1)
        allocator.free(allocated_indices[0][0], dp_rank=0)
        allocator.free(allocated_indices[2][0], dp_rank=2)
        allocator.free(allocated_indices[1][1], dp_rank=1)
        allocator.free(allocated_indices[0][1], dp_rank=0)
        allocator.free(allocated_indices[2][1], dp_rank=2)

        # Verify indices accumulated in dict
        for rank in range(dp_size):
            self.assertEqual(len(allocator.free_group[rank]), 2)

        # Verify NOT immediately freed
        for rank in range(dp_size):
            self.assertEqual(
                len(allocator.free_pages[rank]),
                initial_free_pages[rank],
                f"Rank {rank} pages should not be freed yet",
            )

        # End batching
        allocator.free_group_end()

        # Merge released pages back to free pool for each rank
        for rank in range(dp_size):
            allocator.merge_and_sort_free(dp_rank=rank)

        # Verify each rank's indices freed to correct pool
        for rank in range(dp_size):
            expected_pages = initial_free_pages[rank] + 4  # 2 allocations × 2 pages each
            self.assertEqual(
                len(allocator.free_pages[rank]),
                expected_pages,
                f"Rank {rank} should have correct pages after batch free",
            )

        # Verify free_group reset to empty lists
        for rank in range(dp_size):
            self.assertEqual(allocator.free_group[rank], [])

    def test_per_rank_pool_isolation(self):
        """Test that per-rank memory pools are isolated."""
        dp_size = 2
        allocator = self._create_allocator(dp_size=dp_size)

        num_pages = self.pool_size // self.page_size
        pages_per_rank = num_pages // dp_size

        # Allocate all pages for rank 0
        all_indices_rank0 = []
        for _ in range(pages_per_rank):
            indices = allocator.alloc(self.page_size, dp_rank=0)
            if indices is not None:
                all_indices_rank0.append(indices)

        # Verify rank 0 pool exhausted
        self.assertEqual(len(allocator.free_pages[0]), 0)
        rank0_alloc = allocator.alloc(self.page_size, dp_rank=0)
        self.assertIsNone(rank0_alloc, "Rank 0 should be out of pages")

        # Verify rank 1 pool still has free pages
        self.assertEqual(len(allocator.free_pages[1]), pages_per_rank)
        rank1_alloc = allocator.alloc(self.page_size, dp_rank=1)
        self.assertIsNotNone(rank1_alloc, "Rank 1 should still have free pages")

        # Free some pages to rank 0
        allocator.free(all_indices_rank0[0], dp_rank=0)
        allocator.free(all_indices_rank0[1], dp_rank=0)

        # Merge released pages back to free pool
        allocator.merge_and_sort_free(dp_rank=0)

        # Verify rank 0 pool has 2 free pages
        self.assertEqual(len(allocator.free_pages[0]), 2)

        # Verify rank 1 pool unchanged (still pages_per_rank - 1)
        self.assertEqual(len(allocator.free_pages[1]), pages_per_rank - 1)

    def test_cross_rank_free_group_no_contamination(self):
        """Test that batched frees don't contaminate other ranks."""
        dp_size = 3
        allocator = self._create_allocator(dp_size=dp_size)

        # Allocate pages for each rank
        alloc_size = 3 * self.page_size
        allocated_indices = {}
        for rank in range(dp_size):
            allocated_indices[rank] = allocator.alloc(alloc_size, dp_rank=rank)

        # Record initial free page counts
        initial_free_pages = {rank: len(allocator.free_pages[rank]) for rank in range(dp_size)}

        # Begin batched free
        allocator.free_group_begin()

        # Free each rank's pages with correct dp_rank
        for rank in range(dp_size):
            allocator.free(allocated_indices[rank], dp_rank=rank)

        # End batched free
        allocator.free_group_end()

        # Merge released pages back to free pool for each rank
        for rank in range(dp_size):
            allocator.merge_and_sort_free(dp_rank=rank)

        # Verify each rank got back exactly 3 pages (no contamination)
        for rank in range(dp_size):
            expected_pages = initial_free_pages[rank] + 3
            self.assertEqual(
                len(allocator.free_pages[rank]),
                expected_pages,
                f"Rank {rank} should have exactly {expected_pages} pages (no cross-contamination)",
            )

        # Verify total pages unchanged
        num_pages = self.pool_size // self.page_size
        pages_per_rank = num_pages // dp_size
        for rank in range(dp_size):
            self.assertEqual(
                len(allocator.free_pages[rank]),
                pages_per_rank,
                f"Rank {rank} should have all pages back",
            )

    def test_free_group_structure(self):
        """Test that free_group uses correct data structure."""
        # Single rank: verify free_group is list of lists
        allocator_single = self._create_allocator(dp_size=1)
        allocator_single.free_group_begin()
        self.assertIsInstance(
            allocator_single.free_group,
            list,
            "Single-rank allocator should use list for free_group",
        )
        self.assertEqual(len(allocator_single.free_group), 1)
        self.assertEqual(allocator_single.free_group[0], [])

        # Multi rank: verify free_group is list of lists
        allocator_multi = self._create_allocator(dp_size=3)
        allocator_multi.free_group_begin()
        self.assertIsInstance(
            allocator_multi.free_group, list, "Multi-rank allocator should use list for free_group"
        )
        self.assertEqual(len(allocator_multi.free_group), 3)
        for rank in range(3):
            self.assertEqual(allocator_multi.free_group[rank], [])


if __name__ == "__main__":
    unittest.main()
