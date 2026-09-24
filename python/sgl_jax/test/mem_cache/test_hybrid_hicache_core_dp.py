"""Multi-device CPU regression for SWA lock receipts after split (CPU4 suite)."""

from sgl_jax.test.mem_cache.test_hybrid_hicache_core import (
    check_receipt_acquired_after_split,
)


def test_rank1_page128_receipt_acquired_after_split():
    check_receipt_acquired_after_split(page=128, dp_size=2, rank=1)
