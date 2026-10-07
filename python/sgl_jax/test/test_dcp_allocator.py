"""Slice 2: paged allocator virtual token indices under DCP."""

from __future__ import annotations

import numpy as np

from sgl_jax.srt.layers.dcp.layout import owner, physical_index, virtual_page_size
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator


class _DummyKV:
    pass


def _allocator(size: int, page_size: int, dcp_size: int = 1) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        kvcache=_DummyKV(),
        debug_mode=True,
        dcp_size=dcp_size,
    )


def test_dcp_size_1_alloc_matches_pre_dcp_indices():
    page_size = 64
    a = _allocator(size=4096, page_size=page_size, dcp_size=1)
    b = _allocator(size=4096, page_size=page_size, dcp_size=1)
    need = 128
    ia = a.alloc(need)
    ib = b.alloc(need)
    assert ia is not None and ib is not None
    np.testing.assert_array_equal(ia, ib)
    # Page 0 is reserved; first two pages are 64..191.
    np.testing.assert_array_equal(ia, np.arange(page_size, page_size + need, dtype=np.int32))
    assert a.page_size == page_size
    assert a.physical_page_size == page_size
    assert a.dcp_size == 1


def test_one_virtual_page_is_one_physical_page_per_rank():
    page_size, dcp_size = 64, 16
    virt_page = virtual_page_size(page_size, dcp_size)
    assert virt_page == 1024
    alloc = _allocator(size=4096, page_size=page_size, dcp_size=dcp_size)
    assert alloc.page_size == virt_page
    assert alloc.physical_page_size == page_size
    indices = alloc.alloc(virt_page)
    assert indices is not None
    assert len(indices) == virt_page
    for rank in range(dcp_size):
        owned = indices[owner(indices, dcp_size) == rank]
        phys = physical_index(owned, dcp_size)
        assert len(phys) == page_size
        np.testing.assert_array_equal(phys, np.arange(phys[0], phys[0] + page_size))
        # Same physical page id on every rank (page 1 → slots 64..127).
        np.testing.assert_array_equal(phys, np.arange(page_size, 2 * page_size))


def test_extend_across_virtual_page_boundary_uses_seq_owner():
    page_size, dcp_size = 64, 16
    virt_page = virtual_page_size(page_size, dcp_size)
    alloc = _allocator(size=4096, page_size=page_size, dcp_size=dcp_size)
    first = alloc.alloc_extend(
        prefix_lens=[0],
        seq_lens=[virt_page],
        last_loc=[-1],
        extend_num_tokens=virt_page,
    )
    assert first is not None
    assert len(first) == virt_page
    last_loc = int(first[-1])
    extra = alloc.alloc_extend(
        prefix_lens=[virt_page],
        seq_lens=[virt_page + 1],
        last_loc=[last_loc],
        extend_num_tokens=1,
    )
    assert extra is not None
    v = int(extra[0])
    # Sequence position virt_page (=1024) is the first token of the next virtual page.
    assert owner(v, dcp_size) == virt_page % dcp_size
    assert owner(v, dcp_size) == 0
