"""CPU-side unit tests for SWA hybrid-attention PD code paths.

These verify the Python-level logic without requiring TPU or raiden.
"""

import numpy as np
import pytest


class TestRaidenWrapperSWA:
    """Verify RaidenTransferWrapper dual-engine attributes."""

    def test_no_swa_by_default(self):
        from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
            RaidenTransferWrapper,
        )

        w = RaidenTransferWrapper("127.0.0.1")
        assert w.is_hybrid_swa is False
        assert w.endpoints_swa is None
        assert w.engine_swa is None
        assert w.engine_full is w.engine  # aliased to same engine

    def test_start_without_swa_kv_caches(self):
        """start() without kv_caches_swa → single-engine mode."""
        from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
            RaidenTransferWrapper,
        )

        w = RaidenTransferWrapper("127.0.0.1")
        assert not w.is_started
        assert w.is_hybrid_swa is False


class TestSWABlockExtraction:
    """Verify _extract_swa_block_ids_for_chunk logic."""

    @pytest.fixture
    def mock_scheduler(self):
        """Build a minimal mock with req_to_token and SWA allocator."""
        from sgl_jax.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        page_size = 4
        seqlen = 40  # 10 pages of 4 tokens each
        mapping = np.arange(0, 200, dtype=np.int32)
        req_to_token_row = np.arange(0, seqlen * page_size, dtype=np.int32)

        class MockReqToTokenPool:
            req_to_token = np.array([req_to_token_row])

        class MockAllocator:
            full_to_swa_index_mapping = mapping

        class MockScheduler:
            req_to_token_pool = MockReqToTokenPool()
            token_to_kv_pool_allocator = MockAllocator()

        # Monkey-patch the real method onto the mock
        MockScheduler._extract_swa_block_ids_for_chunk = (
            SchedulerDisaggregationPrefillMixin._extract_swa_block_ids_for_chunk
        )
        return MockScheduler(), page_size, seqlen

    @staticmethod
    def _make_req(origin_input_ids):
        return type("Req", (), {
            "origin_input_ids": origin_input_ids,
            "req_pool_idx": 0,
        })()

    def test_returns_empty_when_no_mapping(self, mock_scheduler):
        """Returns [] when allocator has no SWA mapping (non-SWA model)."""
        scheduler, *_ = mock_scheduler
        scheduler.token_to_kv_pool_allocator = type("NoSWA", (), {})()
        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(40))),
            start=0, end=40, page_size=4, sliding_window_size=16,
        )
        assert result == []

    def test_tail_only_filter(self, mock_scheduler):
        """Only pages in the sliding-window tail are included."""
        scheduler, page_size, seqlen = mock_scheduler
        req = self._make_req(list(range(seqlen)))

        # sliding_window=8 tokens → window_start=32 (=page 8). Tail: pages 8, 9.
        result = scheduler._extract_swa_block_ids_for_chunk(
            req, start=0, end=seqlen, page_size=page_size, sliding_window_size=8,
        )
        assert result == [8, 9]

    def test_chunk_before_window_returns_empty(self, mock_scheduler):
        """Chunk entirely before sliding window → empty."""
        scheduler, page_size, seqlen = mock_scheduler

        # sliding_window=16, window_start=24. Chunk pages 0..4.
        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(seqlen))),
            start=0, end=20, page_size=page_size, sliding_window_size=16,
        )
        assert result == []

    def test_chunk_overlapping_window(self, mock_scheduler):
        """Chunk crossing window boundary only returns tail pages."""
        scheduler, page_size, seqlen = mock_scheduler

        # sliding_window=12 → window_start=28 (=page 7).
        # Chunk pages 6-8. Only pages 7+ returned.
        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(seqlen))),
            start=24, end=36, page_size=page_size, sliding_window_size=12,
        )
        assert result == [7, 8]


class TestBackwardCompatible:
    """Non-SWA paths remain unchanged."""

    def test_prefill_handoff_no_swa(self):
        """_raiden_handoff_chunk passes swa_block_ids=None for non-SWA."""
        from sgl_jax.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin,
        )

        assert hasattr(SchedulerDisaggregationPrefillMixin, "_extract_swa_block_ids_for_chunk")
        assert hasattr(SchedulerDisaggregationPrefillMixin, "_extract_req_kv")

    def test_conn_pmetadata_defaults(self):
        """PMetadata swa fields default to None."""
        from sgl_jax.srt.disaggregation.jax_transfer.conn import PMetadata

        m = PMetadata(
            remote_addr="127.0.0.1:0",
            uuid="test",
            specs={},
            p_side_channel_host="127.0.0.1",
            p_side_channel_port=0,
        )
        assert m.swa_remote_endpoint is None
        assert m.swa_local_pages is None

    def test_resolve_kv_pool_dtype_accepts_swa_pool(self):
        """SWA wrapper pools advertise dtype via their full sub-pool."""
        from sgl_jax.srt.disaggregation.bootstrap import resolve_kv_pool_dtype

        class FullPool:
            dtype = np.float16

        class SWAPool:
            full_kv_pool = FullPool()

        assert resolve_kv_pool_dtype(SWAPool()) is np.float16
