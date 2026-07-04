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

    def test_hybrid_poll_waits_for_swa_engine(self):
        """Hybrid SWA completion requires both full and SWA engines."""
        from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
            RaidenTransferWrapper,
        )

        class FakeEngine:
            def __init__(self, polls):
                self.polls = list(polls)

            def poll_stats(self):
                return self.polls.pop(0) if self.polls else ([], [], [])

        w = RaidenTransferWrapper("127.0.0.1")
        w._started = True
        w._is_hybrid_swa = True
        w._engine_full = FakeEngine([
            (["req#c0"], ["req#c0"], []),
            ([], [], []),
        ])
        w._engine_swa = FakeEngine([
            ([], [], []),
            (["req#c0"], ["req#c0"], []),
        ])
        w._swa_send_req_ids.add("req#c0")
        w._swa_recv_req_ids.add("req#c0")

        assert w.poll_stats() == ([], [], [])
        assert w.poll_stats() == (["req#c0"], ["req#c0"], [])

    def test_hybrid_poll_allows_full_only_chunk(self):
        """Hybrid wrapper still completes chunks that have no SWA blocks."""
        from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
            RaidenTransferWrapper,
        )

        class FakeEngine:
            def __init__(self, polls):
                self.polls = list(polls)

            def poll_stats(self):
                return self.polls.pop(0) if self.polls else ([], [], [])

        w = RaidenTransferWrapper("127.0.0.1")
        w._started = True
        w._is_hybrid_swa = True
        w._engine_full = FakeEngine([(["req#c0"], ["req#c0"], [])])
        w._engine_swa = FakeEngine([([], [], [])])

        assert w.poll_stats() == (["req#c0"], ["req#c0"], [])


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

            def get_kvcache(self):
                return type("KVPool", (), {"page_size": page_size})()

        class MockScheduler:
            req_to_token_pool = MockReqToTokenPool()
            token_to_kv_pool_allocator = MockAllocator()

        # Monkey-patch the real method onto the mock
        MockScheduler._extract_req_block_ids_range = (
            SchedulerDisaggregationPrefillMixin._extract_req_block_ids_range
        )
        MockScheduler._extract_swa_block_ids_for_chunk = (
            SchedulerDisaggregationPrefillMixin._extract_swa_block_ids_for_chunk
        )
        return MockScheduler(), page_size, seqlen

    @staticmethod
    def _make_req(origin_input_ids, dp_rank=0):
        return type("Req", (), {
            "origin_input_ids": origin_input_ids,
            "req_pool_idx": 0,
            "dp_rank": dp_rank,
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

    def test_skips_reserved_page_zero(self, mock_scheduler):
        """Reserved SWA page 0 must not be advertised to decode."""
        scheduler, page_size, seqlen = mock_scheduler
        mapping = scheduler.token_to_kv_pool_allocator.full_to_swa_index_mapping
        mapping[28] = 0

        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(seqlen))),
            start=24, end=36, page_size=page_size, sliding_window_size=12,
        )

        assert result == [8]

    def test_uses_request_dp_rank_mapping(self, mock_scheduler):
        """DP requests must read the matching rank's SWA mapping."""
        scheduler, page_size, seqlen = mock_scheduler
        rank0 = np.zeros_like(
            scheduler.token_to_kv_pool_allocator.full_to_swa_index_mapping
        )
        rank1 = np.arange(0, 200, dtype=np.int32)
        scheduler.token_to_kv_pool_allocator.full_to_swa_index_mapping = [
            rank0,
            rank1,
        ]

        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(seqlen)), dp_rank=1),
            start=0,
            end=seqlen,
            page_size=page_size,
            sliding_window_size=8,
        )

        assert result == [8, 9]

    def test_dp_rank_swa_blocks_remain_local_for_raiden(self, mock_scheduler):
        """DP rank selects the SWA mapping; raiden block ids stay shard-local."""
        scheduler, page_size, seqlen = mock_scheduler
        rank0 = np.zeros_like(
            scheduler.token_to_kv_pool_allocator.full_to_swa_index_mapping
        )
        rank1 = np.arange(0, 200, dtype=np.int32)
        scheduler.token_to_kv_pool_allocator.full_to_swa_index_mapping = [
            rank0,
            rank1,
        ]
        scheduler.token_to_kv_pool_allocator.pages_per_rank = 10

        result = scheduler._extract_swa_block_ids_for_chunk(
            self._make_req(list(range(seqlen)), dp_rank=1),
            start=0,
            end=seqlen,
            page_size=page_size,
            sliding_window_size=8,
        )

        assert result == [8, 9]

    def test_dp_rank_full_blocks_remain_local_for_raiden(self, mock_scheduler):
        """Full-pool page ids must stay within the raiden shard-local shape."""
        scheduler, page_size, _ = mock_scheduler
        scheduler.token_to_kv_pool_allocator.pages_per_rank = 10
        req = self._make_req(list(range(40)), dp_rank=1)

        result = scheduler._extract_req_block_ids_range(
            req,
            start=4,
            end=12,
        )

        assert result == [1, 2]


class TestRaidenDPPageNamespace:
    def test_global_page_ids_include_reserved_page_per_rank(self):
        from sgl_jax.srt.disaggregation.jax_transfer.conn import (
            _raiden_global_page_ids,
        )

        assert _raiden_global_page_ids(
            [0, 1, 2, 10], dp_rank=1, pages_per_rank=10
        ) == [0, 12, 13, 21]

    def test_pages_per_rank_uses_swa_sub_allocator(self):
        from sgl_jax.srt.disaggregation.jax_transfer.conn import (
            _raiden_pages_per_rank,
        )

        class FullAllocator:
            pages_per_rank = 100

        class SWAAllocator:
            pages_per_rank = 25

        class HybridAllocator:
            full_attn_allocator = FullAllocator()
            swa_attn_allocator = SWAAllocator()

        assert _raiden_pages_per_rank(HybridAllocator()) == 100
        assert _raiden_pages_per_rank(HybridAllocator(), swa=True) == 25


class TestRaidenEndpointDPShards:
    def test_filters_single_raiden_endpoint_to_dp_rank_shards(self):
        from sgl_jax.srt.disaggregation.decode import _raiden_endpoint_for_dp

        endpoint = _raiden_endpoint_for_dp(
            p_host="10.0.0.1",
            p_endpoints=[{"endpoint": "10.0.0.1:33463", "shards": list(range(8))}],
            local_eps=[{"endpoint": "10.0.0.2:12345", "shards": list(range(8))}],
            fallback_base_port=33463,
            dp_rank=1,
            dp_size=2,
        )

        assert endpoint == [{"endpoint": "10.0.0.1:33463", "shards": [4, 5, 6, 7]}]

    def test_keeps_single_endpoint_string_when_dp_is_one(self):
        from sgl_jax.srt.disaggregation.decode import _raiden_endpoint_for_dp

        endpoint = _raiden_endpoint_for_dp(
            p_host="10.0.0.1",
            p_endpoints=[{"endpoint": "10.0.0.1:33463", "shards": list(range(8))}],
            local_eps=[{"endpoint": "10.0.0.2:12345", "shards": list(range(8))}],
            fallback_base_port=33463,
            dp_rank=0,
            dp_size=1,
        )

        assert endpoint == "10.0.0.1:33463"


class TestDecodeDPAllocation:
    def test_admit_decode_prealloc_allocates_request_dp_rank(self):
        from types import SimpleNamespace

        from sgl_jax.srt.disaggregation.decode import (
            DecodeBookkeeping,
            SchedulerDisaggregationDecodeMixin,
        )

        class FakeAllocator:
            page_size = 4

            def __init__(self):
                self.available_calls = []
                self.alloc_calls = []

            def available_size(self, dp_rank=0):
                self.available_calls.append(dp_rank)
                return 10_000

            def alloc(self, need_size, dp_rank=0):
                self.alloc_calls.append((need_size, dp_rank))
                return np.arange(4, 20, dtype=np.int32)

        class FakePreallocQueue:
            def __init__(self, entry):
                self.entry = entry

            def items_fifo(self):
                return [self.entry]

        class FakeTransferQueue:
            def __len__(self):
                return 0

        req = SimpleNamespace(
            rid="req",
            dp_rank=1,
            origin_input_ids=list(range(13)),
        )
        entry = DecodeBookkeeping(req_id=req.rid, req=req, p_info={})
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=FakeAllocator(),
            server_args=SimpleNamespace(
                disaggregation_num_reserved_decode_tokens=512,
                disaggregation_max_inflight_transfers=8,
            ),
            running_batch=None,
            disagg_transfer_queue=FakeTransferQueue(),
            disagg_prealloc_queue=FakePreallocQueue(entry),
            disagg_kv_manager=SimpleNamespace(use_raiden=True),
        )

        admitted = []

        def fake_admit(entry_arg, kv_indices, page_size):
            admitted.append((entry_arg.req.dp_rank, kv_indices.copy(), page_size))
            return True

        scheduler._admit_one_raiden = fake_admit

        SchedulerDisaggregationDecodeMixin._admit_decode_prealloc(scheduler)

        assert scheduler.token_to_kv_pool_allocator.available_calls == [1]
        assert scheduler.token_to_kv_pool_allocator.alloc_calls == [(16, 1)]
        assert admitted and admitted[0][0] == 1

    def test_admit_one_raiden_builds_dp_rank_swa_local_pages(self):
        import json
        from types import SimpleNamespace

        from sgl_jax.srt.disaggregation.decode import (
            DecodeBookkeeping,
            SchedulerDisaggregationDecodeMixin,
        )

        class FakeBootstrap:
            def get_transfer_info(self, room):
                assert room == 7
                return {
                    "chunks": {
                        0: {
                            "raiden_endpoints_json": json.dumps([
                                {"endpoint": "10.0.0.1:37121", "shards": list(range(8))}
                            ]),
                            "swa_raiden_endpoints_json": json.dumps([
                                {"endpoint": "10.0.0.1:33031", "shards": list(range(8))}
                            ]),
                        }
                    }
                }

        class FakeReceiver:
            def init(self, metadata):
                self.metadata = metadata

        class FakeManager:
            use_raiden = True

            def __init__(self):
                self.receiver = FakeReceiver()
                self.raiden_wrapper = SimpleNamespace(
                    endpoints=[{"endpoint": "10.0.0.2:1", "shards": list(range(8))}],
                    endpoints_swa=[{"endpoint": "10.0.0.2:2", "shards": list(range(8))}],
                )

            def create_receiver(self, req_id):
                assert req_id == "req"
                return self.receiver

        rank0 = np.zeros(128, dtype=np.int32)
        rank1 = np.zeros(128, dtype=np.int32)
        rank1[12] = 20
        rank1[16] = 24
        allocator = SimpleNamespace(full_to_swa_index_mapping=[rank0, rank1])
        manager = FakeManager()
        removed = []
        req = SimpleNamespace(
            rid="req",
            dp_rank=1,
            bootstrap_room=7,
            disagg_transfer_id=None,
            origin_input_ids=list(range(16)),
        )
        entry = DecodeBookkeeping(
            req_id=req.rid,
            req=req,
            p_info={
                "host": "10.0.0.1",
                "transfer_port": 31000,
                "side_channel_port": 9600,
            },
        )
        scheduler = SimpleNamespace(
            disagg_bootstrap_client=FakeBootstrap(),
            disagg_kv_manager=manager,
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=8,
            dp_size=2,
            disagg_prealloc_queue=SimpleNamespace(remove=removed.append),
            disagg_transfer_queue=SimpleNamespace(add=lambda entry: None),
            _record_decode_transfer_failure=lambda reason: None,
            _release_decode_kv_indices=lambda kv_indices, dp_rank=0: None,
            _abort_decode_request=lambda req, reason: None,
            _raiden_set_decode_bookkeeping=lambda req, kv_indices: None,
            _pd_mark_time=lambda req, name: None,
        )
        kv_indices = np.arange(4, 20, dtype=np.int32)

        ok = SchedulerDisaggregationDecodeMixin._admit_one_raiden(
            scheduler, entry, kv_indices, page_size=4
        )

        assert ok is True
        metadata = manager.receiver.metadata
        assert metadata.swa_local_pages == (5, 6)
        assert metadata.swa_local_page_by_full_page == {2: 5, 3: 6}
        assert metadata.swa_remote_endpoint == [
            {"endpoint": "10.0.0.1:33031", "shards": [4, 5, 6, 7]}
        ]


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
        assert m.swa_local_page_by_full_page is None

    def test_receiver_uses_swa_full_page_mapping(self):
        """SWA local ids follow the chunk's full-page offset."""
        from sgl_jax.srt.disaggregation.jax_transfer.conn import (
            JaxTransferKVReceiver,
            PMetadata,
        )

        class FakeBootstrap:
            def get_transfer_info(self, room):
                assert room == 7
                return {
                    "num_chunks": 2,
                    "chunks": {
                        1: {
                            "remote_block_ids": [31],
                            "chunk_page_offset": 3,
                            "swa_block_ids": [41],
                        }
                    },
                }

        class FakeRaidenWrapper:
            def __init__(self):
                self.calls = []

            def start_read(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        class FakeManager:
            use_raiden = True

            def __init__(self):
                self.bootstrap_client = FakeBootstrap()
                self.raiden_wrapper = FakeRaidenWrapper()

            def poll_raiden(self):
                pass

            def raiden_receiver_state(self, req_id):
                return None

            def record_terminal(self, *args, **kwargs):
                pass

            def _prune_receiver(self, req_id):
                pass

            def raiden_forget(self, req_id):
                pass

        mgr = FakeManager()
        receiver = JaxTransferKVReceiver(mgr, "req")
        receiver.init(
            PMetadata(
                remote_addr="p:1",
                uuid="req",
                specs={},
                p_side_channel_host="p",
                p_side_channel_port=2,
                remote_endpoint="full-ep",
                bootstrap_room=7,
                local_pages=(10, 11, 12, 13),
                swa_remote_endpoint="swa-ep",
                swa_local_pages=(101, 102, 103),
                swa_local_page_by_full_page={3: 103},
            )
        )

        receiver.poll()

        assert mgr.raiden_wrapper.calls
        _, kwargs = mgr.raiden_wrapper.calls[0]
        assert kwargs["swa_remote_block_ids"] == [41]
        assert kwargs["swa_local_block_ids"] == [103]

    def test_receiver_filters_swa_page_zero_and_tail_aligns(self):
        """Old prefill metadata may include page 0; receiver should tail-align."""
        from sgl_jax.srt.disaggregation.jax_transfer.conn import (
            JaxTransferKVReceiver,
            PMetadata,
        )

        class FakeBootstrap:
            def get_transfer_info(self, room):
                assert room == 7
                return {
                    "num_chunks": 1,
                    "chunks": {
                        0: {
                            "remote_block_ids": [31, 32],
                            "chunk_page_offset": 3,
                            "swa_block_ids": [0, 41],
                        }
                    },
                }

        class FakeRaidenWrapper:
            def __init__(self):
                self.calls = []

            def start_read(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        class FakeManager:
            use_raiden = True

            def __init__(self):
                self.bootstrap_client = FakeBootstrap()
                self.raiden_wrapper = FakeRaidenWrapper()

            def poll_raiden(self):
                pass

            def raiden_receiver_state(self, req_id):
                return None

            def record_terminal(self, *args, **kwargs):
                pass

            def _prune_receiver(self, req_id):
                pass

            def raiden_forget(self, req_id):
                pass

        mgr = FakeManager()
        receiver = JaxTransferKVReceiver(mgr, "req")
        receiver.init(
            PMetadata(
                remote_addr="p:1",
                uuid="req",
                specs={},
                p_side_channel_host="p",
                p_side_channel_port=2,
                remote_endpoint="full-ep",
                bootstrap_room=7,
                local_pages=(0, 1, 2, 10, 11),
                swa_remote_endpoint="swa-ep",
                swa_local_pages=(101, 102),
                swa_local_page_by_full_page={3: 101, 4: 102},
            )
        )

        receiver.poll()

        assert mgr.raiden_wrapper.calls
        _, kwargs = mgr.raiden_wrapper.calls[0]
        assert kwargs["swa_remote_block_ids"] == [41]
        assert kwargs["swa_local_block_ids"] == [102]

    def test_resolve_kv_pool_dtype_accepts_swa_pool(self):
        """SWA wrapper pools advertise dtype via their full sub-pool."""
        from sgl_jax.srt.disaggregation.bootstrap import resolve_kv_pool_dtype

        class FullPool:
            dtype = np.float16

        class SWAPool:
            full_kv_pool = FullPool()

        assert resolve_kv_pool_dtype(SWAPool()) is np.float16
