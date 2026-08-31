import unittest
from unittest.mock import MagicMock

from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_server_args(**overrides):
    args = MagicMock()
    args.disable_radix_cache = False
    args.chunked_prefill_size = None
    args.max_seq_len = 4096
    args.enable_unified_radix_tree = False
    args.hicache_storage = "disable"
    args.pd_disaggregation = ""
    args.disaggregation_mode = "null"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _make_model_config():
    config = MagicMock()
    config.get_num_kv_heads.return_value = 8
    config.head_dim = 128
    config.num_hidden_layers = 32
    return config


class TestBuildKVCache(unittest.TestCase):
    def test_default_returns_radix_cache(self):
        cache = build_kv_cache(
            server_args=_make_server_args(),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=False,
            sliding_window_size=None,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache

        self.assertIsInstance(cache, RadixCache)

    def test_enable_unified_radix_tree_returns_unified_radix_cache(self):
        cache = build_kv_cache(
            server_args=_make_server_args(enable_unified_radix_tree=True),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=False,
            sliding_window_size=None,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertIsInstance(cache, UnifiedRadixCache)

    def test_disable_radix_no_chunked_prefill_returns_disabled_radix(self):
        cache = build_kv_cache(
            server_args=_make_server_args(disable_radix_cache=True),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=False,
            sliding_window_size=None,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache

        self.assertIsInstance(cache, RadixCache)
        self.assertTrue(cache.disable)

    def test_disable_radix_with_chunked_prefill_returns_chunk_cache(self):
        cache = build_kv_cache(
            server_args=_make_server_args(disable_radix_cache=True, chunked_prefill_size=8192),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=False,
            sliding_window_size=None,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache

        self.assertIsInstance(cache, ChunkCache)

    def test_hybrid_unified_returns_full_and_swa_unified_radix_cache(self):
        from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

        mock_allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        cache = build_kv_cache(
            server_args=_make_server_args(enable_unified_radix_tree=True),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            is_hybrid=True,
            sliding_window_size=4096,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertIsInstance(cache, UnifiedRadixCache)
        self.assertEqual(cache.tree_components, (ComponentType.FULL, ComponentType.SWA))

    def test_hybrid_unified_accepts_spec_algorithm_none(self):
        from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = build_kv_cache(
            server_args=_make_server_args(enable_unified_radix_tree=True),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(spec=SWATokenToKVPoolAllocator),
            page_size=1,
            is_hybrid=True,
            sliding_window_size=4096,
            tp_size=1,
            spec_algorithm=None,
        )

        self.assertIsInstance(cache, UnifiedRadixCache)

    def test_explicit_unified_hybrid_swa_recurrent_fails_fast(self):
        from sgl_jax.srt.mem_cache import kv_cache_builder
        from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

        allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        req_pool = MagicMock()
        with (
            self.assertRaisesRegex(
                ValueError, r"--enable-unified-radix-tree.*FULL\+SWA\+RECURRENT"
            ),
            unittest.mock.patch.object(kv_cache_builder, "create_tree_cache") as create_tree,
            unittest.mock.patch.object(kv_cache_builder, "init_hicache") as init_hicache,
        ):
            build_kv_cache(
                server_args=_make_server_args(enable_unified_radix_tree=True),
                model_config=_make_model_config(),
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                is_hybrid=True,
                is_hybrid_recurrent=True,
                sliding_window_size=4096,
                tp_size=1,
                spec_algorithm=None,
            )

        create_tree.assert_not_called()
        init_hicache.assert_not_called()
        self.assertEqual(allocator.mock_calls, [])
        self.assertEqual(req_pool.mock_calls, [])

    def test_unified_hybrid_rejects_unsupported_combinations_before_cache_creation(self):
        from sgl_jax.srt.mem_cache import kv_cache_builder
        from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

        unsupported = (
            ("HiCache", {"hicache_storage": "none"}, None, "--hicache-storage"),
            (
                "speculative decoding",
                {},
                SpeculativeAlgorithm.EAGLE,
                "--speculative-algorithm",
            ),
            ("PD", {"pd_disaggregation": "pathways"}, None, "--pd-disaggregation"),
            (
                "disaggregation mode",
                {"disaggregation_mode": "decode"},
                None,
                "--disaggregation-mode",
            ),
        )
        for name, server_overrides, spec_algorithm, conflict_flag in unsupported:
            with self.subTest(name=name):
                allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
                req_pool = MagicMock()
                with (
                    self.assertRaisesRegex(
                        ValueError,
                        rf"--enable-unified-radix-tree.*{conflict_flag}",
                    ),
                    unittest.mock.patch.object(
                        kv_cache_builder, "create_tree_cache"
                    ) as create_tree,
                    unittest.mock.patch.object(kv_cache_builder, "init_hicache") as init_hicache,
                ):
                    build_kv_cache(
                        server_args=_make_server_args(
                            enable_unified_radix_tree=True, **server_overrides
                        ),
                        model_config=_make_model_config(),
                        req_to_token_pool=req_pool,
                        token_to_kv_pool_allocator=allocator,
                        page_size=1,
                        is_hybrid=True,
                        sliding_window_size=4096,
                        tp_size=1,
                        spec_algorithm=spec_algorithm,
                    )

                create_tree.assert_not_called()
                init_hicache.assert_not_called()
                self.assertEqual(allocator.mock_calls, [])
                self.assertEqual(req_pool.mock_calls, [])

    def test_hybrid_disable_radix_returns_swa_chunk_cache(self):
        cache = build_kv_cache(
            server_args=_make_server_args(disable_radix_cache=True),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=True,
            sliding_window_size=4096,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.chunk_cache import SWAChunkCache

        self.assertIsInstance(cache, SWAChunkCache)

    def test_eagle_spec_algorithm(self):
        spec = MagicMock()
        spec.is_eagle.return_value = True
        cache = build_kv_cache(
            server_args=_make_server_args(),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            page_size=1,
            is_hybrid=False,
            sliding_window_size=None,
            tp_size=1,
            spec_algorithm=spec,
        )
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache

        self.assertIsInstance(cache, RadixCache)
        self.assertTrue(cache.is_eagle)


if __name__ == "__main__":
    unittest.main()
