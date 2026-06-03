import unittest
from unittest.mock import MagicMock

from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache


def _make_server_args(**overrides):
    args = MagicMock()
    args.disable_radix_cache = False
    args.chunked_prefill_size = None
    args.max_seq_len = 4096
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

    def test_hybrid_returns_swa_radix_cache(self):
        from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator

        mock_allocator = MagicMock(spec=SWATokenToKVPoolAllocator)
        cache = build_kv_cache(
            server_args=_make_server_args(),
            model_config=_make_model_config(),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=mock_allocator,
            page_size=1,
            is_hybrid=True,
            sliding_window_size=4096,
            tp_size=1,
            spec_algorithm=None,
        )
        from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache

        self.assertIsInstance(cache, SWARadixCache)

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
