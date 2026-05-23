"""Tests for ``build_kv_cache`` dispatch.

Three branches must be covered (matches scheduler pre-refactor behavior):

- ``is_hybrid=True``                                             → ``SWARadixCache``
- ``chunked_prefill_size is not None and disable_radix_cache``   → ``ChunkCache``
- otherwise                                                      → ``RadixCache``

The real cache classes touch JAX devices on construction, so we patch
them out — this test guards the dispatch logic and argument forwarding,
not the cache implementations.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _server_args(disable_radix=False, chunked_prefill=None, max_seq_len=1024):
    sa = MagicMock()
    sa.disable_radix_cache = disable_radix
    sa.chunked_prefill_size = chunked_prefill
    sa.max_seq_len = max_seq_len
    return sa


def _model_config(num_kv_heads=4, head_dim=128, num_layers=2):
    mc = MagicMock()
    mc.get_num_kv_heads.return_value = num_kv_heads
    mc.head_dim = head_dim
    mc.num_hidden_layers = num_layers
    return mc


def _default_kwargs(**overrides):
    kwargs = dict(
        req_to_token_pool=MagicMock(name="req_to_token_pool"),
        token_to_kv_pool_allocator=MagicMock(name="alloc"),
        server_args=_server_args(),
        model_config=_model_config(),
        page_size=64,
        tp_size=1,
        is_hybrid=False,
        sliding_window_size=None,
        spec_algorithm=None,
    )
    kwargs.update(overrides)
    return kwargs


@patch("sgl_jax.srt.mem_cache.kv_cache_builder.SWARadixCache")
@patch("sgl_jax.srt.mem_cache.kv_cache_builder.ChunkCache")
@patch("sgl_jax.srt.mem_cache.kv_cache_builder.RadixCache")
class TestKVCacheBuilderDispatch(unittest.TestCase):
    def test_hybrid_dispatches_to_swa_radix_cache(self, MockRadix, MockChunk, MockSWA):
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        result = build_kv_cache(
            **_default_kwargs(
                server_args=_server_args(disable_radix=True, chunked_prefill=2048),
                is_hybrid=True,
                sliding_window_size=256,
            )
        )

        MockSWA.assert_called_once()
        MockChunk.assert_not_called()
        MockRadix.assert_not_called()
        self.assertIs(result, MockSWA.return_value)

        call_kwargs = MockSWA.call_args.kwargs
        self.assertEqual(call_kwargs["sliding_window_size"], 256)
        self.assertEqual(call_kwargs["page_size"], 64)
        self.assertTrue(call_kwargs["disable"])

    def test_chunked_prefill_with_disable_radix_dispatches_to_chunk_cache(
        self, MockRadix, MockChunk, MockSWA
    ):
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        result = build_kv_cache(
            **_default_kwargs(
                server_args=_server_args(disable_radix=True, chunked_prefill=2048),
            )
        )

        MockChunk.assert_called_once()
        MockSWA.assert_not_called()
        MockRadix.assert_not_called()
        self.assertIs(result, MockChunk.return_value)

        call_kwargs = MockChunk.call_args.kwargs
        self.assertEqual(call_kwargs["page_size"], 64)

    def test_default_dispatches_to_radix_cache(self, MockRadix, MockChunk, MockSWA):
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        spec = MagicMock()
        spec.is_eagle.return_value = False

        result = build_kv_cache(**_default_kwargs(spec_algorithm=spec))

        MockRadix.assert_called_once()
        MockSWA.assert_not_called()
        MockChunk.assert_not_called()
        self.assertIs(result, MockRadix.return_value)

        call_kwargs = MockRadix.call_args.kwargs
        self.assertEqual(call_kwargs["page_size"], 64)
        self.assertEqual(call_kwargs["max_seq_len"], 1024)
        self.assertEqual(call_kwargs["kv_head_num"], 4)
        self.assertEqual(call_kwargs["head_dim"], 128)
        self.assertEqual(call_kwargs["layer_num"], 2)
        self.assertFalse(call_kwargs["is_eagle"])

    def test_chunked_prefill_without_disable_radix_still_uses_radix(
        self, MockRadix, MockChunk, MockSWA
    ):
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        build_kv_cache(
            **_default_kwargs(
                server_args=_server_args(disable_radix=False, chunked_prefill=2048),
            )
        )

        MockRadix.assert_called_once()
        MockChunk.assert_not_called()
        MockSWA.assert_not_called()

    def test_eagle_spec_flag_propagates_to_radix_cache(self, MockRadix, MockChunk, MockSWA):
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        spec = MagicMock()
        spec.is_eagle.return_value = True

        build_kv_cache(**_default_kwargs(spec_algorithm=spec))

        self.assertTrue(MockRadix.call_args.kwargs["is_eagle"])

    def test_hybrid_takes_precedence_over_chunked_prefill(
        self, MockRadix, MockChunk, MockSWA
    ):
        """If both is_hybrid and (chunked_prefill + disable_radix) are set,
        is_hybrid wins (matches scheduler if-elif order)."""
        from sgl_jax.srt.mem_cache.kv_cache_builder import build_kv_cache

        build_kv_cache(
            **_default_kwargs(
                server_args=_server_args(disable_radix=True, chunked_prefill=2048),
                is_hybrid=True,
                sliding_window_size=128,
            )
        )

        MockSWA.assert_called_once()
        MockChunk.assert_not_called()
        MockRadix.assert_not_called()


if __name__ == "__main__":
    unittest.main()
