# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_unified_radix_tree_flag.py -q

import os

# Set up multi-device simulation for tensor parallelism
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    # Set JAX to use CPU for testing with simulated devices
    os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixCache
from sgl_jax.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    default_radix_cache_factory,
)
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)

ENV_FLAG = "SGLANG_JAX_ENABLE_UNIFIED_RADIX_TREE"


class _StubModelConfig:
    """Exposes exactly what default_radix_cache_factory reads off ModelConfig."""

    head_dim = 64
    num_hidden_layers = 2

    def get_num_kv_heads(self, tp_size: int) -> int:
        return 8


def _make_server_args(**kwargs) -> ServerArgs:
    """Construct ServerArgs hermetically.

    ``__post_init__`` reads and writes process env vars (e.g. it honors
    SGLANG_JAX_ENABLE_UNIFIED_RADIX_TREE and sets
    SGLANG_ENABLE_DETERMINISTIC_SAMPLING), so build inside a patched env with
    the unified-radix override removed; tests that exercise the env override
    set it explicitly themselves.
    """
    kwargs.setdefault("model_path", "dummy-model")
    kwargs.setdefault("max_seq_len", 2048)
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV_FLAG, None)
        return ServerArgs(**kwargs)


class TestUnifiedRadixTreeFlag(CustomTestCase):
    def setUp(self):
        # Small KV pool dims: routing only constructs the caches, so the
        # actual KV buffers can stay tiny.
        self.kv_head_num = 8
        self.head_dim = 64
        self.layer_num = 2
        self.max_seq_len = 2048
        self.dtype = jnp.bfloat16
        self.pool_size = 8192

    # ------------------------------------------------------------------ #
    #  Pool helpers (fresh pools per test)                                 #
    # ------------------------------------------------------------------ #

    def _create_pools(self, dp_size: int = 1):
        req_pool = ReqToTokenPool(
            size=1024,
            max_context_len=self.max_seq_len,
            dtype=np.int32,
        )
        kv_cache = MHATokenToKVPool(
            size=self.pool_size,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
            dp_size=dp_size,
        )
        allocator = TokenToKVPoolAllocator(
            size=self.pool_size,
            kvcache=kv_cache,
            dp_size=dp_size,
        )
        return req_pool, allocator

    def _create_swa_pools(self):
        req_pool = ReqToTokenPool(
            size=64,
            max_context_len=512,
            dtype=np.int32,
        )
        kv_pool = SWAKVPool(
            size=128,
            size_swa=128,
            page_size=1,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            token_to_kv_pool_class=MHATokenToKVPool,
            dtype=self.dtype,
            # head_num must stay divisible by the tensor-axis size of the
            # module-level 8-device mesh.
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            mesh=mesh,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=128,
            size_swa=128,
            kvcache=kv_pool,
        )
        return req_pool, allocator

    def _build_ctx(
        self,
        server_args: ServerArgs,
        req_pool,
        allocator,
        *,
        is_hybrid_swa: bool = False,
        disable_radix_cache: bool = False,
        effective_chunked_prefill_size: int | None = None,
        sliding_window_size: int | None = None,
    ) -> TreeCacheBuildContext:
        params = CacheInitParams(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            sliding_window_size=sliding_window_size,
        )
        return TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid_swa,
            disable_radix_cache=disable_radix_cache,
            effective_chunked_prefill_size=effective_chunked_prefill_size,
            model_config=_StubModelConfig(),
            tp_size=1,
        )

    # ------------------------------------------------------------------ #
    #  Flag parsing                                                        #
    # ------------------------------------------------------------------ #

    def test_flag_default_off(self):
        server_args = _make_server_args()
        self.assertIs(server_args.enable_unified_radix_tree, False)

    def test_flag_cli_parse(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args_on = parser.parse_args(["--model-path", "dummy-model", "--enable-unified-radix-tree"])
        self.assertTrue(args_on.enable_unified_radix_tree)

        args_off = parser.parse_args(["--model-path", "dummy-model"])
        self.assertFalse(args_off.enable_unified_radix_tree)

        # round-trip through from_cli_args: the namespace attr lands on the field
        with mock.patch.dict(os.environ):
            os.environ.pop(ENV_FLAG, None)
            server_args = ServerArgs.from_cli_args(args_on)
            self.assertTrue(server_args.enable_unified_radix_tree)

    def test_flag_env_honor(self):
        with mock.patch.dict(os.environ, {ENV_FLAG: "1"}):
            server_args = ServerArgs(model_path="dummy-model", max_seq_len=2048)
            self.assertTrue(server_args.enable_unified_radix_tree)

        with mock.patch.dict(os.environ, {ENV_FLAG: "0"}):
            server_args = ServerArgs(model_path="dummy-model", max_seq_len=2048)
            self.assertFalse(server_args.enable_unified_radix_tree)

        # env var absent
        server_args = _make_server_args()
        self.assertFalse(server_args.enable_unified_radix_tree)

    # ------------------------------------------------------------------ #
    #  Factory routing                                                     #
    # ------------------------------------------------------------------ #

    def test_factory_flag_off_returns_radix_cache(self):
        server_args = _make_server_args()
        req_pool, allocator = self._create_pools()
        ctx = self._build_ctx(server_args, req_pool, allocator)

        cache = default_radix_cache_factory(ctx)

        self.assertIsInstance(cache, RadixCache)
        self.assertNotIsInstance(cache, UnifiedRadixCache)
        self.assertFalse(cache.disable)

    def test_factory_flag_on_returns_unified(self):
        server_args = _make_server_args(enable_unified_radix_tree=True)
        req_pool, allocator = self._create_pools()
        ctx = self._build_ctx(server_args, req_pool, allocator)

        cache = default_radix_cache_factory(ctx)

        self.assertIsInstance(cache, UnifiedRadixCache)
        self.assertEqual(cache.tree_components, (ComponentType.FULL,))
        self.assertIs(cache.disable, False)

    def test_factory_flag_on_hybrid_unaffected(self):
        server_args = _make_server_args(enable_unified_radix_tree=True)
        req_pool, allocator = self._create_swa_pools()
        ctx = self._build_ctx(
            server_args,
            req_pool,
            allocator,
            is_hybrid_swa=True,
            sliding_window_size=64,
        )

        cache = default_radix_cache_factory(ctx)

        self.assertIsInstance(cache, SWARadixCache)
        self.assertNotIsInstance(cache, UnifiedRadixCache)

    def test_factory_flag_on_hybrid_disabled_returns_swa_chunk_cache(self):
        # hybrid + disabled radix wins over the unified flag (registry checks
        # is_hybrid_swa first, and its disable branch builds SWAChunkCache).
        server_args = _make_server_args(enable_unified_radix_tree=True)
        req_pool, allocator = self._create_swa_pools()
        ctx = self._build_ctx(
            server_args,
            req_pool,
            allocator,
            is_hybrid_swa=True,
            disable_radix_cache=True,
            sliding_window_size=64,
        )

        cache = default_radix_cache_factory(ctx)

        self.assertIsInstance(cache, SWAChunkCache)
        self.assertNotIsInstance(cache, UnifiedRadixCache)

    def test_factory_flag_on_disabled_radix_unaffected(self):
        server_args = _make_server_args(enable_unified_radix_tree=True)

        # disabled + chunked prefill set -> ChunkCache
        req_pool, allocator = self._create_pools()
        ctx = self._build_ctx(
            server_args,
            req_pool,
            allocator,
            disable_radix_cache=True,
            effective_chunked_prefill_size=4096,
        )
        cache = default_radix_cache_factory(ctx)
        self.assertIsInstance(cache, ChunkCache)
        self.assertNotIsInstance(cache, UnifiedRadixCache)
        self.assertNotIsInstance(cache, RadixCache)

        # disabled + chunked prefill None -> disabled RadixCache
        req_pool, allocator = self._create_pools()
        ctx = self._build_ctx(
            server_args,
            req_pool,
            allocator,
            disable_radix_cache=True,
            effective_chunked_prefill_size=None,
        )
        cache = default_radix_cache_factory(ctx)
        self.assertIsInstance(cache, RadixCache)
        self.assertNotIsInstance(cache, UnifiedRadixCache)
        self.assertTrue(cache.disable)


if __name__ == "__main__":
    unittest.main()
