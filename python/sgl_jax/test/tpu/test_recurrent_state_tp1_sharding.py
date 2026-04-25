"""Phase 5 TPU TP=1 真测: D2 sharding fix verification on real TPU NamedSharding.

Run on tpu-v6e-1-lattn-1775630353 via sky exec (see plan Task 5 runbook).
Skipped on non-TPU runtimes; runs only when jax.devices() reports TPU.

Verifies:
1. MHATokenToKVPool.replace_buffer preserves kv_sharding on real TPU tp=1
   (per-buffer .sharding equality probe; mirrors the CPU-side Step 3.1
   contract but on real TPU NamedSharding).
2. MLATokenToKVPool.replace_buffer preserves kv_sharding on real TPU tp=1.

Note on cache-miss probing: count_pjit_cpp_cache_miss is unreliable for
this scenario because XLA cache key is shape+dtype+sharding-spec; replacing
a buffer with another that has the SAME sharding spec hits the cache
regardless of whether the fix is applied. We assert on the post-replace
.sharding equality directly (the fix's actual contract) instead.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def _is_tpu():
    return any(d.platform == "tpu" for d in jax.devices())


@unittest.skipUnless(_is_tpu(), "TPU runtime required")
class TestKVPoolReplaceBufferShardingOnTPU1(unittest.TestCase):
    def setUp(self):
        devices = np.array(jax.devices()[:1]).reshape(1, 1)
        self.mesh = Mesh(devices, axis_names=("data", "tensor"))

    def test_mha_replace_buffer_preserves_sharding(self):
        """Phase 5 D2: replace_buffer must restore kv_sharding on tp=1
        (otherwise the next JIT trace sees an unconstrained input and may
        produce inconsistent compiled code; issue #233)."""
        from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=2,
            mesh=self.mesh,
        )
        original_sharding = pool.kv_buffer[0].sharding

        # Construct new buffers via raw device_put (without kv_sharding spec)
        # to mimic JIT output whose sharding was not constrained by an
        # out_sharding annotation.
        new_buffers = [
            jax.device_put(jnp.ones_like(buf), device=jax.devices()[0]) for buf in pool.kv_buffer
        ]
        pool.replace_buffer(new_buffers)

        # Each layer's kv_buffer must end up with kv_sharding applied.
        for layer in range(pool.layer_num):
            buf = pool.kv_buffer[layer]
            self.assertEqual(
                buf.sharding,
                original_sharding,
                f"layer {layer}: replace_buffer must preserve kv_sharding "
                f"on tp_size==1; got {buf.sharding} vs expected {original_sharding}",
            )

    def test_mla_replace_buffer_preserves_sharding(self):
        from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool = MLATokenToKVPool(
            size=64,
            page_size=1,
            dtype=jnp.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            mesh=self.mesh,
        )
        original_sharding = pool.kv_buffer[0].sharding

        new_buffers = [
            jax.device_put(jnp.ones_like(buf), device=jax.devices()[0]) for buf in pool.kv_buffer
        ]
        pool.replace_buffer(new_buffers)

        for layer in range(pool.layer_num):
            buf = pool.kv_buffer[layer]
            self.assertEqual(
                buf.sharding,
                original_sharding,
                f"MLA layer {layer}: replace_buffer must preserve kv_sharding " f"on tp_size==1",
            )


if __name__ == "__main__":
    unittest.main()
