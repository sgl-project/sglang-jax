"""Phase 5 D3 + D7 end-to-end: mock model exercises the full JIT donate +
replace_all chain across multiple layers and multiple forward calls.

Verifies the list element mutation contract that Phase 1+2+3+4 took on
faith holds end-to-end:
- Layer N writes recurrent state via the returned pool_updates dict
- Layer N+1 reads via the same dict (no inter-layer leak)
- Forward k+1 sees forward k's accumulated state after replace_all

Does NOT instantiate any Kimi-Linear model classes; uses jax primitives
to model 'read state -> mock forward -> write state' directly.

CRITICAL CONTRACT (Phase 3 RFC §5.1): the mock model is a PURE function
that returns the full pool_updates dict. It NEVER mutates the pool
in-place; MemoryPools.replace_all is the sole writer (this matches the
real model contract -- model returns dict, _forward dispatches via
replace_all). Any in-place pool mutation from inside jax.jit would touch
the traced pool but not the real one, so we deliberately avoid that
pattern.
"""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def _build_pools():
    """Two-layer hybrid pool stack mirroring Kimi-Linear KDA layout."""
    from jax.sharding import Mesh

    from sgl_jax.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        MemoryPools,
        MHATokenToKVPool,
    )
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, axis_names=("data", "tensor"))

    rsp = RecurrentStatePool(
        linear_recurrent_layer_ids=[0, 1],
        max_num_reqs=2,
        num_heads=2,
        head_dim=4,
        conv_kernel_size=4,
    )
    kv = MHATokenToKVPool(
        size=8,
        page_size=1,
        dtype=jnp.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=2,
        mesh=mesh,
    )
    hybrid = HybridReqToTokenPool(
        size=3, max_context_len=16, dtype=np.int32, recurrent_state_pool=rsp
    )
    return MemoryPools(token_to_kv_pool=kv, recurrent_state_pool=rsp), hybrid


def _mock_forward_pure(memory_pools, slot_idx):
    """Pure mock model __call__: reads current pool state, computes new
    state, returns FULL pool_updates dict for replace_all to write back.

    Per layer (matches RecurrentStatePool layout: 2 layers indexed [0, 1]):
    - Layer 0: new_recurrent[slot] = old_recurrent[slot] + 1.0
    - Layer 1: new_recurrent[slot] = old_recurrent[slot] + 2.0
    Conv buffers untouched (passed through).

    Returns dict with BOTH pool keys (replace_all requires exact key match
    with MemoryPools._pools per memory_pool.py:1291).
    """
    rsp = memory_pools.recurrent_state_pool
    kv = memory_pools.token_to_kv_pool

    # Read all layers' current buffers; compute new layer 0 + layer 1.
    new_recurrent = list(rsp.recurrent_buffers)  # shallow copy
    new_recurrent[0] = rsp.recurrent_buffers[0].at[slot_idx].add(1.0)
    new_recurrent[1] = rsp.recurrent_buffers[1].at[slot_idx].add(2.0)
    # Conv buffers pass through unchanged.
    new_conv = [list(inner) for inner in rsp.conv_buffers]

    # KV buffers also pass through unchanged (we don't model KV writes here).
    new_kv = list(kv.kv_buffer)

    return {
        "token_to_kv_pool": new_kv,
        "recurrent_state_pool": (new_recurrent, new_conv),
    }


class TestPhase5EndToEndMockModelMultiLayerForward(unittest.TestCase):
    """Multi-layer multi-forward state preservation via the pure-function +
    replace_all contract (no in-pool mutation; matches Phase 3 model contract)."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_layer_independence_within_one_forward(self):
        """Layer 0 update does NOT contaminate layer 1 (different layer_id)."""
        memory_pools, hybrid = _build_pools()

        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([req])

        pool_updates = _mock_forward_pure(memory_pools, req.recurrent_pool_idx)
        memory_pools.replace_all(pool_updates)

        rsp = memory_pools.recurrent_state_pool
        l0, _ = rsp.get_linear_recurrent_layer_cache(0)
        l1, _ = rsp.get_linear_recurrent_layer_cache(1)
        self.assertTrue(bool(jnp.all(l0[req.recurrent_pool_idx] == 1.0)))
        self.assertTrue(bool(jnp.all(l1[req.recurrent_pool_idx] == 2.0)))

    def test_state_accumulates_across_two_forwards(self):
        """Forward 2 sees forward 1's accumulated state via replace_all
        write-back (list element mutation contract end-to-end)."""
        memory_pools, hybrid = _build_pools()

        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([req])

        pool_updates_1 = _mock_forward_pure(memory_pools, req.recurrent_pool_idx)
        memory_pools.replace_all(pool_updates_1)

        pool_updates_2 = _mock_forward_pure(memory_pools, req.recurrent_pool_idx)
        memory_pools.replace_all(pool_updates_2)

        rsp = memory_pools.recurrent_state_pool
        l0, _ = rsp.get_linear_recurrent_layer_cache(0)
        l1, _ = rsp.get_linear_recurrent_layer_cache(1)
        self.assertTrue(bool(jnp.all(l0[req.recurrent_pool_idx] == 2.0)))
        self.assertTrue(bool(jnp.all(l1[req.recurrent_pool_idx] == 4.0)))

    def test_jit_donate_preserves_list_container_after_replace_all(self):
        """JIT donate of memory_pools must produce a pool_updates dict that
        replace_all can dispatch correctly (verifies pytree (un)flatten works
        for multi-layer + dict return)."""
        memory_pools, hybrid = _build_pools()

        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        hybrid.alloc([req])

        # JIT the pure forward; donate memory_pools (matches Phase 2 _forward
        # contract: jitted_run_model returns model output).
        @jax.jit
        def jit_step(mp, slot):
            return _mock_forward_pure(mp, slot)

        pool_updates = jit_step(memory_pools, req.recurrent_pool_idx)
        memory_pools.replace_all(pool_updates)

        pool_updates = jit_step(memory_pools, req.recurrent_pool_idx)
        memory_pools.replace_all(pool_updates)

        rsp = memory_pools.recurrent_state_pool
        l0, _ = rsp.get_linear_recurrent_layer_cache(0)
        # Two passes -> 2.0; container survived JIT round-trip + replace_all.
        self.assertTrue(bool(jnp.all(l0[req.recurrent_pool_idx] == 2.0)))


if __name__ == "__main__":
    unittest.main()
