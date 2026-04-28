"""Unit tests for HybridLinearKVPool — KV pool wrapper that translates
global layer_id -> compacted physical index for hybrid linear-recurrent
models (e.g. Kimi-Linear KDA + MLA hybrid).

Run:
  cd python && USE_DEVICE_TYPE=cpu python -m pytest \
    sgl_jax/test/mem_cache/test_hybrid_linear_kv_pool.py -v
"""

import os
import unittest
from unittest import mock

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sgl_jax.test.test_utils import CustomTestCase


def _make_mesh():
    devices = np.array(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, axis_names=("data", "tensor"))


# Kimi-Linear-like layout: 16 total layers, 4 MLA at positions [1, 5, 9, 13];
# the other 12 are KDA layers and write RecurrentStatePool, not the KV pool.
NUM_HIDDEN_LAYERS = 16
FULL_ATTN_LAYER_IDS = [1, 5, 9, 13]


def _make_mha_hybrid_pool(mesh):
    return HybridLinearKVPool(
        size=64,
        page_size=1,
        dtype=jnp.bfloat16,
        full_attention_layer_ids=FULL_ATTN_LAYER_IDS,
        mesh=mesh,
        token_to_kv_pool_class=MHATokenToKVPool,
        head_num=2,
        head_dim=128,
    )


class TestHybridLinearKVPoolConstruction(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.pool = _make_mha_hybrid_pool(self.mesh)

    def test_inner_pool_sized_to_full_attention_layer_count(self):
        """Inner pool layer_num is L_full (4), not num_hidden_layers (16)."""
        self.assertIsInstance(self.pool.full_kv_pool, MHATokenToKVPool)
        self.assertEqual(self.pool.full_kv_pool.layer_num, len(FULL_ATTN_LAYER_IDS))

    def test_layer_id_mapping(self):
        """global -> physical mapping is dense from 0."""
        self.assertEqual(
            self.pool.full_attention_layer_id_mapping,
            {1: 0, 5: 1, 9: 2, 13: 3},
        )


class TestHybridLinearKVPoolAccessorTranslation(CustomTestCase):
    """Verify global->physical translation through wrapper public API.

    Tests do NOT couple to inner-pool internal state (e.g. `kv_buffer` field
    layout). Where the pallas-based per-token kernel (`set_kv_buffer`) is
    needed, we mock the inner pool's entry-point method to keep the test
    CPU-runnable while still exercising the wrapper's dispatch contract.
    """

    HEAD_NUM = 2
    HEAD_DIM = 128

    def setUp(self):
        self.mesh = _make_mesh()
        self.pool = _make_mha_hybrid_pool(self.mesh)

    def test_get_fused_kv_buffer_round_trip_via_replace_and_get(self):
        """replace_buffer compacted [A, B, C, D] then get_fused_kv_buffer at
        each global layer_id returns the right buffer. Public API only —
        proves global->physical translation in the GET path without coupling
        to inner-pool field layout.
        """
        # Discover layer-buffer shape via wrapper public API.
        sample_buf = self.pool.get_fused_kv_buffer(FULL_ATTN_LAYER_IDS[0])
        shape = sample_buf.shape
        sentinels = [11.0, 22.0, 33.0, 44.0]
        bufs = [jnp.ones(shape, dtype=jnp.bfloat16) * s for s in sentinels]

        self.pool.replace_buffer(bufs)

        for sentinel, layer_id in zip(sentinels, FULL_ATTN_LAYER_IDS):
            buf = self.pool.get_fused_kv_buffer(layer_id)
            self.assertEqual(
                float(jnp.max(jnp.abs(buf))),
                sentinel,
                msg=(
                    f"global layer {layer_id} read back wrong sentinel "
                    "(off-by-one in get path's _to_physical mapping)"
                ),
            )

    def test_get_fused_kv_buffer_rejects_kda_layer(self):
        """global layer_id 0 is a KDA layer; get must raise."""
        with self.assertRaises(ValueError) as ctx:
            self.pool.get_fused_kv_buffer(0)
        self.assertIn("not a full-attention layer", str(ctx.exception))
        self.assertIn("[1, 5, 9, 13]", str(ctx.exception))

    def test_set_kv_buffer_translates_layer_id_no_cross_write(self):
        """Verify wrapper's set_kv_buffer dispatches inner.set_kv_buffer with
        the right physical layer_id for each global layer in
        full_attention_layer_ids.

        The actual per-token write is mocked because the inner pool's
        set_kv_buffer kernel is pallas-based (TPU-only; CPU raises
        "Only interpret mode is supported"). The mock captures the physical
        layer_id passed to the inner call — any off-by-one in the
        global->physical mapping or cross-write would surface either a wrong
        or duplicated layer_id in the captured list.

        Note: this mocks the inner pool's entry-point METHOD, not its internal
        kv_buffer state, so the test is not coupled to the inner-pool field
        schema (it would still pass if we ever swapped the inner pool's
        storage layout).
        """
        captured: list[int] = []

        def capture(layer_id, *args, **kwargs):
            captured.append(layer_id)

        with mock.patch.object(
            self.pool.full_kv_pool,
            "set_kv_buffer",
            side_effect=capture,
        ):
            for layer_id in FULL_ATTN_LAYER_IDS:
                self.pool.set_kv_buffer(
                    layer_id=layer_id,
                    loc=jnp.array([0], dtype=jnp.int32),
                    cache_k=jnp.zeros((1, self.HEAD_NUM, self.HEAD_DIM), dtype=jnp.bfloat16),
                    cache_v=jnp.zeros((1, self.HEAD_NUM, self.HEAD_DIM), dtype=jnp.bfloat16),
                    is_decode=False,
                )

        # 4 globals -> 4 distinct, dense-from-0 physicals. An off-by-one
        # mapping would either skip a value (e.g. [0, 2, 1, 3]) or repeat
        # one (e.g. [0, 0, 1, 2]).
        self.assertEqual(captured, [0, 1, 2, 3])


class TestHybridLinearKVPoolReplaceBuffer(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.pool = _make_mha_hybrid_pool(self.mesh)

    def _layer_buf_shape(self):
        # Shape via wrapper public API (no inner-pool field access).
        return self.pool.get_fused_kv_buffer(FULL_ATTN_LAYER_IDS[0]).shape

    def test_replace_buffer_round_trip_compacted_list(self):
        """A compacted list of len 4 lands in physical slots [0..3]; reading
        back via global layer_id returns the right buffer per layer."""
        shape = self._layer_buf_shape()
        new_bufs = [jnp.ones(shape, dtype=jnp.bfloat16) * (i + 1) for i in range(4)]

        self.pool.replace_buffer(new_bufs)

        for i, global_id in enumerate(FULL_ATTN_LAYER_IDS):
            buf = self.pool.get_fused_kv_buffer(global_id)
            self.assertEqual(
                float(jnp.max(jnp.abs(buf))),
                float(i + 1),
                msg=f"global layer {global_id} got wrong buffer",
            )

    def test_replace_buffer_rejects_full_length_list(self):
        """Passing 16 buffers (full-length) must raise — KDA outputs must NOT
        appear in the compacted layers_kv_fused emit."""
        shape = self._layer_buf_shape()
        bufs = [jnp.zeros(shape, dtype=jnp.bfloat16) for _ in range(NUM_HIDDEN_LAYERS)]
        with self.assertRaises(ValueError) as ctx:
            self.pool.replace_buffer(bufs)
        self.assertIn("compacted list of length 4", str(ctx.exception))

    def test_replace_buffer_rejects_short_list(self):
        """Length < L_full also raises."""
        shape = self._layer_buf_shape()
        bufs = [jnp.zeros(shape, dtype=jnp.bfloat16) for _ in range(2)]
        with self.assertRaises(ValueError):
            self.pool.replace_buffer(bufs)


class TestHybridLinearKVPoolPytree(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.pool = _make_mha_hybrid_pool(self.mesh)

    def test_tree_flatten_unflatten_roundtrip_preserves_observable_behaviour(self):
        """jax.tree.unflatten(treedef, leaves) reconstructs an equivalent pool.

        Verify via PUBLIC behaviour rather than poking at private attrs:
        - get_fused_kv_buffer translation still routes correctly,
        - KDA-layer rejection still raises,
        - replace_buffer length contract preserved.

        Inner pool kv_buffer MUST be a pytree leaf (not aux_data) — otherwise
        PR #966's MemoryPools `donate_argnames=["memory_pools"]` JIT donate
        path would fail to identify the buffers as donatable.
        """
        # Seed the original pool with distinct values so the round-trip is
        # not trivially passed by zero-init buffers.
        shape = self.pool.get_fused_kv_buffer(FULL_ATTN_LAYER_IDS[0]).shape
        seeded = [jnp.ones(shape, dtype=jnp.bfloat16) * (i + 1) for i in range(4)]
        self.pool.replace_buffer(seeded)

        leaves, treedef = jax.tree.flatten(self.pool)
        rebuilt = jax.tree.unflatten(treedef, leaves)

        for i, global_id in enumerate(FULL_ATTN_LAYER_IDS):
            buf = rebuilt.get_fused_kv_buffer(global_id)
            self.assertEqual(float(jnp.max(jnp.abs(buf))), float(i + 1))

        with self.assertRaises(ValueError):
            rebuilt.get_fused_kv_buffer(0)

        with self.assertRaises(ValueError):
            rebuilt.replace_buffer([jnp.zeros(shape, dtype=jnp.bfloat16)])


def _make_mla_hybrid_pool(mesh):
    return HybridLinearKVPool(
        size=64,
        page_size=1,
        dtype=jnp.bfloat16,
        full_attention_layer_ids=FULL_ATTN_LAYER_IDS,
        mesh=mesh,
        token_to_kv_pool_class=MLATokenToKVPool,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


class TestHybridLinearKVPoolMLA(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.pool = _make_mla_hybrid_pool(self.mesh)

    def test_inner_pool_is_mla_sized_to_full_layer_count(self):
        self.assertIsInstance(self.pool.full_kv_pool, MLATokenToKVPool)
        self.assertEqual(self.pool.full_kv_pool.layer_num, len(FULL_ATTN_LAYER_IDS))

    def test_get_fused_kv_buffer_translation_works_with_mla(self):
        """replace_buffer + get_fused_kv_buffer round-trip via wrapper public
        API; verifies global->physical translation works with the MLA inner
        pool variant (different per-layer buffer shape than MHA)."""
        shape = self.pool.get_fused_kv_buffer(FULL_ATTN_LAYER_IDS[0]).shape
        sentinels = [5.0, 6.0, 7.0, 8.0]
        bufs = [jnp.ones(shape, dtype=jnp.bfloat16) * s for s in sentinels]
        self.pool.replace_buffer(bufs)

        for sentinel, layer_id in zip(sentinels, FULL_ATTN_LAYER_IDS):
            buf = self.pool.get_fused_kv_buffer(layer_id)
            self.assertEqual(float(jnp.max(jnp.abs(buf))), sentinel)


if __name__ == "__main__":
    unittest.main()
