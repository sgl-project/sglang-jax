"""Tests for MiMo-V2-Flash attention_sink_bias correctness.

Run with:
    cd sglang-jax/python
    USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/test_mimo_attention_sink.py -v

Coverage:
  1. Native-backend reference: non-zero attention_sink shifts output.
  2. Fused kernel: raises NotImplementedError when attention_sink is passed.
  3. MiMo model: is_swa_layer() routes by hybrid_layer_pattern.
  4. MiMo model: attention_sink_bias Param is created only when configured.
"""

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.layers.attention.native_backend import forward_attention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

# Module-level mesh — mirrors the pattern in test_split_kv_attention.py
mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_native_forward(seq_len, num_heads, num_kv_heads, head_dim,
                        attention_sink=None, seed=42):
    """Run one decode step through the pure-JAX native attention backend."""
    rng = jax.random.PRNGKey(seed)
    # q: single decode token — 2D [1, num_heads * head_dim]
    q = jax.random.normal(rng, (1, num_heads * head_dim))
    # k/v cache: 3D [cache_size, num_kv_heads, head_dim] — as get_kv_buffer returns
    k_cache = jax.random.normal(jax.random.fold_in(rng, 1), (seq_len, num_kv_heads, head_dim))
    v_cache = jax.random.normal(jax.random.fold_in(rng, 2), (seq_len, num_kv_heads, head_dim))
    # loc: positions 1..seq_len (0 is reserved as "invalid" in the backend)
    loc = jnp.arange(1, seq_len + 1, dtype=jnp.int32)
    seq_lengths = jnp.array([seq_len], dtype=jnp.int32)
    extend_prefix_lens = jnp.zeros(1, dtype=jnp.int32)
    extend_seq_lens = jnp.ones(1, dtype=jnp.int32)

    return forward_attention(
        q, k_cache, v_cache,
        seq_lengths=seq_lengths,
        loc=loc,
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=1.0 / jnp.sqrt(head_dim),
        is_causal=True,
        mode=ForwardMode.DECODE,
        attention_sink=attention_sink,
    )


# ---------------------------------------------------------------------------
# 1. Native-backend reference: attention_sink shifts output
# ---------------------------------------------------------------------------

class TestNativeBackendAttentionSink(unittest.TestCase):
    """Verify the pure-JAX reference implementation applies attention_sink correctly."""

    def test_large_sink_bias_reduces_output_norm(self):
        """Very large sink bias absorbs probability mass into the zero-value virtual sink
        token, reducing the norm of the real attention output."""
        num_heads, num_kv_heads, head_dim = 4, 2, 16
        seq_len = 32
        out_no_sink = _run_native_forward(seq_len, num_heads, num_kv_heads, head_dim)
        large_bias = jnp.full((num_heads,), 1000.0)
        out_large_sink = _run_native_forward(
            seq_len, num_heads, num_kv_heads, head_dim, attention_sink=large_bias
        )
        norm_no_sink = float(jnp.linalg.norm(out_no_sink))
        norm_large_sink = float(jnp.linalg.norm(out_large_sink))
        self.assertLess(norm_large_sink, norm_no_sink,
                        "Large sink bias should reduce output norm (mass to zero-value sink)")

    def test_very_negative_sink_bias_matches_no_sink(self):
        """Very negative sink bias → exp(bias - lse) ≈ 0 → alpha ≈ 1 → output ≈ no-sink."""
        num_heads, num_kv_heads, head_dim = 4, 2, 16
        seq_len = 32
        out_no_sink = _run_native_forward(seq_len, num_heads, num_kv_heads, head_dim)
        very_neg = jnp.full((num_heads,), -1000.0)
        out_neg_sink = _run_native_forward(
            seq_len, num_heads, num_kv_heads, head_dim, attention_sink=very_neg
        )
        max_diff = float(jnp.max(jnp.abs(out_no_sink - out_neg_sink)))
        self.assertLess(max_diff, 1e-4,
                        "Very negative sink bias → output should match no-sink case")

    def test_per_head_sink_bias_accepted(self):
        """Per-head sink bias shape [num_heads] is accepted and output has correct shape."""
        num_heads, num_kv_heads, head_dim = 4, 4, 16
        sink = jnp.array([1.0, -1.0, 2.0, 0.5])
        out = _run_native_forward(16, num_heads, num_kv_heads, head_dim, attention_sink=sink)
        self.assertEqual(out.shape, (1, num_heads * head_dim))

    def test_sink_bias_changes_output(self):
        """Non-trivial sink bias should produce a different output from no-sink."""
        num_heads, num_kv_heads, head_dim = 4, 2, 16
        out_no_sink = _run_native_forward(32, num_heads, num_kv_heads, head_dim)
        out_with_sink = _run_native_forward(
            32, num_heads, num_kv_heads, head_dim,
            attention_sink=jnp.ones((num_heads,))
        )
        self.assertFalse(jnp.allclose(out_no_sink, out_with_sink),
                         "Non-zero sink bias should change the attention output")


# ---------------------------------------------------------------------------
# 2. Fused kernel raises NotImplementedError when attention_sink is passed
# ---------------------------------------------------------------------------

class TestFusedKernelRejectsAttentionSink(unittest.TestCase):
    """The fused ragged_paged_attention must fail loudly, not silently drop sink."""

    def test_raises_not_implemented(self):
        from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
            ragged_paged_attention,
        )
        dummy = jnp.zeros((1,))
        with self.assertRaises(NotImplementedError) as ctx:
            ragged_paged_attention(
                dummy, dummy, dummy,
                kv_cache_fused=dummy,
                kv_lens=dummy,
                page_indices=dummy,
                cu_q_lens=dummy,
                cu_kv_lens=dummy,
                distribution=dummy,
                custom_mask=dummy,
                attention_sink=jnp.ones((4,)),
            )
        self.assertIn("fused KV path", str(ctx.exception))
        self.assertIn("split KV", str(ctx.exception))

    def test_no_error_when_sink_is_none(self):
        """Fused path with attention_sink=None should not raise NotImplementedError."""
        from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
            ragged_paged_attention,
        )
        dummy = jnp.zeros((1,))
        try:
            ragged_paged_attention(
                dummy, dummy, dummy,
                kv_cache_fused=dummy,
                kv_lens=dummy,
                page_indices=dummy,
                cu_q_lens=dummy,
                cu_kv_lens=dummy,
                distribution=dummy,
                custom_mask=dummy,
                attention_sink=None,
            )
        except NotImplementedError:
            self.fail("attention_sink=None should not raise NotImplementedError")
        except Exception:
            pass  # shape errors from dummy inputs are expected


# ---------------------------------------------------------------------------
# 3. MiMo model: is_swa_layer() routes correctly by hybrid_layer_pattern
# ---------------------------------------------------------------------------

class TestMiMoLayerRouting(unittest.TestCase):
    """MiMoMoeDecoderLayer.is_swa_layer() must honour hybrid_layer_pattern."""

    def _make_config(self, pattern):
        return SimpleNamespace(
            hybrid_layer_pattern=pattern,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            sliding_window_size=None,
            add_swa_attention_sink_bias=False,
            add_full_attention_sink_bias=False,
            num_experts=4,
            num_experts_per_tok=2,
            intermediate_size=128,
            shared_expert_intermediate_size=0,
            moe_backend="epmoe",
            topk_method="greedy",
            scoring_func="softmax",
            norm_topk_prob=True,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
            swa_head_dim=16,
            swa_v_head_dim=None,
            swa_sliding_window_size=128,
        )

    def _swa(self, layer_id, config):
        from sgl_jax.srt.models.mimo_v2_flash import MiMoMoeDecoderLayer
        obj = SimpleNamespace(layer_id=layer_id)
        return MiMoMoeDecoderLayer.is_swa_layer(obj, config)

    def test_integer_pattern_alternating(self):
        config = self._make_config([1, 0, 1, 0, 1, 0])
        expected = [True, False, True, False, True, False]
        for i, exp in enumerate(expected):
            self.assertEqual(self._swa(i, config), exp, f"layer_id={i}")

    def test_layer_id_beyond_pattern_returns_false(self):
        config = self._make_config([1, 0])
        self.assertFalse(self._swa(5, config),
                         "Out-of-range layer_id should default to GA (False)")

    def test_mimo_v2_flash_5to1_ratio(self):
        """MiMo-V2-Flash: 8 hybrid blocks × (5 SWA + 1 GA) = 40 SWA + 8 GA."""
        pattern = [1, 1, 1, 1, 1, 0] * 8
        config = self._make_config(pattern)
        swa_count = sum(self._swa(i, config) for i in range(len(pattern)))
        self.assertEqual(swa_count, 40)
        self.assertEqual(len(pattern) - swa_count, 8)


# ---------------------------------------------------------------------------
# 4. MiMo model: attention_sink_bias Param shape and presence
# ---------------------------------------------------------------------------

class TestMiMoAttentionSinkBiasParam(unittest.TestCase):
    """attention_sink_bias Param is created only when the flag is set."""

    def _make_attn(self, attention_sink_bias: bool, num_heads: int = 4):
        from sgl_jax.srt.models.mimo_v2_flash import MiMoMoeAttention
        # Module-level jax.sharding.set_mesh(mesh) is already active.
        return MiMoMoeAttention(
            hidden_size=64,
            num_heads=num_heads,
            num_kv_heads=2,
            max_position_embeddings=512,
            mesh=mesh,
            head_dim=16,
            v_head_dim=None,
            sliding_window_size=128,
            attention_sink_bias=attention_sink_bias,
            layer_id=0,
        )

    def test_sink_bias_created_when_enabled(self):
        attn = self._make_attn(attention_sink_bias=True, num_heads=4)
        self.assertIsNotNone(attn.attention_sink_bias)
        bias_val = attn.attention_sink_bias[...]
        self.assertEqual(bias_val.shape, (4,))
        np.testing.assert_allclose(bias_val, 0.0, atol=1e-6)

    def test_sink_bias_none_when_disabled(self):
        attn = self._make_attn(attention_sink_bias=False)
        self.assertIsNone(attn.attention_sink_bias)


if __name__ == "__main__":
    unittest.main()
