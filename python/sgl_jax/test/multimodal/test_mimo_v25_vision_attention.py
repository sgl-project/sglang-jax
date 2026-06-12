"""CPU unit tests for the V-4 jit-safe MiMo-V2.5 vision attention (design §5.3.3).

The pre-V-4 form `.tolist()`-ed cu_seqlens to the host and ran a Python loop of per-image
variable-length attention -- a host readback + data-dependent trip count that cannot be jitted.
V-4 replaced it with a single batched attention masked from cu_seqlens via searchsorted (all jnp).
These tests fix the previously-pod-only validation as入库 CPU coverage:

  1. cross-segment independence: a segment-A query must NOT attend to any segment-B key (the
     block-diagonal mask), so perturbing segment-B's input leaves segment-A's output bit-identical.
  2. jit-safety: the call compiles with TRACED cu_seqlens (the old .tolist() form could not) and
     matches eager.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def requires_jax(test_class):
    if not HAS_JAX:
        return unittest.skip("JAX/Flax not available")(test_class)
    return test_class


@requires_jax
class TestMiMoVisionAttentionV4(unittest.TestCase):
    HIDDEN = 32
    HEADS = 4
    HEAD_DIM = 8

    def _build(self):
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionAttention

        return MiMoVisionAttention(
            hidden_size=self.HIDDEN,
            num_heads=self.HEADS,
            num_kv_heads=self.HEADS,
            head_dim=self.HEAD_DIM,
            use_sink=True,
            window_size=-1,  # no windowing -> pure block-diagonal segment mask
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )

    def _identity_pos_emb(self, seq):
        # cos=1, sin=0 -> apply_rotary_pos_emb_vision is identity (q*1 + rotate_half(q)*0 = q).
        half = self.HEAD_DIM // 2
        return jnp.ones((seq, half), jnp.float32), jnp.zeros((seq, half), jnp.float32)

    def test_cross_segment_independence(self):
        attn = self._build()
        seq = 9
        cu = jnp.array([0, 4, 9], dtype=jnp.int32)  # two images: rows [0:4] and [4:9]
        cos, sin = self._identity_pos_emb(seq)
        hs = jax.random.normal(jax.random.PRNGKey(0), (seq, self.HIDDEN), jnp.float32)
        out1 = attn(hs, cu, (cos, sin), full_attn=False)
        # Perturb ONLY segment B (rows 4:9).
        hs2 = hs.at[4:].set(jax.random.normal(jax.random.PRNGKey(1), (5, self.HIDDEN), jnp.float32))
        out2 = attn(hs2, cu, (cos, sin), full_attn=False)
        # Segment A (rows 0:4) is bit-identical: cross-image keys are masked to finfo.min ->
        # softmax weight exactly 0, so segment B never leaks into segment A.
        self.assertTrue(bool(jnp.array_equal(out1[:4], out2[:4])))
        # Segment B's own output did change (sanity: the perturbation was real).
        self.assertFalse(bool(jnp.allclose(out1[4:], out2[4:])))

    def test_jit_with_traced_cu_seqlens(self):
        attn = self._build()
        seq = 9
        cu = jnp.array([0, 4, 9], dtype=jnp.int32)
        cos, sin = self._identity_pos_emb(seq)
        hs = jax.random.normal(jax.random.PRNGKey(2), (seq, self.HIDDEN), jnp.float32)
        eager = attn(hs, cu, (cos, sin), full_attn=False)

        @nnx.jit
        def run(m, h, c, cs, sn):
            return m(h, c, (cs, sn), full_attn=False)

        jitted = run(attn, hs, cu, cos, sin)  # cu is a TRACED array (no host .tolist())
        self.assertTrue(bool(jnp.allclose(eager, jitted, atol=1e-5)))


if __name__ == "__main__":
    unittest.main()
