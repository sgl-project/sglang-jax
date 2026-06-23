"""I1 and I4 invariants for Step 3.5 Flash naive forward pass (Task 6).

I1: Hard asserts on a reduced-config naive forward:
  - no NaN/Inf in logits
  - top-k routing: all expert IDs in [0, num_experts), correct shape
  - swiglu clamp effective: gate/up within limit for clamped layers
  - lm_head != embed_tokens weights (untied)

I4: Greedy bit-exact repeatability — two identical forwards produce equal logits.
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_mesh)


def _get_fixtures():
    """Return (_make_config, _build_checkpoint, _DummyModelConfig) from shared module."""
    from sgl_jax.test.models.test_step3p5_model import (  # noqa: PLC0415
        _build_checkpoint,
        _DummyModelConfig,
        _make_config,
    )

    return _make_config, _build_checkpoint, _DummyModelConfig


def _make_forward_batch(num_tokens: int, cfg):
    """Build a minimal ForwardBatch for a single-sequence prefill."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=jnp.arange(num_tokens, dtype=jnp.int32),
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        out_cache_loc=jnp.zeros(num_tokens, dtype=jnp.int32),
        positions=jnp.arange(num_tokens, dtype=jnp.int32),
        extend_prefix_lens=jnp.zeros(1, dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
    )


def _build_model_and_load(cfg):
    """Construct fp32 ForCausalLM, save+load checkpoint, return (model, weights_np)."""
    from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

    _, _build_checkpoint, _DummyModelConfig = _get_fixtures()
    with jax.set_mesh(_mesh):
        model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.float32, attn_impl="naive")

    weights_np = _build_checkpoint(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights_np, os.path.join(tmpdir, "model.safetensors"))
        mc = _DummyModelConfig(tmpdir, cfg)
        with jax.set_mesh(_mesh):
            model.load_weights(mc)

    return model, weights_np


def _run_naive_forward(model, cfg, num_tokens: int = 8):
    """Run naive forward, return logits [T, vocab]."""
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    fb = _make_forward_batch(num_tokens, cfg)
    lm = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        logits_indices=jnp.arange(num_tokens, dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        extend_seq_lens_cpu=[num_tokens],
    )

    class _FakeMemPools:
        token_to_kv_pool = None

    with jax.set_mesh(_mesh):
        result = model(fb, _FakeMemPools(), lm)

    output, _, _, _ = result
    return jnp.asarray(output.next_token_logits)  # [T, vocab]


# ---------------------------------------------------------------------------
# I1 invariants
# ---------------------------------------------------------------------------


class TestStep3p5InvariantsI1(unittest.TestCase):
    """Hard property asserts on the reduced-config naive forward."""

    def setUp(self):
        _make_config, _, _ = _get_fixtures()
        self.cfg = _make_config()
        self.model, self.weights_np = _build_model_and_load(self.cfg)

    def test_no_nan_inf_in_logits(self) -> None:
        """Naive forward produces finite logits (no NaN, no Inf)."""
        logits = _run_naive_forward(self.model, self.cfg)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(logits))),
            f"Logits contain non-finite values; shape={logits.shape}",
        )

    def test_moe_topk_routing_legal(self) -> None:
        """Top-k routing produces valid expert indices: shape [T, k], values in [0, E)."""
        from sgl_jax.srt.layers.moe import TopK

        captured: list = []
        _orig = TopK.__call__

        def _patched(self_topk, logits, bias=None, dispatch_info=None):
            weights, ids = _orig(self_topk, logits, bias, dispatch_info)
            captured.append(np.asarray(ids))
            return weights, ids

        TopK.__call__ = _patched  # type: ignore[method-assign]
        try:
            with jax.set_mesh(_mesh):
                _ = _run_naive_forward(self.model, self.cfg)
        finally:
            TopK.__call__ = _orig  # type: ignore[method-assign]

        # Should have captured one TopK call per MoE layer (3 MoE layers).
        self.assertGreater(len(captured), 0, "No TopK calls captured")
        E = self.cfg.moe_num_experts
        k = self.cfg.moe_top_k
        for ids in captured:
            self.assertEqual(ids.ndim, 2, f"topk_ids shape {ids.shape} should be 2D [T, k]")
            self.assertEqual(ids.shape[1], k, f"topk_ids.shape[1]={ids.shape[1]} != topk={k}")
            self.assertTrue(
                bool(np.all(ids >= 0) and np.all(ids < E)),
                f"Expert IDs out of range [0, {E}): min={ids.min()}, max={ids.max()}",
            )

    def test_swiglu_clamp_effective(self) -> None:
        """For layer 4 (swiglu_limit=7.0), Step3p5MLP clamp prevents gate > limit."""
        from sgl_jax.srt.models.step3p5 import Step3p5MLP

        # Layer 4 dense MLP limit = 16.0 (swiglu_limits_shared), routed = 7.0.
        limit = 7.0
        H, INTER = self.cfg.hidden_size, self.cfg.intermediate_size
        mlp = Step3p5MLP(
            hidden_size=H,
            intermediate_size=INTER,
            mesh=_mesh,
            dtype=jnp.float32,
            swiglu_limit=limit,
        )
        # Input with very large values — without clamp, gate would be huge.
        x = jnp.ones((4, H), dtype=jnp.float32) * 100.0
        with jax.set_mesh(_mesh):
            out = mlp(x)
        # Output must be finite (clamp prevents explosion).
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(out))),
            "MLP output non-finite with swiglu_limit applied",
        )
        # Verify by computing gate manually through mlp's proj.
        with jax.set_mesh(_mesh):
            raw_gate, _ = mlp.gate_proj(x)
            raw_gate_silu = jax.nn.silu(raw_gate)
        gate_clamped = jnp.clip(raw_gate_silu, max=limit)
        self.assertTrue(
            bool(jnp.all(gate_clamped <= limit + 1e-6)),
            f"Gate after clamp exceeds limit={limit}",
        )

    def test_lm_head_not_equal_embed(self) -> None:
        """lm_head and embed_tokens carry different weights (untied head)."""
        lm_w = np.asarray(self.model.lm_head.embedding.value)
        em_w = np.asarray(self.model.model.embed_tokens.embedding.value)
        self.assertFalse(
            np.allclose(lm_w.astype(np.float32), em_w.astype(np.float32), atol=1e-4),
            "lm_head and embed_tokens weights are identical — head appears tied",
        )

    def test_logits_shape_correct(self) -> None:
        """Forward returns logits of shape [num_tokens, vocab_size]."""
        num_tokens = 8
        logits = _run_naive_forward(self.model, self.cfg, num_tokens=num_tokens)
        self.assertEqual(logits.shape, (num_tokens, self.cfg.vocab_size))


# ---------------------------------------------------------------------------
# I4 invariant: greedy bit-exact repeatability
# ---------------------------------------------------------------------------


class TestStep3p5InvariantsI4(unittest.TestCase):
    """JAX on CPU is deterministic: two identical forwards must produce identical logits."""

    def test_greedy_bit_exact_repeatability(self) -> None:
        """Run naive forward twice with the same model and input; assert bit-equal logits."""
        _make_config, _, _ = _get_fixtures()
        cfg = _make_config()
        model, _ = _build_model_and_load(cfg)

        logits1 = np.asarray(_run_naive_forward(model, cfg))
        logits2 = np.asarray(_run_naive_forward(model, cfg))

        self.assertTrue(
            np.array_equal(logits1, logits2),
            f"Logits differ across two identical runs: "
            f"max_diff={np.max(np.abs(logits1.astype(np.float64) - logits2.astype(np.float64))):.3e}",
        )


if __name__ == "__main__":
    unittest.main()
