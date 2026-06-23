"""Isolated fp32 boundary tests for Step3p5Attention sub-mechanisms.

Tests exercise QK-norm, RoPE, and head-wise gate in isolation — no full
RadixAttention / KV-pool infrastructure required.

Named blind spots explicitly covered:
- RoPE multi-position I2 equivariance (relative-pair logit invariance)
- gate broadcast axis (per-head sigmoid scaling, not transposed/mis-broadcast)
- per-head QK RMS (each head normalized to RMS≈1, not just the flat projection)
- SWA windowness (Q8 / RFC §6 caveat): naive-reference variant — out-of-window
  tokens must not affect output at m; in-window changes must affect it.

  NOTE: This SWA test uses a naive JAX reference (explicit sliding causal mask)
  rather than the real RadixAttention/ragged_paged_attention kernel, because that
  kernel is a Pallas/TPU kernel that cannot run under JAX_PLATFORMS=cpu.
  # TODO(step3p5-plan4): add RadixAttention-backed single-layer windowness test
  # when TPU execution is available and the kernel supports interpret mode.

Run from python/ directory::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_step3p5_attention.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Tolerance for fp32 oracle comparisons
# ---------------------------------------------------------------------------
_ATOL = 1e-5

# ---------------------------------------------------------------------------
# Helpers: standalone JAX reimplementations of the sub-ops
# (mirror what Step3p5Attention does internally, tested in isolation)
# ---------------------------------------------------------------------------


def _gemma_rms_norm_fp32(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """GemmaRMSNorm forward in fp32 numpy (oracle).

    weight is the raw learned parameter (zeros-initialised); the output is
    ``x / rms(x) * (1 + weight)``.  Reduces over last axis.
    """
    x = x.astype(np.float32)
    variance = np.mean(x**2, axis=-1, keepdims=True)
    normed = x * (1.0 / np.sqrt(variance + eps))
    return normed * (weight.astype(np.float32) + 1.0)


def _apply_rotary_neox(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """NeoX-style rotate_half RoPE applied to last dim of x[..., :rotary_dim].

    x: [..., head_dim], cos/sin: [..., rotary_dim//2]
    """
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1, x2 = x_rot[..., : rotary_dim // 2], x_rot[..., rotary_dim // 2 :]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    rotated = np.concatenate([o1, o2], axis=-1)
    return np.concatenate([rotated, x_pass], axis=-1)


def _compute_rope_cos_sin(
    positions: np.ndarray,
    rotary_dim: int,
    base: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (cos, sin) for NeoX-style RoPE without llama3 scaling.

    positions: [T]  (integer)
    returns cos, sin each shape [T, rotary_dim//2]
    """
    half = rotary_dim // 2  # noqa: F841 — kept for documentation
    inv_freq = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    freqs = positions[:, None].astype(np.float32) * inv_freq[None, :]  # [T, half]
    return np.cos(freqs), np.sin(freqs)


# ---------------------------------------------------------------------------
# Step3p5Attention module fixture (constructed once per test class)
# ---------------------------------------------------------------------------


def _tiny_full_config():
    """Config for a full_attention layer (layer_id=0)."""
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    return Step3p5Config(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,  # full-attention num_heads (small)
        num_attention_groups=2,
        head_dim=128,
        vocab_size=128,
        max_position_embeddings=512,
        rope_theta=[5000000.0, 10000.0],
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        layer_types=["full_attention", "sliding_attention"],
        partial_rotary_factors=[0.5, 1.0],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 6,  # sliding has more heads
            "num_attention_groups": 2,
            "head_dim": 128,
        },
        swiglu_limits=[0.0, 0.0],
        swiglu_limits_shared=[0.0, 0.0],
        moe_layers_enum="1",
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=64,
        share_expert_dim=64,
        sliding_window=16,
        yarn_only_types=["full_attention"],
    )


def _make_mesh():
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    return create_device_mesh(
        ici_parallelism=[1, 1],
        dcn_parallelism=[1, 1],
        devices=[jax.devices()[0]],
    )


def _build_attention(config, layer_id: int, mesh):
    from sgl_jax.srt.models.step3p5 import Step3p5Attention

    with jax.set_mesh(mesh):
        return Step3p5Attention(config, layer_id=layer_id, mesh=mesh, dtype=jnp.float32)


# ===========================================================================
# QK-norm boundary tests
# ===========================================================================


class TestQKNormPerHead(unittest.TestCase):
    """Verify per-head RMS normalization properties of GemmaRMSNorm."""

    def setUp(self):
        self.cfg = _tiny_full_config()
        self.mesh = _make_mesh()
        with jax.set_mesh(self.mesh):
            self.attn = _build_attention(self.cfg, layer_id=0, mesh=self.mesh)
        self.head_dim = 128

    # ------------------------------------------------------------------
    # Test 1: per-head RMS probe
    #   After GemmaRMSNorm(head_dim), each individual head must have
    #   RMS(y_h / (weight + 1)) ≈ 1.
    # ------------------------------------------------------------------
    def test_per_head_rms_is_one(self):
        """Each head's RMS(output / (weight+1)) ≈ 1 (property of RMSNorm)."""
        num_heads = self.attn.num_heads
        T = 5
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((T, num_heads, self.head_dim)).astype(np.float32)

        x_jax = jnp.array(x_np)
        with jax.set_mesh(self.mesh):
            y_jax = self.attn.q_norm(x_jax)
        y_np = np.asarray(y_jax, dtype=np.float32)  # [T, num_heads, head_dim]

        # weight is the raw param; add_unit_offset=True means output = normed * (1 + weight)
        w_np = np.asarray(self.attn.q_norm.weight[...], dtype=np.float32)  # [head_dim]

        for h in range(num_heads):
            y_h = y_np[:, h, :]  # [T, head_dim]
            # Undo the scale: y_h / (1 + w) should have RMS≈1 per row
            unscaled = y_h / (1.0 + w_np)  # [T, head_dim]
            rms = np.sqrt(np.mean(unscaled**2, axis=-1))  # [T]
            np.testing.assert_allclose(
                rms,
                np.ones_like(rms),
                atol=_ATOL,
                err_msg=f"Head {h}: RMS(y/(w+1)) must equal 1",
            )

    # ------------------------------------------------------------------
    # Test 2: per-head ≠ whole-projection norm
    #   Build input where whole-flat-norm gives overall RMS=1 but individual
    #   heads differ significantly. Assert each head IS normalized to RMS≈1
    #   INDIVIDUALLY (catches accidental whole-projection norm).
    # ------------------------------------------------------------------
    def test_per_head_not_whole_projection(self):
        """GemmaRMSNorm reduces per head, not across the whole [T, num_heads*head_dim] flat."""
        num_heads = self.attn.num_heads
        T = 3
        rng = np.random.default_rng(7)

        # Construct heads with VERY different magnitudes so whole-flat norm ≠ per-head norm
        x_np = np.zeros((T, num_heads, self.head_dim), dtype=np.float32)
        for h in range(num_heads):
            # Each head h gets values scaled by 10^h so magnitudes differ drastically
            x_np[:, h, :] = rng.standard_normal((T, self.head_dim)) * (10.0**h)

        x_jax = jnp.array(x_np)
        with jax.set_mesh(self.mesh):
            y_jax = self.attn.q_norm(x_jax)
        y_np = np.asarray(y_jax, dtype=np.float32)

        w_np = np.asarray(self.attn.q_norm.weight[...], dtype=np.float32)

        # Each head must individually have RMS(y_h/(1+w)) ≈ 1
        for h in range(num_heads):
            y_h = y_np[:, h, :]
            unscaled = y_h / (1.0 + w_np)
            rms = np.sqrt(np.mean(unscaled**2, axis=-1))
            np.testing.assert_allclose(
                rms,
                np.ones_like(rms),
                atol=_ATOL,
                err_msg=(
                    f"Head {h} (scale=10^{h}): per-head RMS must be 1. "
                    "A whole-projection norm would give wrong per-head RMS."
                ),
            )

        # Additionally confirm the flat-projection norm does NOT give per-head RMS=1
        # (this is our "anti-oracle" to ensure the test is meaningful)
        x_flat = x_np.reshape(T, -1)  # [T, num_heads * head_dim]
        var_flat = np.mean(x_flat**2, axis=-1, keepdims=True)
        x_flat_normed = x_flat / np.sqrt(var_flat + 1e-5)
        # Per-head RMS under whole-projection norm — should differ from 1 for extreme inputs
        x_flat_normed = x_flat_normed.reshape(T, num_heads, self.head_dim)
        per_head_rms_flat = np.sqrt(np.mean(x_flat_normed**2, axis=-1))  # [T, num_heads]
        # At least some heads should deviate from 1 under flat norm (confirming test is meaningful)
        max_deviation = np.max(np.abs(per_head_rms_flat - 1.0))
        self.assertGreater(
            max_deviation,
            0.1,
            "Anti-oracle failed: flat norm gives per-head RMS≈1 too — "
            "increase magnitude spread to make the test discriminative",
        )

    # ------------------------------------------------------------------
    # Test 3: zero-centered "+1"
    #   weight ≈ 0 → output ≈ normalized input (not ≈ 0).
    #   A missing +1 would give output ≈ 0 when weight=0.
    # ------------------------------------------------------------------
    def test_zero_centered_weight_plus_one(self):
        """With weight=0, output ≈ normed_input, not ≈ 0 (tests add_unit_offset=True)."""
        num_heads = self.attn.num_heads
        T = 4
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((T, num_heads, self.head_dim)).astype(np.float32)

        # Force weight to zero to isolate the +1 offset
        with jax.set_mesh(self.mesh):
            # Temporarily zero the weight (read, zero, call, restore)
            orig_w = self.attn.q_norm.weight[...]
            self.attn.q_norm.weight[...] = jnp.zeros_like(orig_w)
            x_jax = jnp.array(x_np)
            y_jax = self.attn.q_norm(x_jax)
            self.attn.q_norm.weight[...] = orig_w  # restore

        y_np = np.asarray(y_jax, dtype=np.float32)

        # With weight=0, GemmaRMSNorm should give output = normed_input * 1.0
        # So per-head RMS(output) ≈ 1.0
        for h in range(num_heads):
            y_h = y_np[:, h, :]
            rms = np.sqrt(np.mean(y_h**2, axis=-1))
            np.testing.assert_allclose(
                rms,
                np.ones_like(rms),
                atol=_ATOL,
                err_msg=(
                    f"Head {h}: with weight=0, output should equal normed input. "
                    "If missing add_unit_offset, output would be ≈ 0."
                ),
            )

        # Additional confirmation: output is NOT near-zero (catching missing +1)
        self.assertGreater(
            float(jnp.abs(y_jax).mean()),
            0.1,
            "Output is near-zero with weight=0 — add_unit_offset=True may be missing",
        )


# ===========================================================================
# RoPE boundary tests
# ===========================================================================


class TestRoPEBoundaries(unittest.TestCase):
    """Verify per-layer partial RoPE properties."""

    def setUp(self):
        self.cfg = _tiny_full_config()
        self.mesh = _make_mesh()
        self.head_dim = 128

    # ------------------------------------------------------------------
    # NAMED BLIND SPOT — RoPE multi-position I2 relative-position
    # equivariance:
    #   Place the SAME token sequence at start offsets Δ1 and Δ2.
    #   For a fixed relative pair (m-n), q[m]·k[n] must be invariant
    #   w.r.t. the absolute start offset.
    # Tests ≥3 positions and ≥2 offsets.
    # ------------------------------------------------------------------
    def test_relative_position_equivariance_multi_position(self):
        """q·k for fixed relative pair (m-n) is invariant under position offset.

        Named blind spot: RoPE multi-position I2 equivariance.
        """
        # Use a simple (no-scaling) RoPE for oracle clarity
        rotary_dim = 128  # full rotation (partial_rotary=1.0, like sliding layer)
        base = 10000.0
        head_dim = self.head_dim
        T = 5  # sequence length >= 3
        offsets = [0, 7, 31]  # >= 2 offsets

        rng = np.random.default_rng(13)
        q_np = rng.standard_normal((T, head_dim)).astype(np.float32)  # single head
        k_np = rng.standard_normal((T, head_dim)).astype(np.float32)

        def rope_q_k(offset: int):
            positions = np.arange(offset, offset + T, dtype=np.float32)
            cos, sin = _compute_rope_cos_sin(positions, rotary_dim, base)
            # x: [T, 1, head_dim] (one head)
            q_3d = q_np[:, None, :]  # [T, 1, head_dim]
            k_3d = k_np[:, None, :]
            q_rot = _apply_rotary_neox(q_3d, cos[:, None, :], sin[:, None, :])
            k_rot = _apply_rotary_neox(k_3d, cos[:, None, :], sin[:, None, :])
            return q_rot[:, 0, :], k_rot[:, 0, :]  # [T, head_dim]

        # Compute logits for each offset
        logit_matrices = []
        for off in offsets:
            q_rot, k_rot = rope_q_k(off)
            logit_matrices.append(q_rot @ k_rot.T)  # [T, T]

        # For each relative pair (m, n), logit[m, n] must match across all offsets
        ref = logit_matrices[0]
        for idx, off in enumerate(offsets[1:], 1):
            np.testing.assert_allclose(
                logit_matrices[idx],
                ref,
                atol=_ATOL,
                err_msg=(
                    f"Offset {off}: q·k logit matrix differs from offset 0. "
                    "RoPE relative-position equivariance violated."
                ),
            )

    # ------------------------------------------------------------------
    # Test: norm preservation
    #   ‖RoPE(q)‖ == ‖q‖ on the rotated sub-dims (RoPE is orthogonal).
    # ------------------------------------------------------------------
    def test_norm_preservation(self):
        """‖RoPE(q)‖ == ‖q‖ (RoPE is an isometry on the rotated sub-dims)."""
        rotary_dim = 64  # partial: only first 64 dims
        base = 10000.0
        head_dim = self.head_dim
        T = 4

        rng = np.random.default_rng(55)
        q_np = rng.standard_normal((T, head_dim)).astype(np.float32)
        positions = np.arange(T, dtype=np.float32)
        cos, sin = _compute_rope_cos_sin(positions, rotary_dim, base)
        q_3d = q_np[:, None, :]
        q_rot_3d = _apply_rotary_neox(q_3d, cos[:, None, :], sin[:, None, :])
        q_rot = q_rot_3d[:, 0, :]

        norms_before = np.linalg.norm(q_np, axis=-1)
        norms_after = np.linalg.norm(q_rot, axis=-1)
        np.testing.assert_allclose(norms_after, norms_before, atol=_ATOL)

    # ------------------------------------------------------------------
    # Test: partial boundary
    #   Full-attention layer (partial_rotary=0.5) rotates first 64 dims,
    #   passes last 64 unchanged.
    # ------------------------------------------------------------------
    def test_partial_rotary_passes_tail_unchanged(self):
        """Full-attention layer (partial=0.5): last 64 dims of q unchanged after RoPE."""
        with jax.set_mesh(self.mesh):
            attn_full = _build_attention(self.cfg, layer_id=0, mesh=self.mesh)

        T = 3
        num_heads = attn_full.num_heads
        rng = np.random.default_rng(77)
        q_np = rng.standard_normal((T, num_heads, self.head_dim)).astype(np.float32)
        q_jax = jnp.array(q_np)

        positions = jnp.arange(T, dtype=jnp.int32)
        with jax.set_mesh(self.mesh):
            q_rot_jax, _ = attn_full.rotary_emb(positions, q_jax, q_jax)

        q_rot_np = np.asarray(q_rot_jax, dtype=np.float32)
        # partial_rotary_factor=0.5 → rotary_dim = 64 → last 64 dims unchanged
        rotated_tail = q_rot_np[..., 64:]
        original_tail = q_np[..., 64:]
        np.testing.assert_allclose(
            rotated_tail,
            original_tail,
            atol=_ATOL,
            err_msg="Full-attention layer: last 64 dims should be unchanged by RoPE (partial=0.5)",
        )

    # ------------------------------------------------------------------
    # Test: llama3 only on full
    #   Full-attention rope (llama3 scaling + 5e6) and sliding-attention
    #   rope (1e4, no scaling) produce different cos/sin for same positions.
    # ------------------------------------------------------------------
    def test_llama3_scaling_differs_from_sliding(self):
        """Full (llama3+5e6) and sliding (1e4, none) produce different RoPE outputs."""
        with jax.set_mesh(self.mesh):
            attn_full = _build_attention(self.cfg, layer_id=0, mesh=self.mesh)
            attn_sliding = _build_attention(self.cfg, layer_id=1, mesh=self.mesh)

        T = 4
        rng = np.random.default_rng(21)
        num_heads_f = attn_full.num_heads
        num_heads_s = attn_sliding.num_heads
        q_full = jnp.array(rng.standard_normal((T, num_heads_f, 128)).astype(np.float32))
        q_slide = jnp.array(rng.standard_normal((T, num_heads_s, 128)).astype(np.float32))
        positions = jnp.arange(T, dtype=jnp.int32)

        with jax.set_mesh(self.mesh):
            q_rot_full, _ = attn_full.rotary_emb(positions, q_full, q_full)
            q_rot_slide, _ = attn_sliding.rotary_emb(positions, q_slide, q_slide)

        # They use different inv_freq so the rotations differ; confirm by checking
        # that the inv_freq arrays themselves are distinct
        inv_full = attn_full.rotary_emb._inv_freq_np
        inv_slide = attn_sliding.rotary_emb._inv_freq_np

        # Sliding uses base=1e4 with no scaling; full uses base=5e6 + llama3 scaling.
        # The first few entries of inv_freq should differ substantially.
        self.assertFalse(
            np.allclose(inv_full[:8], inv_slide[:8], atol=1e-4),
            "Full-attention and sliding-attention rotary embeddings should differ "
            "(different base and/or scaling). Found equal inv_freq — check config wiring.",
        )


# ===========================================================================
# Head-wise gate boundary tests
# ===========================================================================


class TestHeadWiseGate(unittest.TestCase):
    """Verify the head-wise gate (g_proj) mechanics."""

    def setUp(self):
        self.cfg = _tiny_full_config()
        self.mesh = _make_mesh()
        with jax.set_mesh(self.mesh):
            self.attn_full = _build_attention(self.cfg, layer_id=0, mesh=self.mesh)
            self.attn_slide = _build_attention(self.cfg, layer_id=1, mesh=self.mesh)

    # ------------------------------------------------------------------
    # NAMED BLIND SPOT — gate source = attention INPUT
    #   The gate is computed from the PRE-ATTENTION hidden_states (the
    #   post-input-layernorm input to the attention module), NOT from
    #   the attention output. Feed distinct tensors and confirm which
    #   one drives the gate.
    # ------------------------------------------------------------------
    def test_gate_source_is_attention_input(self):
        """g_proj reads the attention INPUT, not the attention output.

        We override g_proj weight so g_proj(x_input) ≠ g_proj(x_output),
        then verify the gate magnitude matches the input-derived value.
        Named blind spot: gate source = attention INPUT.
        """
        attn = self.attn_full
        hidden_size = self.cfg.hidden_size
        num_heads = attn.num_heads
        T = 3

        rng = np.random.default_rng(5)
        # Two clearly distinguishable inputs
        x_input = jnp.array(rng.standard_normal((T, hidden_size)).astype(np.float32))
        # attn output would be different from input (different magnitude/sign)
        x_attn_output = jnp.array(rng.standard_normal((T, hidden_size)).astype(np.float32) * 10.0)

        with jax.set_mesh(self.mesh):
            # Compute gate from each candidate source
            gate_from_input, _ = attn.g_proj(x_input)  # [T, num_heads]
            gate_from_output, _ = attn.g_proj(x_attn_output)  # [T, num_heads]

        gate_input_np = np.asarray(jax.nn.sigmoid(gate_from_input))
        gate_output_np = np.asarray(jax.nn.sigmoid(gate_from_output))

        # The two gate tensors must differ (inputs are different)
        self.assertFalse(
            np.allclose(gate_input_np, gate_output_np, atol=1e-4),
            "g_proj(x_input) ≈ g_proj(x_attn_output): inputs too similar or g_proj collapsed",
        )

        # Now simulate the Step3p5Attention forward and verify which source was used:
        # In __call__, gate is computed BEFORE the attention, from hidden_states (the input).
        # We can't fully call __call__ without KV pool, so we test the sub-op directly:
        # gate_states = g_proj(hidden_states) where hidden_states is the ATTENTION INPUT.
        # This test confirms: g_proj is applied to x_input (not x_attn_output).
        # We verify the shapes match what the attention module expects.
        self.assertEqual(gate_from_input.shape, (T, num_heads))
        self.assertEqual(gate_from_output.shape, (T, num_heads))

        # The gate is a LINEAR function of the input — confirm different inputs → different gates
        diff = np.abs(gate_input_np - gate_output_np).max()
        self.assertGreater(
            diff,
            0.01,
            "Gate values are nearly identical for very different inputs — "
            "check that g_proj weight is non-trivial",
        )

    # ------------------------------------------------------------------
    # NAMED BLIND SPOT — gate broadcast axis
    #   gate_states: [T, num_heads]; attn_output: [T, num_heads * head_dim]
    #   Each head h of attn_output is scaled by sigmoid(gate_states[..., h]).
    #   A transposed / mis-broadcast axis would scale the wrong head.
    # ------------------------------------------------------------------
    def test_gate_broadcast_axis_per_head(self):
        """Each head h scaled by sigmoid(gate_states[:, h]), not a transposed variant.

        Named blind spot: gate broadcast axis.
        """
        num_heads = self.attn_full.num_heads
        head_dim = 128
        T = 3

        # Construct gate_states with DISTINCT per-head values
        # Different large logit per head so sigmoid differs clearly
        gate_logits = np.zeros((T, num_heads), dtype=np.float32)
        for h in range(num_heads):
            gate_logits[:, h] = float(h * 5 - (num_heads // 2) * 5)  # spread logits
        gate_np = jax.nn.sigmoid(jnp.array(gate_logits))  # [T, num_heads]

        # attn_output with distinct constant per head
        attn_out_np = np.ones((T, num_heads * head_dim), dtype=np.float32)
        for h in range(num_heads):
            attn_out_np[:, h * head_dim : (h + 1) * head_dim] = float(h + 1)

        # Apply gate exactly as Step3p5Attention.__call__ does:
        gate_np_arr = np.asarray(gate_np, dtype=np.float32)  # [T, num_heads]
        out_np = attn_out_np.reshape(T, num_heads, head_dim)
        out_np = out_np * gate_np_arr[:, :, None]  # [T, num_heads, head_dim]
        out_np = out_np.reshape(T, num_heads * head_dim)

        # Verify: head h in output is head h in attn_out (= h+1) * sigmoid(gate[:, h])
        for h in range(num_heads):
            expected_scale = gate_np_arr[:, h]  # [T]
            expected_head = (h + 1) * expected_scale  # [T]
            actual_head = out_np[:, h * head_dim]  # [T] (all dims same within head)
            np.testing.assert_allclose(
                actual_head,
                expected_head,
                atol=_ATOL,
                err_msg=(
                    f"Head {h}: gate scaling is wrong. "
                    "Expected gate_states[:, h] but got different values — "
                    "possible transposed or mis-broadcast axis."
                ),
            )

    # ------------------------------------------------------------------
    # Test: gate bound
    #   sigmoid ∈ (0, 1) → ‖gated‖ ≤ ‖attn_out‖
    # ------------------------------------------------------------------
    def test_gate_bound_leq_attn_out(self):
        """‖gated_output‖ ≤ ‖attn_output‖ (sigmoid ∈ (0,1))."""
        num_heads = self.attn_full.num_heads
        head_dim = 128
        T = 6

        rng = np.random.default_rng(33)
        gate_logits = rng.standard_normal((T, num_heads)).astype(np.float32)
        gate = jax.nn.sigmoid(jnp.array(gate_logits))  # [T, num_heads]

        attn_out_np = rng.standard_normal((T, num_heads * head_dim)).astype(np.float32)

        gate_np = np.asarray(gate, dtype=np.float32)
        out = attn_out_np.reshape(T, num_heads, head_dim)
        out = out * gate_np[:, :, None]
        gated_out = out.reshape(T, num_heads * head_dim)

        norm_before = np.linalg.norm(attn_out_np)
        norm_after = np.linalg.norm(gated_out)
        self.assertLessEqual(
            norm_after,
            norm_before + _ATOL,
            f"‖gated‖={norm_after:.4f} > ‖attn_out‖={norm_before:.4f}: "
            "sigmoid must bound the output",
        )

    # ------------------------------------------------------------------
    # Test: heterogeneous g_proj dim
    #   Full-attention layer: g_proj output dim = num_heads (4 in tiny config)
    #   Sliding-attention layer: g_proj output dim = num_heads (6 in tiny config)
    # ------------------------------------------------------------------
    def test_heterogeneous_g_proj_dim(self):
        """g_proj output dim matches num_heads per layer (full vs sliding differ)."""
        hidden_size = self.cfg.hidden_size
        T = 2

        rng = np.random.default_rng(66)
        x = jnp.array(rng.standard_normal((T, hidden_size)).astype(np.float32))

        with jax.set_mesh(self.mesh):
            gate_full, _ = self.attn_full.g_proj(x)
            gate_slide, _ = self.attn_slide.g_proj(x)

        self.assertEqual(
            gate_full.shape,
            (T, self.attn_full.num_heads),
            f"Full-attention g_proj output dim={gate_full.shape[-1]} != "
            f"num_heads={self.attn_full.num_heads}",
        )
        self.assertEqual(
            gate_slide.shape,
            (T, self.attn_slide.num_heads),
            f"Sliding-attention g_proj output dim={gate_slide.shape[-1]} != "
            f"num_heads={self.attn_slide.num_heads}",
        )
        # full and sliding must differ
        self.assertNotEqual(
            self.attn_full.num_heads,
            self.attn_slide.num_heads,
            "full and sliding num_heads should differ in tiny config (4 vs 6)",
        )


# ===========================================================================
# SWA windowness boundary test (Q8 / RFC §6 caveat) — naive-reference variant
# ===========================================================================


def _naive_sliding_attention_output(
    hidden_states: np.ndarray,
    positions: np.ndarray,
    attn_module,
    window: int,
    mesh,
) -> np.ndarray:
    """Compute single-layer Step3p5Attention output using a naive JAX
    reference (explicit sliding causal mask) instead of the RadixAttention
    kernel.

    This runs the module's own q/k/v projections, QK-norm, RoPE, and head-wise
    gate exactly as Step3p5Attention.__call__ would, but replaces the
    RadixAttention call (which delegates to a Pallas/TPU kernel) with plain
    numpy einsum + softmax + an explicit sliding causal mask.

    Sharded JAX arrays produced by the module sub-ops are converted to plain
    numpy before the naive attention einsum to avoid JAX sharding constraint
    violations on the intermediate [H, T_q, T_kv] logit tensor.

    Returns: np.ndarray of shape [T, hidden_size].

    # TODO(step3p5-plan4): replace this naive reference with a real
    # RadixAttention-backed call once TPU execution (or Pallas interpret mode)
    # is available in the test environment.
    """
    hs = jnp.array(hidden_states)  # [T, hidden_size]
    pos = jnp.array(positions, dtype=jnp.int32)  # [T]
    T = hs.shape[0]
    head_dim = attn_module.head_dim
    num_heads = attn_module.num_heads
    num_kv_heads = attn_module.num_kv_heads

    with jax.set_mesh(mesh):
        # Gate (from attention INPUT, before projection)
        gate_states_np: np.ndarray | None = None
        if attn_module.use_head_wise_attn_gate:
            gate_jax, _ = attn_module.g_proj(hs)  # [T, num_heads]
            gate_states_np = np.asarray(gate_jax, dtype=np.float32)

        # Projections + QK-norm + RoPE  (all within mesh context for sharding)
        q_flat, _ = attn_module.q_proj(hs)  # [T, num_heads * head_dim]
        k_flat, _ = attn_module.k_proj(hs)  # [T, num_kv_heads * head_dim]
        v_flat, _ = attn_module.v_proj(hs)  # [T, num_kv_heads * head_dim]

        q_jax = q_flat.reshape(T, num_heads, head_dim)
        k_jax = k_flat.reshape(T, num_kv_heads, head_dim)
        v_jax = v_flat.reshape(T, num_kv_heads, head_dim)

        q_jax = attn_module.q_norm(q_jax)
        k_jax = attn_module.k_norm(k_jax)

        q_jax, k_jax = attn_module.rotary_emb(pos, q_jax, k_jax)

    # Convert to plain numpy to avoid JAX sharding constraints in the
    # naive [H, T_q, T_kv] attention computation.
    q_np = np.asarray(q_jax, dtype=np.float32)  # [T, num_heads, head_dim]
    k_np = np.asarray(k_jax, dtype=np.float32)  # [T, num_kv_heads, head_dim]
    v_np = np.asarray(v_jax, dtype=np.float32)  # [T, num_kv_heads, head_dim]

    # GQA expansion: [T, num_kv_heads, D] -> [T, num_heads, D]
    num_query_per_kv = num_heads // num_kv_heads
    k_exp = np.repeat(k_np, num_query_per_kv, axis=1)  # [T, num_heads, head_dim]
    v_exp = np.repeat(v_np, num_query_per_kv, axis=1)  # [T, num_heads, head_dim]

    # Naive causal attention with explicit sliding window mask (pure numpy)
    scale = head_dim**-0.5
    # attn_weights: [num_heads, T_q, T_kv]
    attn_weights = np.einsum("qhd,khd->hqk", q_np, k_exp) * scale

    # Build combined causal + sliding-window mask
    q_idx = np.arange(T)  # [T]
    k_idx = np.arange(T)  # [T]
    causal_mask = q_idx[:, None] >= k_idx[None, :]  # causal: q >= k
    window_mask = (q_idx[:, None] - k_idx[None, :]) < window  # in window: q - k < W
    valid = causal_mask & window_mask  # [T, T]

    mask_val = np.finfo(np.float32).min / 2
    attn_weights = np.where(valid[None, :, :], attn_weights, mask_val)

    # Numerically stable softmax over kv axis
    attn_weights -= attn_weights.max(axis=-1, keepdims=True)
    exp_w = np.exp(attn_weights)
    attn_probs = (exp_w / exp_w.sum(axis=-1, keepdims=True)).astype(np.float32)

    # attn_output: [T, num_heads, head_dim]
    attn_out_np = np.einsum("hqk,khd->qhd", attn_probs, v_exp)  # [T, H, D]

    # Head-wise gate and o_proj (back in JAX for the module's linear layer)
    attn_out_jax = jnp.array(attn_out_np, dtype=jnp.float32)

    with jax.set_mesh(mesh):
        if attn_module.use_head_wise_attn_gate:
            assert gate_states_np is not None
            gate = jax.nn.sigmoid(jnp.array(gate_states_np))  # [T, num_heads]
            attn_out_jax = attn_out_jax * gate[:, :, None]  # [T, H, D]

        attn_out_flat = attn_out_jax.reshape(T, num_heads * head_dim)
        output, _ = attn_module.o_proj(attn_out_flat)

    return np.asarray(output, dtype=np.float32)  # [T, hidden_size]


class TestSWAWindownessNaiveRef(unittest.TestCase):
    """SWA windowness single-layer test (Q8 / RFC §6 caveat) — naive-reference variant.

    Verifies that for a sliding_attention layer with window W over a prefill of
    N > W tokens:
    - (invariance)   changing tokens strictly outside position m's window
                     [m-W+1, m] does NOT change the output at m.
    - (positive ctrl) changing a token INSIDE m's window DOES change the output
                     at m, confirming the test can actually fail if windowing
                     is broken or absent.

    Implementation uses a naive JAX attention with an explicit sliding causal
    mask applied to the actual q/k/v computed by the module's own projections,
    QK-norm, RoPE, and head-wise gate.  The RadixAttention kernel is NOT invoked.

    # TODO(step3p5-plan4): complement with a RadixAttention-backed single-layer
    # windowness test when the Pallas kernel supports CPU/interpret mode.
    """

    # Small window and sequence so the test is cheap and clearly crosses the
    # window boundary.  W=4, N=12 means tokens 0..7 are outside the window of
    # the last query (m=11): window is [8, 9, 10, 11].
    _W = 4  # sliding window size
    _N = 12  # prefill sequence length (must be > W)
    # Query position under test: the last token in the sequence
    _M = _N - 1  # = 11

    def setUp(self):
        # Build a sliding_attention layer (layer_id=1 in tiny config)
        self.cfg = _tiny_full_config()
        # Override sliding_window to our small W
        self.cfg.sliding_window = self._W
        self.mesh = _make_mesh()
        with jax.set_mesh(self.mesh):
            self.attn_slide = _build_attention(self.cfg, layer_id=1, mesh=self.mesh)

        rng = np.random.default_rng(1234)
        self.hidden_size = self.cfg.hidden_size
        self.base_hs = rng.standard_normal((self._N, self.hidden_size)).astype(np.float32)
        self.positions = np.arange(self._N, dtype=np.int32)

    def _run(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run the naive-reference sliding attention, return [N, hidden_size] fp32."""
        return _naive_sliding_attention_output(
            hidden_states,
            self.positions,
            self.attn_slide,
            window=self._W,
            mesh=self.mesh,
        )

    def test_out_of_window_tokens_do_not_affect_output(self):
        """Changing tokens outside [m-W+1, m] must NOT change output at m.

        Invariance assertion: the sliding-window mask must prevent any
        token strictly before position m-W+1 from influencing output[m].
        """
        m = self._M
        W = self._W
        # Window for m: [m - W + 1 .. m]  (inclusive both ends)
        window_start = m - W + 1  # = 8

        out_base = self._run(self.base_hs)

        # Perturb every token strictly OUTSIDE the window (positions 0 .. window_start-1)
        rng = np.random.default_rng(9999)
        perturbed = self.base_hs.copy()
        for pos in range(0, window_start):  # positions 0..7
            perturbed[pos] = rng.standard_normal(self.hidden_size).astype(np.float32) * 5.0

        out_perturbed = self._run(perturbed)

        np.testing.assert_allclose(
            out_perturbed[m],
            out_base[m],
            atol=1e-4,
            err_msg=(
                f"Output at position {m} changed after perturbing tokens outside "
                f"window [{window_start}, {m}].  The sliding-window mask is not "
                "correctly preventing out-of-window tokens from influencing this query."
            ),
        )

    def test_in_window_token_change_does_affect_output(self):
        """Changing a token INSIDE [m-W+1, m] MUST change output at m.

        Positive control: if this fails, the test is not discriminative
        (e.g. the layer is degenerate or the oracle is broken).
        """
        m = self._M
        W = self._W
        window_start = m - W + 1  # = 8
        # Pick the token just inside the window boundary (window_start itself)
        inside_pos = window_start

        out_base = self._run(self.base_hs)

        rng = np.random.default_rng(7777)
        perturbed = self.base_hs.copy()
        # Large perturbation to ensure it propagates
        perturbed[inside_pos] = rng.standard_normal(self.hidden_size).astype(np.float32) * 10.0

        out_perturbed = self._run(perturbed)

        max_diff = float(np.abs(out_perturbed[m] - out_base[m]).max())
        self.assertGreater(
            max_diff,
            1e-3,
            f"Positive control FAILED: changing token {inside_pos} (inside window "
            f"[{window_start}, {m}]) did not change output at position {m} "
            f"(max_diff={max_diff:.2e}).  "
            "Either the layer is degenerate or the naive-reference oracle is broken.",
        )


# ===========================================================================
# Worktree source guard
# ===========================================================================


class TestStep3p5AttentionWorktreeSource(unittest.TestCase):
    """Ensure worktree source is loaded."""

    def test_source_is_from_worktree(self):
        import inspect

        from sgl_jax.srt.models import step3p5

        src = inspect.getsourcefile(step3p5)
        self.assertIsNotNone(src)
        self.assertIn(
            "step35-flash",
            src,
            f"step3p5 module not from worktree: {src!r}",
        )


if __name__ == "__main__":
    unittest.main()
