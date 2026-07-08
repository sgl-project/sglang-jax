"""Tests for Step3p5 full model: DecoderLayer, Model, ForCausalLM.

Covers Task 4 deliverables:
1. Weight round-trip: load real checkpoint key names (incl. pre-stacked experts)
   via load_weights; assert all params loaded, embedding bit-exact.
2. Forward smoke (attn_impl="naive"): construct ForCausalLM on CPU, run single-
   sequence prefill, assert logits shape + finite (no NaN).

Run from python/ directory::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_step3p5_model.py -x -v -o "pythonpath=."
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
    ici_parallelism=[1, 1],
    dcn_parallelism=[1, 1],
    devices=[jax.devices()[0]],
)
jax.sharding.set_mesh(_mesh)


# ---------------------------------------------------------------------------
# Reduced test config (5 layers, small dims, real checkpoint key layout)
# ---------------------------------------------------------------------------
#
# Layer map:
#   0 → dense + full_attention
#   1 → dense + sliding_attention
#   2 → MoE  + sliding_attention
#   3 → MoE  + full_attention
#   4 → MoE  + full_attention  (with non-zero swiglu_limits)
#
# moe_layers_enum = "2,3,4"  → dense layers: 0, 1.
# This exercises both the dense and MoE branches and both attention types.

_VOCAB = 64
_HIDDEN = 64
_INTER = 128  # intermediate_size (dense MLP)
_MOE_INTER = 32  # moe_intermediate_size
_SHARE_DIM = 32  # share_expert_dim
_NUM_EXPERTS = 4  # moe_num_experts (small for CPU)
_TOPK = 2
_NUM_HEADS_FULL = 4
_NUM_HEADS_SLIDE = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
_NUM_LAYERS = 5


def _make_config():
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    return Step3p5Config(
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS_FULL,
        num_attention_groups=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=_VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=64,
        rope_theta=[5000000.0, 10000.0, 10000.0, 5000000.0, 5000000.0],
        rope_scaling=None,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ],
        partial_rotary_factors=[0.5, 1.0, 1.0, 0.5, 0.5],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": _NUM_HEADS_SLIDE,
            "num_attention_groups": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        },
        # Non-zero swiglu limits on layer 4 to exercise the clamp path.
        swiglu_limits=[0.0, 0.0, 0.0, 0.0, 7.0],
        swiglu_limits_shared=[0.0, 0.0, 0.0, 0.0, 16.0],
        moe_layers_enum="2,3,4",
        moe_num_experts=_NUM_EXPERTS,
        moe_top_k=_TOPK,
        moe_intermediate_size=_MOE_INTER,
        share_expert_dim=_SHARE_DIM,
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
        use_moe_router_bias=True,
        sliding_window=16,
        yarn_only_types=["full_attention"],
    )


# ---------------------------------------------------------------------------
# Helpers to build a checkpoint with real key names
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _rand(*shape, dtype=np.float32):
    return _RNG.standard_normal(shape).astype(dtype)


def _build_checkpoint(cfg) -> dict[str, np.ndarray]:
    """Build a minimal safetensors checkpoint matching real key names/shapes.

    Expert weights are pre-stacked [E, out, in] matching the Step 3.5 checkpoint.
    """
    weights: dict[str, np.ndarray] = {}
    H = cfg.hidden_size

    # Global embeddings + final norm + lm_head
    weights["model.embed_tokens.weight"] = _rand(_VOCAB, H)
    weights["model.norm.weight"] = _rand(H)
    weights["lm_head.weight"] = _rand(_VOCAB, H)

    from sgl_jax.srt.models.step3p5 import _moe_layer_ids

    moe_ids = set(_moe_layer_ids(cfg))

    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        layer_types = cfg.layer_types or []
        is_sliding = layer_types[i] == "sliding_attention" if i < len(layer_types) else False

        num_q = (
            cfg.attention_other_setting["num_attention_heads"]
            if is_sliding
            else cfg.num_attention_heads
        )
        num_kv = (
            cfg.attention_other_setting.get("num_attention_groups", cfg.num_attention_groups)
            if is_sliding
            else cfg.num_attention_groups
        )

        # Layer norms (GemmaRMSNorm, weight only, size H).
        weights[f"{p}.input_layernorm.weight"] = _rand(H)
        weights[f"{p}.post_attention_layernorm.weight"] = _rand(H)

        # Attention projections: HF stores [out_dim, in_dim].
        q_dim = num_q * _HEAD_DIM
        kv_dim = num_kv * _HEAD_DIM
        weights[f"{p}.self_attn.q_proj.weight"] = _rand(q_dim, H)
        weights[f"{p}.self_attn.k_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.v_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.o_proj.weight"] = _rand(H, q_dim)
        weights[f"{p}.self_attn.g_proj.weight"] = _rand(num_q, H)
        # QK-norm: [head_dim]
        weights[f"{p}.self_attn.q_norm.weight"] = _rand(_HEAD_DIM)
        weights[f"{p}.self_attn.k_norm.weight"] = _rand(_HEAD_DIM)

        if i not in moe_ids:
            # Dense FFN: HF [out, in]
            weights[f"{p}.mlp.gate_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.up_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.down_proj.weight"] = _rand(H, _INTER)
        else:
            E = _NUM_EXPERTS
            M = _MOE_INTER
            S = _SHARE_DIM
            # MoE router gate [E, H] and bias [E] (f32)
            weights[f"{p}.moe.gate.weight"] = _rand(E, H)
            weights[f"{p}.moe.router_bias"] = _rand(E)
            # Pre-stacked expert weights [E, out, in] — real checkpoint layout.
            weights[f"{p}.moe.gate_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.up_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.down_proj.weight"] = _rand(E, H, M)
            # Shared expert [out, in] — real checkpoint key is share_expert.* (no moe. prefix).
            weights[f"{p}.share_expert.gate_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.up_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.down_proj.weight"] = _rand(H, S)

    return weights


class _DummyModelConfig:
    """Minimal stand-in for ModelConfig with only the fields WeightLoader needs."""

    def __init__(self, model_path: str, cfg):
        self.model_path = model_path
        self._dummy_mode = False
        self.quantization_config = None
        self.num_attention_heads = cfg.num_attention_heads
        # For kv_head_padding: use the base num_kv_heads (no TP replication in CPU test).
        self.num_kv_heads = cfg.num_attention_groups
        self.num_hidden_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        # hf_text_config and hf_config share the same cfg object here.
        self.hf_text_config = cfg
        self.hf_config = cfg

    def get_total_num_kv_heads(self):
        return self.num_kv_heads

    def get_num_kv_head_replicas(self, tensor_parallel_size: int = 1) -> int:
        return 1

    def needs_kv_head_replication(self, tensor_parallel_size: int) -> bool:
        return False

    def get_kv_padding_strategy(self) -> str:
        return "replicate"

    @property
    def ep_size(self):
        return 1


# ---------------------------------------------------------------------------
# Test: weight round-trip
# ---------------------------------------------------------------------------


class TestStep3p5WeightLoading(unittest.TestCase):
    """Save a checkpoint with real key names, load via load_weights, assert correctness."""

    def _build_model_and_load(self, cfg):
        """Construct ForCausalLM, save checkpoint, load, return (model, source_weights)."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        src = _build_checkpoint(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(src, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, cfg)
            with jax.set_mesh(_mesh):
                model.load_weights(mc)

        return model, src

    def test_no_exception_on_load(self):
        """load_weights completes without error."""
        cfg = _make_config()
        self._build_model_and_load(cfg)

    def test_embedding_bit_exact(self):
        """embed_tokens.embedding matches the source checkpoint value exactly (BF16)."""
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        src_emb = jnp.asarray(src["model.embed_tokens.weight"]).astype(jnp.bfloat16)
        loaded_emb = jnp.asarray(model.model.embed_tokens.embedding.value)
        # Embedding is loaded with transpose=False so shape is [vocab, hidden].
        np.testing.assert_array_equal(
            np.asarray(loaded_emb),
            np.asarray(src_emb),
            err_msg="embed_tokens embedding must be bit-exact after BF16 cast",
        )

    def test_dense_mlp_weight_loaded(self):
        """Dense layer's gate_proj.weight is loaded (non-zero, matches source after transpose+BF16)."""
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        # Dense layer 0: gate_proj.weight loaded [INTER, H] → transposed to [H, INTER] in JAX.
        src_gate = jnp.asarray(src["model.layers.0.mlp.gate_proj.weight"]).astype(jnp.bfloat16)
        loaded_gate = jnp.asarray(model.model.layers[0].mlp.gate_proj.weight.value)
        # WeightLoader transposes [out, in] → [in, out], so compare src.T with loaded.
        np.testing.assert_array_equal(
            np.asarray(loaded_gate),
            np.asarray(src_gate.T),
            err_msg="Dense gate_proj.weight must equal checkpoint.T in BF16",
        )

    def test_moe_expert_weight_loaded(self):
        """MoE layer 2's wi_0 (gate_proj stacked) is loaded and transposed correctly.

        Checkpoint: [E, out=M, in=H] → after transpose(0,2,1) → [E, H, M] = EPMoE.wi_0 shape.
        """
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        src_gate = jnp.asarray(src["model.layers.2.moe.gate_proj.weight"]).astype(jnp.bfloat16)
        # src_gate: [E, M, H]; after transpose(0,2,1): [E, H, M]
        expected = jnp.transpose(src_gate, (0, 2, 1))
        loaded_wi0 = jnp.asarray(model.model.layers[2].mlp.experts.wi_0.value)
        np.testing.assert_array_equal(
            np.asarray(loaded_wi0),
            np.asarray(expected),
            err_msg="wi_0 must equal transpose(0,2,1) of checkpoint gate_proj",
        )

    def test_shared_expert_weight_loaded(self):
        """MoE layer 2 shared-expert gate_proj is loaded (key is share_expert.*, not moe.*)."""
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        src_gate = jnp.asarray(src["model.layers.2.share_expert.gate_proj.weight"]).astype(
            jnp.bfloat16
        )
        # LinearBase transposes [out, in] → [in, out].
        expected = jnp.transpose(src_gate, (1, 0))
        loaded = jnp.asarray(model.model.layers[2].mlp.shared_experts.gate_proj.weight.value)
        np.testing.assert_array_equal(
            np.asarray(loaded),
            np.asarray(expected),
            err_msg="shared_experts.gate_proj must be loaded from share_expert.* (not random init)",
        )

    def test_lm_head_loaded(self):
        """lm_head.embedding is loaded correctly."""
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        src_lm = jnp.asarray(src["lm_head.weight"]).astype(jnp.bfloat16)
        loaded_lm = jnp.asarray(model.lm_head.embedding.value)
        np.testing.assert_array_equal(
            np.asarray(loaded_lm),
            np.asarray(src_lm),
            err_msg="lm_head embedding must match checkpoint",
        )

    def test_moe_router_bias_loaded(self):
        """MoE layer 2 router_bias is loaded (float32, shape [E])."""
        cfg = _make_config()
        model, src = self._build_model_and_load(cfg)

        src_bias = src["model.layers.2.moe.router_bias"]
        loaded_bias = np.asarray(model.model.layers[2].mlp.moe_gate.bias.value)
        self.assertEqual(loaded_bias.shape, (cfg.moe_num_experts,))
        self.assertEqual(loaded_bias.dtype, np.float32)
        np.testing.assert_allclose(
            loaded_bias,
            src_bias,
            rtol=1e-6,
            atol=1e-6,
            err_msg="moe_gate.bias values must be close to checkpoint router_bias",
        )


# ---------------------------------------------------------------------------
# Test: forward smoke with attn_impl="naive"
# ---------------------------------------------------------------------------


class TestStep3p5ForwardSmoke(unittest.TestCase):
    """Naive forward on CPU: assert logits shape + finite (no NaN/inf)."""

    def _make_forward_batch(self, num_tokens: int, cfg):
        """Minimal ForwardBatch for a single-sequence prefill."""
        from sgl_jax.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

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

    def test_fp32_linear_acc_env_is_scoped_to_step3p5_linears(self):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        old = os.environ.get("SGL_JAX_STEP35_FP32_LINEAR_ACC")
        os.environ["SGL_JAX_STEP35_FP32_LINEAR_ACC"] = "1"
        try:
            with jax.set_mesh(_mesh):
                model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")
        finally:
            if old is None:
                os.environ.pop("SGL_JAX_STEP35_FP32_LINEAR_ACC", None)
            else:
                os.environ["SGL_JAX_STEP35_FP32_LINEAR_ACC"] = old

        self.assertEqual(model.model.layers[0].self_attn.q_proj.preferred_element_type, jnp.float32)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.output_dtype, jnp.bfloat16)
        self.assertEqual(
            model.model.layers[2].mlp.shared_experts.gate_proj.preferred_element_type,
            jnp.float32,
        )
        self.assertEqual(
            model.model.layers[2].mlp.shared_experts.gate_proj.output_dtype,
            jnp.bfloat16,
        )
        with jax.set_mesh(_mesh):
            q_out, _ = model.model.layers[0].self_attn.q_proj(
                jnp.zeros((2, cfg.hidden_size), dtype=jnp.bfloat16)
            )
        self.assertEqual(q_out.dtype, jnp.bfloat16)

    def test_naive_forward_logits_shape_and_finite(self):
        """ForCausalLM(attn_impl="naive") produces finite logits of correct shape."""
        from sgl_jax.srt.layers.logits_processor import LogitsMetadata
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        num_tokens = 8

        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        fb = self._make_forward_batch(num_tokens, cfg)

        # LogitsMetadata for a simple prefill: return the last token's logits.
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

        lm = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            logits_indices=jnp.array([num_tokens - 1], dtype=jnp.int32),
            extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
            extend_seq_lens_cpu=[num_tokens],
        )

        # naive path does not use memory_pools / token_to_kv_pool.
        class _FakeMemPools:
            token_to_kv_pool = None

        with jax.set_mesh(_mesh):
            result = model(fb, _FakeMemPools(), lm)

        output, _, _, _ = result
        logits = jnp.asarray(output.next_token_logits)

        # Shape: [num_selected_tokens, vocab_size] — 1 output token per request.
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[1], cfg.vocab_size)
        self.assertTrue(
            bool(jnp.all(jnp.isfinite(logits))),
            f"Logits contain non-finite values: {logits}",
        )

    def test_model_is_not_none(self):
        """After construction, self.model is a Step3p5Model (not None)."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM, Step3p5Model

        cfg = _make_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        self.assertIsInstance(model.model, Step3p5Model)

    def test_decoder_layer_is_moe_flag(self):
        """DecoderLayer.is_moe_layer matches expected dense/MoE split."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        # moe_layers_enum = "2,3,4"
        expected = [False, False, True, True, True]
        for i, layer in enumerate(model.model.layers):
            self.assertEqual(
                layer.is_moe_layer,
                expected[i],
                f"Layer {i}: expected is_moe_layer={expected[i]}, got {layer.is_moe_layer}",
            )

    def test_prestacked_expert_loader_fires(self):
        """Verify that the pre-stacked expert branch fires for MoE layer 2.

        Loads a checkpoint, then checks that wi_0 shape == [E, hidden, inter]
        (transpose(0,2,1) was applied), confirming _create_prestacked_moe_lazy_tensor ran.
        """
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        src = _build_checkpoint(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(src, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, cfg)
            with jax.set_mesh(_mesh):
                model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")
                model.load_weights(mc)

        wi0 = model.model.layers[2].mlp.experts.wi_0.value
        # EPMoE shape: [E, hidden_size, intermediate_dim]
        self.assertEqual(wi0.shape, (_NUM_EXPERTS, _HIDDEN, _MOE_INTER))


if __name__ == "__main__":
    unittest.main()
