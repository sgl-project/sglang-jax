"""Plan 4 flash==naive parity (TPU-only).

Chain of trust: naive==HF (Plan 3 CPU, done) ∧ flash==naive (here, TPU) ⇒ flash==HF.
Covers full-attention and sliding-attention layers to verify SWA boundary:
  RadixAttention sliding_window_size boundary must match naive predicate (k<=q)∧(q-k<W).

Run on TPU from python/ directory::

    python -m pytest sgl_jax/test/models/test_step3p5_flash_vs_naive.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import os
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# TPU guard: flash uses RadixAttention (Pallas/TPU kernel), CPU skips cleanly.
_IS_TPU = jax.devices()[0].platform == "tpu"

# ---------------------------------------------------------------------------
# Shared config / checkpoint helpers (mirrors test_step3p5_model.py)
# ---------------------------------------------------------------------------

_VOCAB = 64
_HIDDEN = 64
_INTER = 128
_MOE_INTER = 32
_SHARE_DIM = 32
_NUM_EXPERTS = 4
_TOPK = 2
_NUM_HEADS_FULL = 4
_NUM_HEADS_SLIDE = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
_NUM_LAYERS = 5
_SLIDING_WIN = 16
_NUM_TOKENS = 24  # > window so SWA boundary is exercised

_RNG = np.random.default_rng(7)


def _rand(*shape, dtype=np.float32):
    return _RNG.standard_normal(shape).astype(dtype)


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
            "full_attention",  # layer 0: full MHA  (covers full-attn path)
            "sliding_attention",  # layer 1: SWA dense (covers sliding boundary)
            "sliding_attention",  # layer 2: SWA MoE
            "full_attention",  # layer 3: full MoE
            "full_attention",  # layer 4: full MoE + swiglu_limits
        ],
        partial_rotary_factors=[0.5, 1.0, 1.0, 0.5, 0.5],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": _NUM_HEADS_SLIDE,
            "num_attention_groups": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        },
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
        sliding_window=_SLIDING_WIN,
        yarn_only_types=["full_attention"],
    )


def _build_checkpoint(cfg) -> dict[str, np.ndarray]:
    """Random fp32 weights with real HF key names (shared between flash and naive)."""
    weights: dict[str, np.ndarray] = {}
    H = cfg.hidden_size

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

        weights[f"{p}.input_layernorm.weight"] = _rand(H)
        weights[f"{p}.post_attention_layernorm.weight"] = _rand(H)
        q_dim = num_q * _HEAD_DIM
        kv_dim = num_kv * _HEAD_DIM
        weights[f"{p}.self_attn.q_proj.weight"] = _rand(q_dim, H)
        weights[f"{p}.self_attn.k_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.v_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.o_proj.weight"] = _rand(H, q_dim)
        weights[f"{p}.self_attn.g_proj.weight"] = _rand(num_q, H)
        weights[f"{p}.self_attn.q_norm.weight"] = _rand(_HEAD_DIM)
        weights[f"{p}.self_attn.k_norm.weight"] = _rand(_HEAD_DIM)

        if i not in moe_ids:
            weights[f"{p}.mlp.gate_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.up_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.down_proj.weight"] = _rand(H, _INTER)
        else:
            E, M, S = _NUM_EXPERTS, _MOE_INTER, _SHARE_DIM
            weights[f"{p}.moe.gate.weight"] = _rand(E, H)
            weights[f"{p}.moe.router_bias"] = _rand(E)
            weights[f"{p}.moe.gate_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.up_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.down_proj.weight"] = _rand(E, H, M)
            weights[f"{p}.share_expert.gate_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.up_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.down_proj.weight"] = _rand(H, S)

    return weights


class _DummyModelConfig:
    """Minimal stand-in for ModelConfig — only fields WeightLoader reads."""

    def __init__(self, model_path: str, cfg):
        self.model_path = model_path
        self._dummy_mode = False
        self.quantization_config = None
        self.num_attention_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_attention_groups
        self.num_hidden_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
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


def _make_kv_pool(cfg, mesh, dtype, num_tokens):
    """Minimal MHATokenToKVPool large enough for num_tokens in a single page."""
    from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool

    # Use max kv_heads across all layers (full-attn layers may have more).
    num_kv = cfg.num_attention_groups
    page_size = max(num_tokens, 1)
    size = page_size  # one page suffices for a single-sequence prefill
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv,
        head_dim=_HEAD_DIM,
        layer_num=cfg.num_hidden_layers,
        mesh=mesh,
    )


def _make_forward_batch(num_tokens: int):
    """ForwardBatch for a single-sequence prefill (out_cache_loc[i]=i)."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=jnp.arange(num_tokens, dtype=jnp.int32),
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        out_cache_loc=jnp.arange(num_tokens, dtype=jnp.int32),
        positions=jnp.arange(num_tokens, dtype=jnp.int32),
        extend_prefix_lens=jnp.zeros(1, dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
    )


def _load_weights(model, weights, cfg, mesh):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        mc = _DummyModelConfig(tmpdir, cfg)
        with jax.set_mesh(mesh):
            model.load_weights(mc)


def _run_model(model, mesh, kv_pool, num_tokens, dtype):
    """Run full ForCausalLM forward; return last-token logits [1, vocab]."""
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    fb = _make_forward_batch(num_tokens)
    lm = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        logits_indices=jnp.array([num_tokens - 1], dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        extend_seq_lens_cpu=[num_tokens],
    )

    class _MemPools:
        token_to_kv_pool = kv_pool

    with jax.set_mesh(mesh):
        result = model(fb, _MemPools(), lm)

    output, _, _, _ = result
    return jnp.asarray(output.next_token_logits, dtype=jnp.float32)


@unittest.skipUnless(_IS_TPU, "flash path requires RadixAttention (TPU kernel) — skip on CPU")
class TestFlashVsNaiveFp32(unittest.TestCase):
    """Flash==naive in fp32 with shared random weights.

    Covers layer 0 (full_attention) and layer 1 (sliding_attention, window=16, T=24)
    so the SWA boundary is exercised end-to-end through the whole model.
    Tolerance 1e-3: flash uses fp32 accumulation; kernel reduction order may differ.
    """

    _mesh = None
    _weights = None
    _cfg = None

    @classmethod
    def setUpClass(cls):
        cls._cfg = _make_config()
        cls._mesh = create_device_mesh(
            ici_parallelism=[1, 1],
            dcn_parallelism=[1, 1],
            devices=[jax.devices()[0]],
        )
        jax.sharding.set_mesh(cls._mesh)
        cls._weights = _build_checkpoint(cls._cfg)

    def _build_and_load(self, attn_impl: str, dtype):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(self._cfg, mesh=self._mesh, dtype=dtype, attn_impl=attn_impl)
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model

    def test_flash_equals_naive_fp32_logits(self):
        """Last-token logits: flash ≈ naive within 1e-3 (fp32 weights, fp32 tolerance)."""
        kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
        flash_model = self._build_and_load("flash", jnp.float32)
        naive_model = self._build_and_load("naive", jnp.float32)

        # Naive path ignores kv_pool; flash path reads/writes it.
        flash_logits = _run_model(flash_model, self._mesh, kv_pool, _NUM_TOKENS, jnp.float32)
        naive_logits = _run_model(naive_model, self._mesh, None, _NUM_TOKENS, jnp.float32)

        np.testing.assert_allclose(
            np.asarray(flash_logits),
            np.asarray(naive_logits),
            atol=1e-3,
            rtol=1e-3,
            err_msg=(
                "flash logits differ from naive (fp32). "
                "Covers full-attn (layer 0) and SWA (layers 1-2, W=16, T=24). "
                "SWA boundary: RadixAttention sliding_window_size must match naive "
                "predicate (k<=q)∧(q-k<W)."
            ),
        )

    def test_flash_equals_naive_fp32_swa_layer(self):
        """Per-layer hidden output for sliding-attention layer (layer 1).

        Uses instrumented forward to isolate the SWA layer output directly.
        This is the off-by-one verification: flash SWA boundary == naive predicate.
        """

        naive_model = self._build_and_load("naive", jnp.float32)
        flash_model = self._build_and_load("flash", jnp.float32)
        kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
        fb = _make_forward_batch(_NUM_TOKENS)

        def _layer1_hidden(model, pool):
            # Run embed → layer 0 → layer 1 (swa) and return that layer's attn output.
            with jax.set_mesh(self._mesh):
                hidden = model.model.embed_tokens(fb.input_ids)
                residual = None
                for i, layer in enumerate(model.model.layers[:2]):
                    hidden, residual, _, _ = layer(fb.positions, hidden, fb, pool, residual)
                    if i == 1:  # sliding-attention layer
                        return np.asarray(jnp.asarray(hidden, dtype=jnp.float32))
            return None

        naive_h = _layer1_hidden(naive_model, None)
        flash_h = _layer1_hidden(flash_model, kv_pool)
        np.testing.assert_allclose(
            flash_h,
            naive_h,
            atol=1e-3,
            rtol=1e-3,
            err_msg=(
                "flash SWA layer 1 hidden != naive (W=16, T=24). "
                "Tokens at positions 0..7 are outside the SWA window for query at 24 "
                "— kernel must mask them exactly as naive predicate does."
            ),
        )


@unittest.skipUnless(_IS_TPU, "flash path requires RadixAttention (TPU kernel) — skip on CPU")
class TestFlashVsNaiveBf16(unittest.TestCase):
    """Flash==naive in bf16 (production dtype) — bf16 floor tolerance.

    Validates the production inference path: same weights, both cast to bf16,
    flash kernel output must match naive reference within the bf16 rounding floor.
    bf16 floor ≈ 1e-2 for small models; allow 3e-2 as safe margin.
    """

    _mesh = None
    _weights = None
    _cfg = None

    @classmethod
    def setUpClass(cls):
        cls._cfg = _make_config()
        cls._mesh = create_device_mesh(
            ici_parallelism=[1, 1],
            dcn_parallelism=[1, 1],
            devices=[jax.devices()[0]],
        )
        jax.sharding.set_mesh(cls._mesh)
        cls._weights = _build_checkpoint(cls._cfg)

    def _build_and_load(self, attn_impl: str):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(
                self._cfg, mesh=self._mesh, dtype=jnp.bfloat16, attn_impl=attn_impl
            )
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model

    def test_flash_equals_naive_bf16_logits(self):
        """Production dtype check: bf16 flash logits ≈ bf16 naive within 3e-2.

        bf16 has ~1e-2 rounding error per operation; 3e-2 covers accumulated error
        from kernel reduction order differences between flash and naive.
        """
        kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.bfloat16, _NUM_TOKENS)
        flash_model = self._build_and_load("flash")
        naive_model = self._build_and_load("naive")

        flash_logits = _run_model(flash_model, self._mesh, kv_pool, _NUM_TOKENS, jnp.bfloat16)
        naive_logits = _run_model(naive_model, self._mesh, None, _NUM_TOKENS, jnp.bfloat16)

        np.testing.assert_allclose(
            np.asarray(flash_logits),
            np.asarray(naive_logits),
            atol=3e-2,
            rtol=3e-2,
            err_msg="bf16 flash logits differ from naive beyond bf16 floor (3e-2).",
        )


if __name__ == "__main__":
    unittest.main()
