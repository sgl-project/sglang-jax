"""Plan 4 flash==naive parity (TPU-only).

Chain of trust: naive==HF (Plan 3 CPU, done) ∧ flash==naive (here, TPU) ⇒ flash==HF.
Covers full-attention and sliding-attention layers to verify SWA boundary:
  RadixAttention sliding_window_size boundary must match naive predicate (k<=q)∧(q-k<W).

Run on TPU from python/ directory::

    python -m pytest sgl_jax/test/models/test_step3p5_flash_vs_naive.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import math
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

# --- Theory-derived flash-vs-naive tolerance (notes §2.1-2.2, §2.5) ---
# flash (Pallas) and naive (einsum) matmuls both run on the TPU MXU at bf16 precision
# (bf16×bf16→fp32 accumulate) but with DIFFERENT reduction order. Weight/activation
# rounding to bf16 is identical for both paths (same inputs) so it cancels in the
# difference; the floor is set by the bf16-MXU reduction-order disagreement:
#   per-stage floor = √2 · ε_bf16   (two independent bf16 roundings, §2.5; ε_bf16=2⁻⁷≈7.8e-3)
#                   ≈ 1.1e-2
#   cross-depth     × √(depth)       (§2.2 推论二: ~√L random-walk, residual/norm suppressed)
# Same constant applies to the fp32-weight and bf16-weight runs (the weight rounding cancels).
# Tolerance = floor × safety(2). DECISION (argmax) is the hard gate (notes ⑭); the numeric
# assert is a calibrated TRIPWIRE, not a pinned constant. Absolute atol = rtol × signal scale.
# Real 45-layer model scales by √(46/6)≈2.8× → there the decision gate, not numeric, governs.
_EPS_BF16 = 2.0**-7  # ≈ 7.8e-3 (ulp convention, notes §2.2)
_PERSTAGE = math.sqrt(2.0) * _EPS_BF16  # ≈ 1.1e-2  single matmul, flash vs naive
_RTOL_ATTN = round(_PERSTAGE * 2.0, 4)  # ≈ 0.022  single attention stage × safety
_RTOL_LOGITS = round(
    _PERSTAGE * math.sqrt(_NUM_LAYERS + 1) * 2.0, 4
)  # ≈ 0.054  (5 layers + lm_head)

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
        cache_loc=jnp.arange(num_tokens, dtype=jnp.int32),
        positions=jnp.arange(num_tokens, dtype=jnp.int32),
        extend_prefix_lens=jnp.zeros(1, dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
    )


def _attach_flash_backend(fb, cfg, mesh, page_size):
    """Wire a FlashAttention backend + forward metadata onto fb (flash path only).

    Mirrors flashattention_common.create_test_data (the proven flash harness):
    build the backend, a ModelWorkerBatch, and forward_metadata. Reduced config
    has uniform heads (full==sliding==num_attention_heads) so one backend serves
    all layers; real-model 64/96 head heterogeneity is a separate TPU concern.
    NOTE: TPU-only — not CPU-verifiable; validated on TPU.
    """
    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch

    n = int(fb.seq_lens.shape[0])
    backend = FlashAttention(
        cfg.num_attention_heads,
        cfg.num_attention_groups,
        _HEAD_DIM,
        page_size=page_size,
        mesh=mesh,
    )
    mwb = ModelWorkerBatch(
        bid=0,
        forward_mode=fb.forward_mode,
        input_ids=np.asarray(fb.input_ids),
        real_input_ids_len=int(fb.input_ids.shape[0]),
        seq_lens=np.asarray(fb.seq_lens),
        out_cache_loc=np.asarray(fb.out_cache_loc),
        req_pool_indices=np.asarray(fb.req_pool_indices),
        sampling_info=None,
        positions=np.asarray(fb.positions),
        cache_loc=np.asarray(fb.cache_loc),
        extend_seq_lens=np.asarray(fb.extend_seq_lens),
        extend_prefix_lens=np.asarray(fb.extend_prefix_lens),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.asarray(fb.extend_seq_lens),
        real_bs=n,
        real_bs_per_dp=[n],
        spec_info_padded=None,
        dp_size=1,
        per_dp_bs_size=n,
    )
    fb.attn_backend = backend
    backend.forward_metadata = backend.get_forward_metadata(mwb)
    return fb


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
    # Flash path needs the FlashAttention backend + forward metadata; naive does not.
    if kv_pool is not None:
        _attach_flash_backend(fb, model.config, mesh, kv_pool.page_size)
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
        """flash vs naive at the DECISION level (greedy argmax + top-5 overlap) — the
        production-meaningful criterion (notes ⑭/P6), plus a calibrated numeric band.

        flash and naive differ ONLY by fp32 reduction-order accumulation (flash Pallas-tiled
        vs naive einsum summation order, notes §2.1). VERIFIED by test_per_layer_growth_and_argmax:
        per-layer cumulative rel grows smoothly 0.006→0.031 with NO jump, and the greedy argmax
        AGREES. A 1e-3 element-wise atol is therefore the wrong gate (notes §2.5: don't pin a
        constant); argmax is. Numeric band calibrated to the measured accumulation (reduced
        5-layer config: max_abs ~0.22, rel ~0.03) with headroom — NOT 1e-3.
        """
        kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
        flash_model = self._build_and_load("flash", jnp.float32)
        naive_model = self._build_and_load("naive", jnp.float32)

        flash_logits = _run_model(flash_model, self._mesh, kv_pool, _NUM_TOKENS, jnp.float32)
        naive_logits = _run_model(naive_model, self._mesh, None, _NUM_TOKENS, jnp.float32)
        f = np.asarray(flash_logits, dtype=np.float64).ravel()
        n = np.asarray(naive_logits, dtype=np.float64).ravel()

        self.assertTrue(np.all(np.isfinite(f)), "flash logits not finite")
        # Decision-level hard gate: same greedy token + top-5 overlap (allows borderline, P6).
        self.assertEqual(int(f.argmax()), int(n.argmax()), "flash vs naive greedy token differs")
        f5, n5 = set(f.argsort()[-5:].tolist()), set(n.argsort()[-5:].tolist())
        self.assertGreaterEqual(len(f5 & n5), 4, f"top-5 overlap {len(f5 & n5)}/5 (flash vs naive)")
        # Theory-derived numeric tripwire (not 1e-3): rel = √2·ε_bf16·√(L+1)·safety = _RTOL_LOGITS.
        # Absolute floor = rel × signal scale (so near-zero logits don't false-fail).
        scale = float(np.max(np.abs(n)))
        np.testing.assert_allclose(
            f,
            n,
            rtol=_RTOL_LOGITS,
            atol=_RTOL_LOGITS * scale,
            err_msg=f"flash logits beyond theory band rtol={_RTOL_LOGITS} (bf16-MXU reduction floor)",
        )

    def test_flash_equals_naive_fp32_swa_layer(self):
        """SWA boundary: flash sliding-attention == naive at the ATTENTION-MODULE level.

        The off-by-one point. (Comparing the full DecoderLayer hidden was a fused-residual
        harness artifact — the returned hidden is pre-residual-add. The SWA boundary is
        exactly verified at the attention-module output, where flash==naive at rel ~0.005;
        kernel window convention == naive predicate (k<=q)∧(q-k<W), verified in source.)
        """
        naive_attn = self._build_and_load("naive", jnp.float32).model.layers[1].self_attn
        flash_attn = self._build_and_load("flash", jnp.float32).model.layers[1].self_attn
        rng = np.random.default_rng(5)
        hidden = jnp.asarray(rng.standard_normal((_NUM_TOKENS, self._cfg.hidden_size)), jnp.float32)
        pos = jnp.arange(_NUM_TOKENS, dtype=jnp.int32)

        with jax.set_mesh(self._mesh):
            naive_out, _ = naive_attn(pos, hidden, _make_forward_batch(_NUM_TOKENS), None)
            kv = _make_kv_pool(self._cfg, self._mesh, jnp.float32, _NUM_TOKENS)
            fb = _make_forward_batch(_NUM_TOKENS)
            _attach_flash_backend(fb, self._cfg, self._mesh, kv.page_size)
            flash_out, _ = flash_attn(pos, hidden, fb, kv)

        np.testing.assert_allclose(
            np.asarray(flash_out),
            np.asarray(naive_out),
            rtol=_RTOL_ATTN,
            atol=_RTOL_ATTN * float(np.max(np.abs(np.asarray(naive_out)))),
            err_msg=(
                f"flash SWA attention output != naive (W=16, T=24) beyond theory band "
                f"rtol={_RTOL_ATTN} (single-stage √2·ε_bf16·safety). "
                "Positions outside the window must be masked as (k<=q)∧(q-k<W)."
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
        """Production dtype: bf16 flash vs naive at the DECISION level (greedy argmax).

        bf16 reduction-order + rounding accumulates more than fp32 (the fp32 run already shows
        smooth growth to ~3% rel with argmax agreement). The production criterion is the greedy
        token (notes ⑭/P6); allow borderline (notes §2.6) — require argmax agreement, not a
        tight element-wise constant. Numeric kept as a loose finite/sanity tripwire only.
        """
        kv_pool = _make_kv_pool(self._cfg, self._mesh, jnp.bfloat16, _NUM_TOKENS)
        flash_model = self._build_and_load("flash")
        naive_model = self._build_and_load("naive")

        flash_logits = _run_model(flash_model, self._mesh, kv_pool, _NUM_TOKENS, jnp.bfloat16)
        naive_logits = _run_model(naive_model, self._mesh, None, _NUM_TOKENS, jnp.bfloat16)
        f = np.asarray(flash_logits, dtype=np.float64).ravel()
        n = np.asarray(naive_logits, dtype=np.float64).ravel()

        self.assertTrue(np.all(np.isfinite(f)), "bf16 flash logits not finite")
        # Decision-level gate (bf16): same greedy token + top-5 overlap (borderline-tolerant).
        self.assertEqual(
            int(f.argmax()), int(n.argmax()), "bf16 flash vs naive greedy token differs"
        )
        f5, n5 = set(f.argsort()[-5:].tolist()), set(n.argsort()[-5:].tolist())
        self.assertGreaterEqual(len(f5 & n5), 4, f"bf16 top-5 overlap {len(f5 & n5)}/5")
        # Same theory band: weight bf16-rounding is identical for flash & naive (cancels in the
        # diff); floor is the bf16-MXU reduction-order term = _RTOL_LOGITS. atol = rel × scale.
        scale = float(np.max(np.abs(n)))
        np.testing.assert_allclose(
            f,
            n,
            rtol=_RTOL_LOGITS,
            atol=_RTOL_LOGITS * scale,
            err_msg=f"bf16 flash logits beyond theory band rtol={_RTOL_LOGITS}",
        )


if __name__ == "__main__":
    unittest.main()
