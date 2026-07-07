"""Task C engine self-consistency tests (TPU-only).

All tests skip cleanly on CPU — collect and skip, zero errors.

Run on TPU from python/ directory::

    python -m pytest sgl_jax/test/models/test_step3p5_engine.py -x -v -o "pythonpath=."
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

_IS_TPU = jax.devices()[0].platform == "tpu"

# ---------------------------------------------------------------------------
# Config / tolerances — reuse the proven constants from test_step3p5_flash_vs_naive
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
_NUM_LAYERS = 6
_SLIDING_WIN = 16
_PREFIX_LEN = 8  # prefix prefilled before decode steps
_DECODE_STEPS = 3  # number of incremental decode positions to verify

# Theory tolerances from test_step3p5_flash_vs_naive._RTOL_* (same derivation).
_EPS_BF16 = 2.0**-7
_PERSTAGE = math.sqrt(2.0) * _EPS_BF16
# prefill and decode use the same flash kernel — tighter single-attn band:
_RTOL_ATTN = round(_PERSTAGE * 2.0, 4)  # ~0.022
_RTOL_LOGITS = round(
    _PERSTAGE * math.sqrt(_NUM_LAYERS + 1) * 2.0, 4
)  # ~0.0585 = √2·ε_bf16·√(L+1=7)·2
# TRUE fp32 logic gate (see test_prefill_equals_decode): the flash/RPA kernel HONORS
# jax.default_matmul_precision("highest") (6-pass fp32 on the MXU). Under it the fp32
# prefill==decode residual drops to ~1e-6 (measured 1.33e-6 @ 6 layers) vs the DEFAULT
# bf16-input-truncation (~0.035, and it even NaNs on some configs at depth). That isolates
# LOGIC: a structural KV/position/mask/cache bug gives O(0.01-1) — far above this band —
# so any breach is a real bug, not floating-point noise.
_RTOL_FP32_LOGIC = 1e-4

_RNG = np.random.default_rng(13)


def _rand(*shape, dtype=np.float32):
    return _RNG.standard_normal(shape).astype(dtype)


# ---------------------------------------------------------------------------
# Helpers — mirrors test_step3p5_flash_vs_naive (shared by both test classes)
# ---------------------------------------------------------------------------


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
        rope_theta=[5000000.0, 10000.0, 5000000.0, 10000.0, 10000.0, 5000000.0],
        rope_scaling=None,
        # 6 layers covering every real-model combo: dense/MoE × full/sliding ×
        # no-clamp/routed-only/routed+shared. layer 4 = sliding+routed-only (= real 43),
        # layer 5 = full+routed+shared (= real 44).
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        partial_rotary_factors=[0.5, 1.0, 0.5, 1.0, 1.0, 0.5],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": _NUM_HEADS_SLIDE,
            "num_attention_groups": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        },
        swiglu_limits=[0.0, 0.0, 0.0, 0.0, 7.0, 7.0],
        swiglu_limits_shared=[0.0, 0.0, 0.0, 0.0, 0.0, 16.0],
        moe_layers_enum="2,3,4,5",
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
    """Random fp32 weights — same structure as test_step3p5_flash_vs_naive._build_checkpoint."""
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
    """Minimal ModelConfig stand-in — mirrors test_step3p5_flash_vs_naive."""

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


def _make_kv_pool(cfg, mesh, dtype, num_tokens, page_size=None):
    """MHATokenToKVPool — mirrors test_step3p5_flash_vs_naive._make_kv_pool."""
    from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool

    num_kv = cfg.num_attention_groups
    page_size = page_size or max(num_tokens, 1)
    size = ((num_tokens + page_size - 1) // page_size) * page_size
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        head_num=num_kv,
        head_dim=_HEAD_DIM,
        layer_num=cfg.num_hidden_layers,
        mesh=mesh,
    )


def _load_weights(model, weights, cfg, mesh):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        mc = _DummyModelConfig(tmpdir, cfg)
        with jax.set_mesh(mesh):
            model.load_weights(mc)


def _make_prefill_fb(num_tokens: int, token_ids: jax.Array):
    """ForwardBatch for a single-sequence full prefill.

    Mirrors test_step3p5_flash_vs_naive._make_forward_batch exactly (EXTEND mode,
    out_cache_loc[i]=i, cache_loc[i]=i, prefix_lens=0).
    """
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=token_ids,
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        out_cache_loc=jnp.arange(num_tokens, dtype=jnp.int32),
        cache_loc=jnp.arange(num_tokens, dtype=jnp.int32),
        positions=jnp.arange(num_tokens, dtype=jnp.int32),
        extend_prefix_lens=jnp.zeros(1, dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
    )


def _make_decode_fb(prefix_len: int, token_id: int):
    """ForwardBatch for a single DECODE step at position prefix_len.

    seq_lens = prefix_len + 1 (total KV sequence after writing the new token).
    cache_loc covers ALL prefix_len+1 slots so get_forward_metadata can build
    the full page_indices (DECODE needs the whole KV context, not just the new
    token). out_cache_loc points to the single new slot at position prefix_len.
    extend_seq_lens/extend_prefix_lens are None for DECODE.

    Mirrors flashattention_common.create_test_data decode branch.
    """
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    total = prefix_len + 1
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=jnp.array([token_id], dtype=jnp.int32),
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([total], dtype=jnp.int32),
        out_cache_loc=jnp.array([prefix_len], dtype=jnp.int32),
        cache_loc=jnp.arange(total, dtype=jnp.int32),
        positions=jnp.array([prefix_len], dtype=jnp.int32),
        extend_prefix_lens=None,
        extend_seq_lens=None,
    )


def _make_extend_fb(prefix_len: int, token_ids: jax.Array):
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    num_tokens = int(token_ids.shape[0])
    total = prefix_len + num_tokens
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=token_ids,
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([total], dtype=jnp.int32),
        out_cache_loc=jnp.arange(prefix_len, total, dtype=jnp.int32),
        cache_loc=jnp.arange(total, dtype=jnp.int32),
        positions=jnp.arange(prefix_len, total, dtype=jnp.int32),
        extend_prefix_lens=jnp.array([prefix_len], dtype=jnp.int32),
        extend_seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
    )


def _attach_backend(fb, cfg, mesh, page_size, *, mode):
    """Wire FlashAttention backend + forward_metadata onto fb.

    Mirrors test_step3p5_flash_vs_naive._attach_flash_backend for EXTEND;
    uses ForwardMode.DECODE fields (cu_q_lens = [0,1,2,...,bs]) for DECODE.
    get_forward_metadata handles both modes via the proven conditional in
    flashattention_backend.py lines 158-191.
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
    # Build ModelWorkerBatch with mode-appropriate fields.
    if mode == "decode":
        ext_seq = np.ones(n, dtype=np.int32)  # 1 new token per seq
        ext_pre = np.zeros(n, dtype=np.int32)
    else:
        ext_seq = np.asarray(fb.extend_seq_lens)
        ext_pre = np.asarray(fb.extend_prefix_lens)
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
        extend_seq_lens=ext_seq,
        extend_prefix_lens=ext_pre,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=ext_seq,
        real_bs=n,
        real_bs_per_dp=[n],
        spec_info_padded=None,
        dp_size=1,
        per_dp_bs_size=n,
    )
    fb.attn_backend = backend
    backend.forward_metadata = backend.get_forward_metadata(mwb)
    return fb


def _run_prefill_all_positions(model, mesh, kv_pool, token_ids):
    """Full-sequence prefill; return per-position logits [T, vocab] via lm_head on all hidden.

    We run the model forward with extend_seq_lens=[T] so all T positions are processed.
    The LogitsProcessor (EXTEND, no logprob) selects only the LAST position by default.
    To get ALL positions we read from the pre-logits hidden states directly via
    the model.model (backbone) forward, then apply lm_head ourselves.
    """

    T = int(token_ids.shape[0])
    fb = _make_prefill_fb(T, token_ids)
    _attach_backend(fb, model.config, mesh, kv_pool.page_size, mode="prefill")

    class _Pools:
        token_to_kv_pool = kv_pool

    with jax.set_mesh(mesh):
        # Run backbone only to get all T hidden states.
        hidden, layers_kv_fused, _, _ = model.model(fb, kv_pool)
        # Apply final norm then lm_head at every position.
        logits = model.logits_processor._get_logits(hidden, model.lm_head)
    # KV writes are functional in JAX (the kernel returns the updated fused buffer
    # via .at[].set(); it does NOT mutate the pool in place). Persist them exactly
    # as the production model runner does — replace_buffer(layers_kv_fused) — so the
    # subsequent decode steps read the prefix KV instead of an all-zero cache.
    kv_pool.replace_buffer(layers_kv_fused)
    return jnp.asarray(logits, dtype=jnp.float32)  # [T, vocab]


def _run_decode_step(model, mesh, kv_pool, prefix_len, token_id):
    """Single DECODE step: 1 new token at position prefix_len.

    The KV pool must already hold positions 0..prefix_len-1 (from the prefix
    prefill and any prior decode steps). The step reads 0..prefix_len, writes the
    new token's KV at slot prefix_len, and — because JAX KV writes are functional
    — persists the returned fused buffer via replace_buffer so the NEXT decode
    step can read what this step wrote (true autoregressive accumulation).
    Returns logits [1, vocab] for the new position.
    """
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    fb = _make_decode_fb(prefix_len, token_id)
    _attach_backend(fb, model.config, mesh, kv_pool.page_size, mode="decode")
    lm = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        logits_indices=None,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
    )

    class _Pools:
        token_to_kv_pool = kv_pool

    with jax.set_mesh(mesh):
        result = model(fb, _Pools(), lm)
    output, extras, _, _ = result
    # Persist this step's KV write so the next decode step reads it.
    kv_pool.replace_buffer(extras["token_to_kv_pool"])
    return jnp.asarray(output.next_token_logits, dtype=jnp.float32)  # [1, vocab]


def _run_extend_all_positions(model, mesh, kv_pool, prefix_len, token_ids):
    fb = _make_extend_fb(prefix_len, token_ids)
    _attach_backend(fb, model.config, mesh, kv_pool.page_size, mode="prefill")

    with jax.set_mesh(mesh):
        hidden, layers_kv_fused, _, _ = model.model(fb, kv_pool)
        logits = model.logits_processor._get_logits(hidden, model.lm_head)
    kv_pool.replace_buffer(layers_kv_fused)
    return jnp.asarray(logits, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Test C-⓪: multi-token EXTEND causality
# ---------------------------------------------------------------------------


@unittest.skipUnless(_IS_TPU, "multi-token EXTEND causality requires TPU flash kernel")
class TestMultiTokenExtendCausality(unittest.TestCase):
    _mesh = None
    _cfg = None
    _weights = None

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

    def _build_and_load(self):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(
                self._cfg,
                mesh=self._mesh,
                dtype=jnp.float32,
                attn_impl="flash",
            )
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model

    def _prefill_pool(self, model, prefix_ids, total_tokens):
        pool = _make_kv_pool(self._cfg, self._mesh, jnp.float32, total_tokens, page_size=4)
        _run_prefill_all_positions(model, self._mesh, pool, prefix_ids)
        return pool

    def test_first_extend_position_does_not_depend_on_future_tokens(self):
        model = self._build_and_load()
        prefix_ids = jnp.array(np.arange(_PREFIX_LEN, dtype=np.int32) % _VOCAB)
        root_token = jnp.array([17], dtype=jnp.int32)
        future_a = jnp.array([3, 5, 7], dtype=jnp.int32)
        future_b = jnp.array([41, 43, 47], dtype=jnp.int32)
        tokens_a = jnp.concatenate([root_token, future_a])
        tokens_b = jnp.concatenate([root_token, future_b])

        total_tokens = _PREFIX_LEN + int(tokens_a.shape[0])
        with jax.default_matmul_precision("highest"):
            pool_a = self._prefill_pool(model, prefix_ids, total_tokens)
            logits_a = _run_extend_all_positions(model, self._mesh, pool_a, _PREFIX_LEN, tokens_a)
            jax.block_until_ready(logits_a)

            pool_b = self._prefill_pool(model, prefix_ids, total_tokens)
            logits_b = _run_extend_all_positions(model, self._mesh, pool_b, _PREFIX_LEN, tokens_b)
            jax.block_until_ready(logits_b)

        first_a = np.asarray(logits_a[0], dtype=np.float64)
        first_b = np.asarray(logits_b[0], dtype=np.float64)
        max_abs = float(np.max(np.abs(first_a - first_b)))
        scale = float(max(np.max(np.abs(first_a)), 1e-6))
        print(
            "\n=== multi-token EXTEND causality ===\n"
            f" first argmax A={int(first_a.argmax())} B={int(first_b.argmax())}\n"
            f" first max_abs_diff={max_abs:.6e} scale={scale:.6e}"
        )
        self.assertEqual(
            int(first_a.argmax()),
            int(first_b.argmax()),
            "The first EXTEND query changed argmax when only future tokens changed",
        )
        np.testing.assert_allclose(
            first_a,
            first_b,
            rtol=0.0,
            atol=_RTOL_FP32_LOGIC * scale,
            err_msg=(
                "The first EXTEND query must be causally independent of later EXTEND "
                "tokens when the prefix and first token are identical"
            ),
        )


# ---------------------------------------------------------------------------
# Test C-①: prefill == decode (flash self-consistency, I3)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_IS_TPU, "flash prefill==decode requires TPU KV kernel — skip on CPU")
class TestPrefillDecodeConsistency(unittest.TestCase):
    """prefill==decode: full-prefill logit[i] must equal incremental-decode logit[i].

    Both paths use attn_impl="flash". This is flash-vs-flash (same kernel, same weights).
    The only difference is KV context source: prefill writes+reads in one EXTEND pass;
    decode reads the KV written by the prior prefill. Equality proves the KV cache
    round-trip is exact — the strongest engine gate (I3).

    Fixture mirrors test_step3p5_flash_vs_naive._attach_flash_backend + _make_forward_batch
    for the prefill side, and flashattention_common.create_test_data decode branch for the
    decode side (DECODE mode: cu_q_lens=[0,1,...,bs], cache_loc covers full seq).
    """

    _mesh = None
    _cfg = None
    _weights = None

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

    def _build_and_load(self, dtype=jnp.float32):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(self._cfg, mesh=self._mesh, dtype=dtype, attn_impl="flash")
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model

    def _prefill_decode_rows(self, dtype, page_size=None):
        """Run full-prefill vs autoregressive-decode chain at ``dtype``.

        page_size: KV pool page size. bf16 needs an EVEN-derived bkv_sz (RPA v3 packs 2
        bf16 values into 32-bit: kernel asserts bkv_sz % kv_packing == 0). The default odd
        page_size (=T) yields bkv_sz=max_kv//2 odd and the bf16 kernel asserts; pass an even
        page_size (e.g. 4) for bf16. fp32 (kv_packing=1) is unaffected.

        Protocol (per position k):
          1. Reference: one full EXTEND prefill of T = PREFIX_LEN + DECODE_STEPS tokens on
             its own pool -> logits_prefill[k].
          2. Decode side (SEPARATE pool): prefill ONLY the prefix, then a TRUE autoregressive
             chain — decode k reads 0..k, writes its KV at slot k and persists it so decode
             k+1 reads what decode k wrote (exercises decode KV-WRITE accumulation; the pool
             never holds future-token KV, so no same-page future contamination).
        Returns rows (k, argmax_p, argmax_d, max_abs, scale, rel); logits cast to fp64.
        """
        T = _PREFIX_LEN + _DECODE_STEPS
        token_ids = jnp.array(np.random.default_rng(42).integers(0, _VOCAB, T), dtype=jnp.int32)
        model = self._build_and_load(dtype)

        pool_ref = _make_kv_pool(self._cfg, self._mesh, dtype, T, page_size=page_size)
        prefill_logits = _run_prefill_all_positions(model, self._mesh, pool_ref, token_ids)
        jax.block_until_ready(prefill_logits)

        pool_dec = _make_kv_pool(self._cfg, self._mesh, dtype, T, page_size=page_size)
        _run_prefill_all_positions(model, self._mesh, pool_dec, token_ids[:_PREFIX_LEN])

        rows = []
        for step in range(_DECODE_STEPS):
            k = _PREFIX_LEN + step
            decode_token = int(np.asarray(token_ids)[k])
            decode_logits = _run_decode_step(model, self._mesh, pool_dec, k, decode_token)
            jax.block_until_ready(decode_logits)
            p = np.asarray(prefill_logits[k], dtype=np.float64)
            d = np.asarray(decode_logits[0], dtype=np.float64)
            max_abs = float(np.max(np.abs(d - p)))
            scale = float(np.max(np.abs(p)))
            rows.append(
                (k, int(p.argmax()), int(d.argmax()), max_abs, scale, max_abs / (scale + 1e-9))
            )
        return rows

    @staticmethod
    def _print_pd_rows(tag, rows):
        print(f"\n=== prefill==decode per-position [{tag}] (EXTEND vs autoregressive DECODE) ===")
        print(" pos | argmax p | argmax d | max_abs_diff |  scale  |  rel_err")
        for k, ap, ad, mx, sc, rl in rows:
            print(f" {k:>3} | {ap:>8} | {ad:>8} | {mx:>11.4e} | {sc:>7.3f} | {rl:>7.4f}")
        print(f" band: _RTOL_LOGITS={_RTOL_LOGITS} (depth √(L+1))")

    def test_prefill_equals_decode(self):
        """fp32 LOGIC gate: EXTEND-prefill logit[k] == autoregressive-decode logit[k].

        Runs under jax.default_matmul_precision("highest") so the MXU does 6-pass fp32
        (the DEFAULT truncates matmul inputs to bf16 — that is NOT true fp32 and leaves a
        bf16-magnitude residual, defeating the point of a logic gate). At true fp32 the
        reduction-order noise between the two paths collapses to ~1e-6, so this is a TIGHT
        check that the EXTEND-vs-DECODE KV-cache logic (positions/mask/cache indexing) is
        structurally correct: argmax must AGREE and logits within _RTOL_FP32_LOGIC. The
        bf16 production regime is guarded separately by test_prefill_equals_decode_bf16.
        """
        with jax.default_matmul_precision("highest"):
            rows = self._prefill_decode_rows(jnp.float32)
        self._print_pd_rows("fp32", rows)
        for k, ap, ad, *_ in rows:
            self.assertEqual(ap, ad, f"[fp32] prefill vs decode argmax differ at position {k}")
        for k, ap, ad, mx, sc, rl in rows:
            np.testing.assert_allclose(
                mx,
                0.0,
                atol=_RTOL_FP32_LOGIC * max(sc, 1e-6),
                err_msg=(
                    f"[fp32] prefill vs decode logits at pos {k} exceed _RTOL_FP32_LOGIC; "
                    f"max_abs={mx:.6f} scale={sc:.3f} rel={rl:.6f} — a breach at true fp32 "
                    f"means a structural KV/position/mask/cache bug, not fp noise"
                ),
            )

    def test_prefill_equals_decode_bf16(self):
        """bf16 (production regime): same invariant at the production dtype AND precision.

        bf16 is MXU-native, so DEFAULT matmul precision here IS how production serves — the
        measured residual (~0.041) is the genuine bf16 floor, not an artifact. EXTEND and
        DECODE are two reduction orders accumulating bf16 error over L layers, so the
        criterion is logit agreement within the √(L+1) cross-depth band _RTOL_LOGITS, NOT
        strict argmax — near-tie argmax flips within the band are expected (doc §3.1/P6) and
        the numeric band subsumes them (a flip needs top1≈top2 inside the band).

        A per-LAYER jump-detector (assert no single layer's increment >> median) was the
        original design but is infeasible here: microscale is only 6 layers (too few for a
        curve) and the synthetic depth harness NaNs at true depth under DEFAULT precision.
        The √(L+1) band + near-tie tolerance is the right microscale guard; the fp32 test
        above (true-fp32 logic gate) already proves the structure is correct.
        """
        rows = self._prefill_decode_rows(jnp.bfloat16, page_size=4)
        self._print_pd_rows("bf16", rows)
        n_flip = sum(ap != ad for _, ap, ad, *_ in rows)
        print(f" argmax flips (bf16, near-tie expected): {n_flip}/{len(rows)}")
        for k, ap, ad, mx, sc, rl in rows:
            np.testing.assert_allclose(
                mx,
                0.0,
                atol=_RTOL_LOGITS * max(sc, 1e-6),
                err_msg=(
                    f"[bf16] prefill vs decode logits at pos {k} exceed _RTOL_LOGITS; "
                    f"max_abs={mx:.4f} scale={sc:.3f} rel={rl:.4f}"
                ),
            )


# ---------------------------------------------------------------------------
# Test C-②: decode flash == naive (decode-mode flash kernel vs naive oracle)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_IS_TPU, "decode flash==naive requires TPU flash kernel — skip on CPU")
class TestDecodeFlashVsNaive(unittest.TestCase):
    """decode flash==naive: decode-mode flash output matches naive oracle.

    After prefilling T tokens (flash, writing KV), decode token T via flash and
    compare its attention output to naive computed on the full (T+1)-token sequence.
    Validated at the attention-module level (Step3p5Attention.__call__) for one
    full-attention and one sliding-attention layer, mirroring
    test_step3p5_flash_vs_naive.test_flash_equals_naive_fp32_swa_layer.
    This is cleaner than full-model decode (no logits-processor/position ambiguity).
    """

    _mesh = None
    _cfg = None
    _weights = None

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

    def _build_model(self, attn_impl: str, dtype=jnp.float32):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        with jax.set_mesh(self._mesh):
            model = Step3p5ForCausalLM(self._cfg, mesh=self._mesh, dtype=dtype, attn_impl=attn_impl)
        _load_weights(model, self._weights, self._cfg, self._mesh)
        return model

    def _run_attn_layer(self, attn_layer, positions, hidden, fb, kv_pool):
        with jax.set_mesh(self._mesh):
            out, _ = attn_layer(positions, hidden, fb, kv_pool)
        return jnp.asarray(out, dtype=jnp.float32)

    def _check_layer(
        self,
        layer_idx: int,
        label: str,
        prefix_len: int = _PREFIX_LEN,
        page_size=None,
        dtype=jnp.float32,
    ):
        """Compare decode-mode flash output to naive on layer `layer_idx`.

        prefix_len: tokens prefilled before the decode step. Set > sliding_window to
            exercise the SWA window edge at DECODE time (early tokens fall outside
            [k-W+1, k] and must be excluded).
        page_size: KV pool page size. Set < prefix_len+1 so the decode reads KV across
            MULTIPLE pages (the prefix fills earlier pages, the decode token lands in a
            new page), exercising cross-page block indexing in decode mode.
        dtype: fp32 (logic correctness, strict argmax) or bf16 (production regime; argmax
            flips within the band are near-tie, doc §3.1/P6 — numeric band is the gate).
        """
        T = prefix_len  # tokens in the prefix (prefill sequence length)
        rng = np.random.default_rng(99 + layer_idx + prefix_len)
        hidden_np = rng.standard_normal((T + 1, self._cfg.hidden_size)).astype(np.float32)
        hidden_full = jnp.asarray(hidden_np)  # [T+1, H] for naive full sequence
        hidden_prefix = jnp.asarray(hidden_np[:T])  # [T, H] for prefill pass
        hidden_decode = jnp.asarray(hidden_np[T : T + 1])  # [1, H] for decode step

        pos_full = jnp.arange(T + 1, dtype=jnp.int32)
        pos_prefix = jnp.arange(T, dtype=jnp.int32)
        pos_decode = jnp.array([T], dtype=jnp.int32)

        # Build flash model and kv_pool.
        flash_model = self._build_model("flash", dtype)
        kv_pool = _make_kv_pool(self._cfg, self._mesh, dtype, T + 1, page_size=page_size)
        flash_attn = flash_model.model.layers[layer_idx].self_attn

        # Step 1: prefill the T-token prefix to populate KV pool.
        prefix_token_ids = jnp.zeros(T, dtype=jnp.int32)
        fb_prefill = _make_prefill_fb(T, prefix_token_ids)
        _attach_backend(fb_prefill, self._cfg, self._mesh, kv_pool.page_size, mode="prefill")
        with jax.set_mesh(self._mesh):
            _, kv_fused = flash_attn(pos_prefix, hidden_prefix, fb_prefill, kv_pool)
        # KV writes are functional in JAX — the attention returns the updated fused
        # buffer for this layer; it does NOT mutate the pool in place. Persist it (the
        # production model runner does this via replace_buffer(layers_kv_fused)) so the
        # decode step reads the prefix KV instead of an all-zero cache.
        kv_pool.kv_buffer[layer_idx - kv_pool.start_layer] = kv_fused

        # Step 2: decode step for position T.
        fb_decode = _make_decode_fb(T, 0)
        _attach_backend(fb_decode, self._cfg, self._mesh, kv_pool.page_size, mode="decode")
        flash_out = self._run_attn_layer(flash_attn, pos_decode, hidden_decode, fb_decode, kv_pool)

        # Naive oracle: full [T+1] sequence, causal.
        naive_model = self._build_model("naive", dtype)
        naive_attn = naive_model.model.layers[layer_idx].self_attn
        fb_naive = _make_prefill_fb(T + 1, jnp.zeros(T + 1, dtype=jnp.int32))
        with jax.set_mesh(self._mesh):
            naive_out_full, _ = naive_attn(pos_full, hidden_full, fb_naive, None)
        # Extract only the last position (= decode position T).
        naive_out = jnp.asarray(naive_out_full[T : T + 1], dtype=jnp.float32)

        f = np.asarray(flash_out, dtype=np.float64).ravel()
        n = np.asarray(naive_out, dtype=np.float64).ravel()

        # argmax gate: hard for fp32 (logic); soft for bf16 (near-tie flips expected, P6).
        if dtype == jnp.float32:
            self.assertEqual(
                int(f.argmax()),
                int(n.argmax()),
                f"decode flash vs naive argmax differ at layer {layer_idx} ({label})",
            )
        elif int(f.argmax()) != int(n.argmax()):
            print(f"  [bf16] argmax flip at layer {layer_idx} ({label}) — near-tie expected")
        scale = float(np.max(np.abs(n)))
        np.testing.assert_allclose(
            f,
            n,
            rtol=_RTOL_ATTN,
            atol=_RTOL_ATTN * max(scale, 1e-6),
            err_msg=(
                f"[{('fp32' if dtype == jnp.float32 else 'bf16')}] decode flash vs naive attn "
                f"output ({label}) beyond theory band rtol={_RTOL_ATTN} (single-stage √2·ε_bf16·safety)"
            ),
        )

    def test_decode_flash_equals_naive_full_attention(self):
        """Decode-mode flash == naive on layer 0 (full attention, no SWA)."""
        self._check_layer(0, "full_attention")

    def test_decode_flash_equals_naive_sliding_attention(self):
        """Decode-mode flash == naive on layer 1 (sliding_attention, window=16).

        Verifies that the decode-mode flash kernel applies the same SWA boundary
        predicate (k<=q)∧(q-k<W) as the naive oracle at DECODE time.
        """
        self._check_layer(1, "sliding_attention W=16")

    def test_decode_flash_equals_naive_sliding_window_edge(self):
        """(b2) Decode at the SWA window edge: prefix_len (20) > window (16).

        The decode query at position 20 must attend only [20-15 .. 20] and EXCLUDE the
        early tokens 0..4. The previous sliding decode test used prefix_len=8 < window,
        so the window never truncated and the boundary predicate was untested in decode
        mode. This drives the actual exclusion: flash decode must drop the same early
        keys the naive oracle drops.
        """
        self._check_layer(1, "sliding W-edge prefix=20", prefix_len=20)

    def test_decode_flash_equals_naive_multipage(self):
        """(b1) Decode reading KV across MULTIPLE pages (page_size=4, prefix_len=8).

        The prefix fills pages 0,1 (slots 0..7); the decode token lands in a new page 2
        (slot 8) and the decode read spans pages 0,1,2. The previous decode tests used a
        single page (page_size = num_tokens), so cross-page block indexing in decode mode
        was untested. Layer 0 (full attention) isolates the paging from the window logic.
        """
        self._check_layer(0, "full multipage page_size=4", page_size=4)

    # bf16 (production dtype) variants — same invariants in the regime production runs in.
    # fp32 (above) proves the logic; these guard the bf16 decode kernel. argmax flips within
    # the single-stage band are near-tie (doc §3.1/P6); the numeric band _RTOL_ATTN is the gate.
    def test_decode_flash_equals_naive_full_attention_bf16(self):
        self._check_layer(0, "full_attention", page_size=4, dtype=jnp.bfloat16)

    def test_decode_flash_equals_naive_sliding_attention_bf16(self):
        self._check_layer(1, "sliding_attention W=16", page_size=4, dtype=jnp.bfloat16)

    def test_decode_flash_equals_naive_sliding_window_edge_bf16(self):
        self._check_layer(
            1, "sliding W-edge prefix=20", prefix_len=20, page_size=4, dtype=jnp.bfloat16
        )

    def test_decode_flash_equals_naive_multipage_bf16(self):
        self._check_layer(0, "full multipage page_size=4", page_size=4, dtype=jnp.bfloat16)


# ---------------------------------------------------------------------------
# Task C-③ (best-effort): SWA pool == full MHATokenToKVPool at model level
# ---------------------------------------------------------------------------


@unittest.skipUnless(_IS_TPU, "SWA==full requires TPU flash kernel — skip on CPU")
class TestSWAKVPoolVsFullMHA(unittest.TestCase):
    """SWAKVPool output == full MHATokenToKVPool output (storage-only, no numeric change).

    SPEC: output(--disable-hybrid-swa-memory, full MHATokenToKVPool) ==
           output(default SWAKVPool) for the same sequence.
    SWAKVPool construction requires swa_index_mapping wired by ModelRunner
    (model_runner_kv_cache_mixin.py:489 + lines 635-639). Building a correct
    swa_index_mapping in a unit fixture without ModelRunner is fragile (the mapping
    depends on RadixCache eviction logic). Deferred to serving e2e (tests D/E/F).
    """

    def test_swa_pool_equals_full_mha(self):
        self.skipTest(
            "Serving-level: SWAKVPool wiring needs ModelRunner (no real unit fixture "
            "without 测假). Implemented as a real-server self-consistency test in "
            "test/srt/test_step3p5_serving_e2e.py::TestStep3p5SWAEqualsFull "
            "(microscale dummy weights — NOT the 398GB checkpoint)."
        )


# ---------------------------------------------------------------------------
# Serving-level stubs (cache hit==miss / chunked==full) — left as skeletons
# ---------------------------------------------------------------------------


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestRadixCacheHitVsMiss(unittest.TestCase):
    """cache_hit==miss: serving-level, requires RadixCache + real checkpoint."""

    def test_cache_hit_equals_miss(self):
        self.skipTest(
            "Serving-level (needs RadixCache + scheduler — no real unit fixture without "
            "测假). Implemented as a real-server self-consistency test in "
            "test/srt/test_step3p5_serving_e2e.py::TestStep3p5ServingSmokeAndCache "
            "(microscale dummy weights — NOT the 398GB checkpoint)."
        )


@unittest.skipUnless(_IS_TPU, "engine tests require KV pool + RadixCache (TPU) — skip on CPU")
class TestChunkedVsFullPrefill(unittest.TestCase):
    """chunked==full: serving-level, requires chunked-prefill scheduler."""

    def test_chunked_equals_full_prefill(self):
        self.skipTest(
            "Serving-level (needs chunked-prefill scheduler — no real unit fixture "
            "without 测假). Implemented as a real-server self-consistency test in "
            "test/srt/test_step3p5_serving_e2e.py::TestStep3p5ChunkedEqualsFull "
            "(microscale dummy weights — NOT the 398GB checkpoint)."
        )


if __name__ == "__main__":
    unittest.main()
