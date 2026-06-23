"""Plan 4 / Q9 bf16 error-propagation floor (CPU-verifiable).

Runs JAX naive model in fp32 and bf16 with IDENTICAL weights/input.
Computes per-stage relative error |bf16 - fp32| / |fp32| and prints a table.
Documents the bf16 propagation law: real-model bf16 tolerance ≈ this floor
scaled by sqrt(num_layers) + e2e backstop. Does NOT require HF oracle.

Run from python/ directory::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_step3p5_bf16_floor.py -x -v -o "pythonpath=."
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
# Config / checkpoint (mirrors test_step3p5_model.py)
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
_NUM_TOKENS = 12

_RNG = np.random.default_rng(42)


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


def _build_checkpoint(cfg) -> dict[str, np.ndarray]:
    from sgl_jax.srt.models.step3p5 import _moe_layer_ids

    weights: dict[str, np.ndarray] = {}
    H = cfg.hidden_size
    weights["model.embed_tokens.weight"] = _rand(_VOCAB, H)
    weights["model.norm.weight"] = _rand(H)
    weights["lm_head.weight"] = _rand(_VOCAB, H)

    moe_ids = set(_moe_layer_ids(cfg))
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        lt = (cfg.layer_types or [])[i] if i < len(cfg.layer_types or []) else "full_attention"
        is_sliding = lt == "sliding_attention"
        nq = (
            cfg.attention_other_setting["num_attention_heads"]
            if is_sliding
            else cfg.num_attention_heads
        )
        nkv = (
            cfg.attention_other_setting.get("num_attention_groups", cfg.num_attention_groups)
            if is_sliding
            else cfg.num_attention_groups
        )
        weights[f"{p}.input_layernorm.weight"] = _rand(H)
        weights[f"{p}.post_attention_layernorm.weight"] = _rand(H)
        weights[f"{p}.self_attn.q_proj.weight"] = _rand(nq * _HEAD_DIM, H)
        weights[f"{p}.self_attn.k_proj.weight"] = _rand(nkv * _HEAD_DIM, H)
        weights[f"{p}.self_attn.v_proj.weight"] = _rand(nkv * _HEAD_DIM, H)
        weights[f"{p}.self_attn.o_proj.weight"] = _rand(H, nq * _HEAD_DIM)
        weights[f"{p}.self_attn.g_proj.weight"] = _rand(nq, H)
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


def _load_weights(model, weights, cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(weights, os.path.join(tmpdir, "model.safetensors"))
        mc = _DummyModelConfig(tmpdir, cfg)
        with jax.set_mesh(_mesh):
            model.load_weights(mc)


# ---------------------------------------------------------------------------
# Instrumented forward: capture per-stage activations (mirrors alignment harness)
# ---------------------------------------------------------------------------


def _make_forward_batch(num_tokens: int):
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


def _capture_stages(model, forward_batch) -> dict[str, np.ndarray]:
    """Run naive forward and capture per-stage activations as fp64 numpy."""
    caps: dict[str, np.ndarray] = {}

    with jax.set_mesh(_mesh):
        hidden = model.model.embed_tokens(forward_batch.input_ids)
        caps["embed"] = np.asarray(hidden, dtype=np.float64)

        residual = None
        for i, layer in enumerate(model.model.layers):
            if residual is None:
                residual = hidden
                ln_out = layer.input_layernorm(hidden)
            else:
                hidden = hidden + residual
                residual = hidden
                ln_out = layer.input_layernorm(hidden)
            caps[f"ln_in_{i}"] = np.asarray(ln_out, dtype=np.float64)

            attn_out, _ = layer.self_attn(forward_batch.positions, ln_out, forward_batch, None)
            caps[f"attn_{i}"] = np.asarray(attn_out, dtype=np.float64)

            hidden = attn_out + residual
            residual = hidden
            post_ln = layer.post_attention_layernorm(hidden)
            ffn_out = layer.mlp(post_ln)
            caps[f"layer_out_{i}"] = np.asarray(ffn_out + residual, dtype=np.float64)
            hidden = ffn_out

        hidden = hidden + residual
        caps["final_norm"] = np.asarray(model.model.norm(hidden), dtype=np.float64)

    return caps


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a - b| / (mean |b| + eps); both in float64."""
    diff = np.abs(a - b)
    return float(np.mean(diff) / (np.mean(np.abs(b)) + 1e-10))


class TestBf16Floor(unittest.TestCase):
    """Q9 bf16 propagation-floor: per-stage |bf16 - fp32| / |fp32|.

    NOT TPU-gated — uses attn_impl="naive" (pure JAX, CPU-runnable).
    Asserts each stage is in a sane range (≤ 5e-2); does NOT compare vs HF.
    This measures how far bf16 drifts from fp32 per layer on this reduced model.
    Real-model bf16 tolerance scales as: reduced_floor × sqrt(real_layers/test_layers)
    plus an e2e backstop (Q9: real model can't run full fp32 to compare directly).
    """

    # Upper bound per stage: bf16 epsilon ~1e-2; 5e-2 gives 5x headroom for accumulation.
    _MAX_STAGE_ERR = 5e-2

    def test_bf16_floor_per_stage(self):
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        weights = _build_checkpoint(cfg)
        fb = _make_forward_batch(_NUM_TOKENS)

        with jax.set_mesh(_mesh):
            fp32_model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.float32, attn_impl="naive")
            bf16_model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        _load_weights(fp32_model, weights, cfg)
        _load_weights(bf16_model, weights, cfg)

        fp32_caps = _capture_stages(fp32_model, fb)
        bf16_caps = _capture_stages(bf16_model, fb)

        # Ordered stages for the table.
        stage_keys = (
            ["embed"]
            + [s for i in range(_NUM_LAYERS) for s in [f"ln_in_{i}", f"attn_{i}", f"layer_out_{i}"]]
            + ["final_norm"]
        )

        rows = []
        for key in stage_keys:
            if key in fp32_caps and key in bf16_caps:
                err = _rel_err(bf16_caps[key], fp32_caps[key])
                rows.append((key, err))

        # Print floor table (verbatim in test output for the report).
        sep = "=" * 52
        print(f"\n{sep}")
        print("  bf16 propagation floor: |bf16 - fp32| / |fp32|")
        print(f"  Model: {_NUM_LAYERS}-layer reduced, T={_NUM_TOKENS}, naive path")
        print(f"  Max sane threshold per stage: {self._MAX_STAGE_ERR:.1e}")
        print(sep)
        print(f"  {'Stage':<18} | {'Rel-Err':^10} | {'OK?':^6}")
        print("-" * 52)
        for key, err in rows:
            ok = "PASS" if err <= self._MAX_STAGE_ERR else "FAIL"
            print(f"  {key:<18} | {err:^10.3e} | {ok:^6}")
        print(sep)
        print(
            "  NOTE: real-model tolerance ≈ floor × sqrt(real_layers / test_layers) + e2e backstop"
        )
        print(f"  Scaling factor for 40-layer model: ~{(40 / _NUM_LAYERS) ** 0.5:.1f}×")
        print(sep)

        # Assert all stages are within sane band (propagation, not explosion).
        exceeded = [(k, e) for k, e in rows if e > self._MAX_STAGE_ERR]
        self.assertEqual(
            exceeded,
            [],
            f"bf16 floor exceeded {self._MAX_STAGE_ERR:.1e} at stages: {exceeded}. "
            "Possible NaN/explosion in bf16 reduced model.",
        )


if __name__ == "__main__":
    unittest.main()
