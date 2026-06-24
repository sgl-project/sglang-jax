"""HF<->JAX fp32 per-layer alignment harness for Step 3.5 Flash (Task 5).

Emits a per-stage/layer error table with rel-err and cosine similarity.
Uses REAL HF AutoModelForCausalLM eager as the oracle (fp32, trust_remote_code).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

try:
    import torch
except ImportError:  # torch is the HF reference oracle; absent in pure-JAX envs (pod/CI-cpu)
    torch = None
from safetensors.numpy import save_file as save_np_safetensors

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_mesh)

# ---------------------------------------------------------------------------
# Tiny 5-layer config covering all archetypes:
#   0: full_attention + dense
#   1: sliding_attention + dense
#   2: sliding_attention + MoE
#   3: full_attention + MoE
#   4: full_attention + MoE + swiglu_limits (clamp)
# ---------------------------------------------------------------------------

_VOCAB, _HIDDEN, _INTER = 64, 256, 256
_MOE_INTER, _SHARE_DIM, _NUM_EXPERTS, _TOPK = 64, 64, 8, 2
_NUM_HEADS_FULL, _NUM_HEADS_SLIDE, _NUM_KV = 4, 6, 2
_HEAD_DIM, _NUM_LAYERS = 128, 5

_TINY_CFG_DICT: dict = {
    "architectures": ["Step3p5ForCausalLM"],
    "model_type": "step3p5",
    "auto_map": {
        "AutoConfig": "configuration_step3p5.Step3p5Config",
        "AutoModelForCausalLM": "modeling_step3p5.Step3p5ForCausalLM",
    },
    "hidden_size": _HIDDEN,
    "intermediate_size": _INTER,
    "num_hidden_layers": _NUM_LAYERS,
    "num_attention_heads": _NUM_HEADS_FULL,
    "num_attention_groups": _NUM_KV,
    "head_dim": _HEAD_DIM,
    "vocab_size": _VOCAB,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 64,
    "rope_theta": [5000000.0, 10000.0, 10000.0, 5000000.0, 5000000.0],
    "rope_scaling": None,
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ],
    "partial_rotary_factors": [0.5, 1.0, 1.0, 0.5, 0.5],
    "attention_other_setting": {
        "attention_type": "sliding_attention",
        "num_attention_heads": _NUM_HEADS_SLIDE,
        "num_attention_groups": _NUM_KV,
        "head_dim": _HEAD_DIM,
    },
    "swiglu_limits": [0.0, 0.0, 0.0, 0.0, 7.0],
    "swiglu_limits_shared": [0.0, 0.0, 0.0, 0.0, 16.0],
    "moe_layers_enum": "2,3,4",
    "moe_num_experts": _NUM_EXPERTS,
    "moe_top_k": _TOPK,
    "moe_intermediate_size": _MOE_INTER,
    "share_expert_dim": _SHARE_DIM,
    "share_expert_dims": _SHARE_DIM,
    "moe_router_scaling_factor": 3.0,
    "moe_router_activation": "sigmoid",
    "norm_expert_weight": True,
    "use_moe_router_bias": True,
    "use_head_wise_attn_gate": True,
    "sliding_window": 4,
    "yarn_only_types": ["full_attention"],
    "tie_word_embeddings": False,
    "need_fp32_gate": True,
    "use_qk_norm": True,
    "zero_centered": True,
    "use_cache": False,
}

_HF_SRC = os.environ.get("STEP35_HF_SRC", "/Users/infiscale/develop")

# Dev-local oracle: needs the HF modeling/config files present. Skip on CI where absent.
_HF_AVAILABLE = (
    torch is not None
    and os.path.isfile(os.path.join(_HF_SRC, "modeling_step3p5.py"))
    and os.path.isfile(os.path.join(_HF_SRC, "configuration_step3p5.py"))
)

# ---------------------------------------------------------------------------
# Shared checkpoint generator (fp32, fixed seed)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _rand(*shape: int) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _build_checkpoint_np() -> dict[str, np.ndarray]:
    """Generate random fp32 weights with real HF key names (shared seed for HF+JAX)."""
    w: dict[str, np.ndarray] = {}
    H = _HIDDEN
    w["model.embed_tokens.weight"] = _rand(_VOCAB, H)
    w["model.norm.weight"] = _rand(H)
    w["lm_head.weight"] = _rand(_VOCAB, H)

    for i in range(_NUM_LAYERS):
        p = f"model.layers.{i}"
        is_sliding = _TINY_CFG_DICT["layer_types"][i] == "sliding_attention"
        nq = _NUM_HEADS_SLIDE if is_sliding else _NUM_HEADS_FULL
        nkv = _NUM_KV
        w[f"{p}.input_layernorm.weight"] = _rand(H)
        w[f"{p}.post_attention_layernorm.weight"] = _rand(H)
        w[f"{p}.self_attn.q_proj.weight"] = _rand(nq * _HEAD_DIM, H)
        w[f"{p}.self_attn.k_proj.weight"] = _rand(nkv * _HEAD_DIM, H)
        w[f"{p}.self_attn.v_proj.weight"] = _rand(nkv * _HEAD_DIM, H)
        w[f"{p}.self_attn.o_proj.weight"] = _rand(H, nq * _HEAD_DIM)
        w[f"{p}.self_attn.g_proj.weight"] = _rand(nq, H)
        w[f"{p}.self_attn.q_norm.weight"] = _rand(_HEAD_DIM)
        w[f"{p}.self_attn.k_norm.weight"] = _rand(_HEAD_DIM)
        if i in (0, 1):  # dense
            w[f"{p}.mlp.gate_proj.weight"] = _rand(_INTER, H)
            w[f"{p}.mlp.up_proj.weight"] = _rand(_INTER, H)
            w[f"{p}.mlp.down_proj.weight"] = _rand(H, _INTER)
        else:  # MoE
            E, M, S = _NUM_EXPERTS, _MOE_INTER, _SHARE_DIM
            w[f"{p}.moe.gate.weight"] = _rand(E, H)
            w[f"{p}.moe.router_bias"] = _rand(E)
            w[f"{p}.moe.gate_proj.weight"] = _rand(E, M, H)
            w[f"{p}.moe.up_proj.weight"] = _rand(E, M, H)
            w[f"{p}.moe.down_proj.weight"] = _rand(E, H, M)
            w[f"{p}.share_expert.gate_proj.weight"] = _rand(S, H)
            w[f"{p}.share_expert.up_proj.weight"] = _rand(S, H)
            w[f"{p}.share_expert.down_proj.weight"] = _rand(H, S)
    return w


# ---------------------------------------------------------------------------
# HF model loading helpers
# ---------------------------------------------------------------------------


def _make_hf_tmpdir(weights_np: dict[str, np.ndarray]) -> str:
    """Write config.json + HF py files + torch safetensors; return tmpdir path."""
    tmpdir = tempfile.mkdtemp(prefix="step35_align_")
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(_TINY_CFG_DICT, f)
    shutil.copy(os.path.join(_HF_SRC, "modeling_step3p5.py"), tmpdir)
    shutil.copy(os.path.join(_HF_SRC, "configuration_step3p5.py"), tmpdir)
    # HF safetensors loader requires torch tensors.
    from safetensors.torch import save_file as _save_torch

    weights_torch = {k: torch.from_numpy(v) for k, v in weights_np.items()}
    _save_torch(weights_torch, os.path.join(tmpdir, "model.safetensors"))
    return tmpdir


def _load_hf_model(tmpdir: str, weights_np: dict[str, np.ndarray]):
    """Load HF fp32 eager model and break the weight tie for lm_head."""
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        tmpdir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    hf_model.eval()

    # HF Step3p5ForCausalLM/_tied_weights_keys ties lm_head to embed_tokens.
    # Break the tie so lm_head carries the correct independent weights.
    if hf_model.lm_head.weight.data_ptr() == hf_model.model.embed_tokens.weight.data_ptr():
        import warnings

        warnings.warn(
            "Breaking HF weight tie for lm_head — reloading from checkpoint",
            stacklevel=2,
        )
        hf_model.lm_head.weight = torch.nn.Parameter(hf_model.model.embed_tokens.weight.clone())
        with torch.no_grad():
            hf_model.lm_head.weight.copy_(torch.from_numpy(weights_np["lm_head.weight"]))

    return hf_model


# ---------------------------------------------------------------------------
# HF forward-hook capture
# ---------------------------------------------------------------------------


def _capture_hf_activations(hf_model, input_ids_torch) -> dict[tuple, np.ndarray]:
    """Register forward hooks, run HF model, return dict of (stage, layer) -> np.ndarray."""
    caps: dict = {}
    hooks = []

    def _make_hook(key: tuple):
        def _hook(mod, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            # Squeeze batch dim [1, T, H] -> [T, H].
            caps[key] = tensor.detach().float().squeeze(0).numpy()

        return _hook

    hooks.append(hf_model.model.embed_tokens.register_forward_hook(_make_hook(("embed", None))))
    for i, layer in enumerate(hf_model.model.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(_make_hook(("ln_in", i))))
        hooks.append(layer.self_attn.register_forward_hook(_make_hook(("attn", i))))
        hooks.append(layer.register_forward_hook(_make_hook(("layer_out", i))))
    hooks.append(hf_model.model.norm.register_forward_hook(_make_hook(("final_norm", None))))

    with torch.no_grad():
        hf_model(input_ids=input_ids_torch)

    for h in hooks:
        h.remove()
    return caps


# ---------------------------------------------------------------------------
# JAX model loading helpers
# ---------------------------------------------------------------------------


def _make_jax_config():
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    return Step3p5Config(
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS_FULL,
        num_attention_groups=_NUM_KV,
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
            "num_attention_groups": _NUM_KV,
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
        sliding_window=4,
        yarn_only_types=["full_attention"],
    )


class _DummyModelConfig:
    """Minimal stand-in for ModelConfig with only the fields WeightLoader needs."""

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


def _load_jax_weights(jax_model, weights_np: dict[str, np.ndarray], jax_cfg) -> None:
    """Save fp32 weights to a temp safetensors file and load via load_weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_np_safetensors(weights_np, os.path.join(tmpdir, "model.safetensors"))
        mc = _DummyModelConfig(tmpdir, jax_cfg)
        with jax.set_mesh(_mesh):
            jax_model.load_weights(mc)


# ---------------------------------------------------------------------------
# JAX instrumented forward (captures same stages as HF hooks)
# ---------------------------------------------------------------------------


def _make_forward_batch(input_ids_np: np.ndarray):
    """Build a minimal ForwardBatch for a single-sequence prefill."""
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    T = len(input_ids_np)
    return ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=jnp.asarray(input_ids_np, dtype=jnp.int32),
        req_pool_indices=jnp.zeros(1, dtype=jnp.int32),
        seq_lens=jnp.array([T], dtype=jnp.int32),
        out_cache_loc=jnp.zeros(T, dtype=jnp.int32),
        positions=jnp.arange(T, dtype=jnp.int32),
        extend_prefix_lens=jnp.zeros(1, dtype=jnp.int32),
        extend_seq_lens=jnp.array([T], dtype=jnp.int32),
    )


def _jax_instrumented_forward(model, forward_batch, jax_caps: dict) -> dict[tuple, np.ndarray]:
    """Run JAX model layer-by-layer, capturing the same activation stages as HF.

    Mirrors Step3p5DecoderLayer.__call__ and Step3p5Model.__call__ exactly.
    Returns jax_caps populated in-place.
    """
    hidden = model.model.embed_tokens(forward_batch.input_ids)
    jax_caps[("embed", None)] = np.asarray(hidden)

    residual = None
    for i, layer in enumerate(model.model.layers):
        # Replicate Step3p5DecoderLayer.__call__ pre-norm fused-residual logic.
        if residual is None:
            # Layer 0: residual = embed, hidden = layernorm(embed)
            residual = hidden
            ln_out = layer.input_layernorm(hidden)
        else:
            # Layer N>0: hidden = mlp_out + residual → then layernorm
            hidden = hidden + residual
            residual = hidden
            ln_out = layer.input_layernorm(hidden)
        jax_caps[("ln_in", i)] = np.asarray(ln_out)

        # Attention (naive path, no KV pool).
        attn_out, _ = layer.self_attn(forward_batch.positions, ln_out, forward_batch, None)
        jax_caps[("attn", i)] = np.asarray(attn_out)

        # Post-attention residual add + FFN.
        hidden = attn_out + residual
        residual = hidden
        post_ln = layer.post_attention_layernorm(hidden)
        ffn_out = layer.mlp(post_ln)

        # HF layer output = ffn_out + residual (full residual connection).
        # JAX Step3p5DecoderLayer returns (ffn_out, residual, ...) — next layer adds them.
        layer_full_out = ffn_out + residual
        jax_caps[("layer_out", i)] = np.asarray(layer_full_out)

        # Set state for next iteration to match Step3p5Model.__call__ loop:
        # hidden_states=ffn_out, residual=residual (post-attn hidden).
        hidden = ffn_out
        # residual already holds post-attn hidden (set above); unchanged.

    # Final residual add + norm (mirrors Step3p5Model.__call__ tail).
    hidden = hidden + residual
    final_norm_out = model.model.norm(hidden)
    jax_caps[("final_norm", None)] = np.asarray(final_norm_out)
    return jax_caps


# ---------------------------------------------------------------------------
# Error metrics and table printing
# ---------------------------------------------------------------------------


def _compute_error_metrics(hf: np.ndarray, jax: np.ndarray) -> tuple[float, float]:
    """Return (rel_err, cosine_sim) between two tensors (float64 arithmetic)."""
    diff = np.abs(jax.astype(np.float64) - hf.astype(np.float64))
    ref_norm = np.abs(hf.astype(np.float64))
    rel_err = float(np.mean(diff) / (np.mean(ref_norm) + 1e-10))
    a = jax.flatten().astype(np.float64)
    b = hf.flatten().astype(np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    return rel_err, cosine


def _print_alignment_table(
    rows: list,
    floor: float,
    safety: float,
    embed_bit_exact: bool,
    first_fail: tuple | None,
    embed_max_diff: float,
) -> None:
    """Print formatted alignment table to stdout."""
    threshold = floor * safety
    sep = "=" * 60
    dash = "-" * 60
    print(f"\n{sep}")
    print("  HF <-> JAX fp32 Per-Layer Alignment Table")
    print(f"  Floor: {floor:.2e}, Safety: {safety:.0f}x, Threshold: {threshold:.2e}")
    print(sep)
    print(f" {'Stage':<12} | {'Layer':^5} | {'Rel-Err':^9} | {'Cosine':^9} | Verdict")
    print(dash)
    for stage, layer, rel_err, cosine, _floor, verdict in rows:
        layer_str = f"{layer:^5}" if layer is not None else "  -  "
        if verdict == "MISSING":
            print(f" {stage:<12} | {layer_str} | {'MISSING':^9} | {'MISSING':^9} | MISSING")
        else:
            print(f" {stage:<12} | {layer_str} | {rel_err:^9.2e} | {cosine:^9.6f} | {verdict}")
    print(dash)
    be_str = "YES" if embed_bit_exact else f"NO (max_abs_diff={embed_max_diff:.2e})"
    print(f" Embedding bit-exact: {be_str}")
    ff_str = "None" if first_fail is None else f"layer={first_fail[1]}, stage={first_fail[0]}"
    print(f" First failing (layer, stage): {ff_str}")
    all_ok = all(v in ("pass", "MISSING") for _, _, _, _, _, v in rows)
    print(f" All stages within floor: {'YES' if all_ok else 'NO'}")
    print(sep)


# ---------------------------------------------------------------------------
# Main alignment test
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HF_AVAILABLE, "HF step3p5 source files not found (set STEP35_HF_SRC)")
class TestStep3p5Alignment(unittest.TestCase):
    """Real HF eager oracle vs JAX naive fp32 forward — per-stage/layer error table.

    Asserts embedding bit-exact and every stage within the fp32 implementation
    floor. The first stage exceeding the floor localizes a wiring bug.
    """

    FLOOR = 2e-5
    SAFETY = 10.0
    NUM_TOKENS = 12

    def test_hf_jax_alignment(self) -> None:
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        # 1. Build shared fp32 weights (same seed → bit-identical for HF and JAX).
        weights_np = _build_checkpoint_np()

        # 2. Load HF model.
        tmpdir = _make_hf_tmpdir(weights_np)
        try:
            hf_model = _load_hf_model(tmpdir, weights_np)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # 3. Build JAX fp32 model and load same weights.
        jax_cfg = _make_jax_config()
        with jax.set_mesh(_mesh):
            jax_model = Step3p5ForCausalLM(
                jax_cfg, mesh=_mesh, dtype=jnp.float32, attn_impl="naive"
            )
        _load_jax_weights(jax_model, weights_np, jax_cfg)

        # 4. Identical input.
        input_ids_np = np.arange(self.NUM_TOKENS, dtype=np.int32)
        input_ids_torch = torch.from_numpy(input_ids_np).unsqueeze(0)  # [1, T]

        # 5. Capture HF activations.
        hf_caps = _capture_hf_activations(hf_model, input_ids_torch)

        # 6. Capture JAX activations.
        jax_caps: dict = {}
        fb = _make_forward_batch(input_ids_np)
        with jax.set_mesh(_mesh):
            _jax_instrumented_forward(jax_model, fb, jax_caps)

        # 7. Embedding bit-exact check (pure gather, no arithmetic).
        hf_embed = hf_caps[("embed", None)]
        jax_embed = jax_caps[("embed", None)]
        embed_max_diff = float(
            np.max(np.abs(hf_embed.astype(np.float64) - jax_embed.astype(np.float64)))
        )
        embed_bit_exact = embed_max_diff == 0.0

        # 8. Build ordered error table rows.
        stages_ordered: list[tuple] = (
            [("embed", None)]
            + [
                item
                for i in range(_NUM_LAYERS)
                for item in [("ln_in", i), ("attn", i), ("layer_out", i)]
            ]
            + [("final_norm", None)]
        )

        rows = []
        first_fail = None
        for key in stages_ordered:
            stage, layer = key
            if key not in hf_caps or key not in jax_caps:
                rows.append((stage, layer, float("nan"), float("nan"), self.FLOOR, "MISSING"))
                continue
            rel_err, cosine = _compute_error_metrics(hf_caps[key], jax_caps[key])
            exceeds = rel_err > self.FLOOR * self.SAFETY
            verdict = "EXCEED" if exceeds else "pass"
            if exceeds and first_fail is None:
                first_fail = key
            rows.append((stage, layer, rel_err, cosine, self.FLOOR, verdict))

        # 9. Print table.
        _print_alignment_table(
            rows, self.FLOOR, self.SAFETY, embed_bit_exact, first_fail, embed_max_diff
        )

        # 10. Assertions — full gate, no per-stage exemptions.
        self.assertTrue(
            embed_bit_exact,
            f"Embedding NOT bit-exact: max_abs_diff={embed_max_diff:.3e}. "
            "Weight load mismatch for embed_tokens.",
        )

        exceeded = [(s, lyr, re) for s, lyr, re, cs, fl, v in rows if v == "EXCEED"]
        self.assertEqual(
            exceeded,
            [],
            f"Stages exceeded fp32 floor (first = {first_fail}, the wiring-bug "
            f"localization point): {exceeded}. See table above.",
        )
        missing = [(s, lyr) for s, lyr, re, cs, fl, v in rows if v == "MISSING"]
        self.assertEqual(missing, [], f"Stages not captured: {missing}. See table above.")


@unittest.skipUnless(_HF_AVAILABLE, "HF step3p5 source files not found (set STEP35_HF_SRC)")
class TestStep3p5DecisionAgreement(unittest.TestCase):
    """E2: final-logits argmax + top-k overlap agree between HF and JAX per token.

    Complements numeric alignment (TestStep3p5Alignment) with a decision-layer check:
    greedy argmax and top-5 overlap must agree for all token positions.
    """

    NUM_TOKENS = 12
    TOP_K = 5

    def test_argmax_and_topk_agreement(self) -> None:
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        # 1. Build shared fp32 weights (same seed → bit-identical for HF and JAX).
        weights_np = _build_checkpoint_np()

        # 2. Load HF model and get all-token logits [T, vocab].
        tmpdir = _make_hf_tmpdir(weights_np)
        try:
            hf_model = _load_hf_model(tmpdir, weights_np)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        input_ids_np = np.arange(self.NUM_TOKENS, dtype=np.int32)
        input_ids_torch = torch.from_numpy(input_ids_np).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            hf_out = hf_model(input_ids=input_ids_torch)
        hf_logits = hf_out.logits[0].float().numpy()  # [T, vocab]

        # 3. Build JAX fp32 model, load weights, capture final_norm hidden states.
        jax_cfg = _make_jax_config()
        with jax.set_mesh(_mesh):
            jax_model = Step3p5ForCausalLM(
                jax_cfg, mesh=_mesh, dtype=jnp.float32, attn_impl="naive"
            )
        _load_jax_weights(jax_model, weights_np, jax_cfg)

        fb = _make_forward_batch(input_ids_np)
        jax_caps: dict = {}
        with jax.set_mesh(_mesh):
            _jax_instrumented_forward(jax_model, fb, jax_caps)

        # 4. Compute JAX all-token logits from final_norm hidden states.
        final_norm_np = jax_caps[("final_norm", None)]  # [T, hidden]
        lm_emb = np.asarray(jax_model.lm_head.embedding.value).astype(np.float32)  # [vocab, H]
        jax_logits = final_norm_np.astype(np.float32) @ lm_emb.T  # [T, vocab]

        T, vocab = hf_logits.shape
        assert jax_logits.shape == (
            T,
            vocab,
        ), f"Shape mismatch: {jax_logits.shape} vs {hf_logits.shape}"

        # 5. Per-token argmax agreement.
        hf_argmax = np.argmax(hf_logits, axis=-1)  # [T]
        jax_argmax = np.argmax(jax_logits, axis=-1)  # [T]
        argmax_matches = int(np.sum(hf_argmax == jax_argmax))

        mismatches = []
        for t in range(T):
            if hf_argmax[t] != jax_argmax[t]:
                gap = float(abs(hf_logits[t, hf_argmax[t]] - hf_logits[t, jax_argmax[t]]))
                mismatches.append((t, int(hf_argmax[t]), int(jax_argmax[t]), gap))

        # 6. Per-token top-K overlap.
        hf_topk = np.argsort(hf_logits, axis=-1)[:, -self.TOP_K :]
        jax_topk = np.argsort(jax_logits, axis=-1)[:, -self.TOP_K :]
        min_overlap = min(
            len(set(hf_topk[t].tolist()) & set(jax_topk[t].tolist())) for t in range(T)
        )

        print(
            f"\nE2 decision agreement: argmax {argmax_matches}/{T} tokens match, "
            f"min top-{self.TOP_K} overlap={min_overlap}/{self.TOP_K}"
        )
        if mismatches:
            print(f"  Argmax mismatches (token, hf_id, jax_id, logit_gap): {mismatches}")

        self.assertEqual(
            argmax_matches,
            T,
            f"Argmax mismatch at {len(mismatches)} token(s): {mismatches}",
        )
        self.assertGreaterEqual(
            min_overlap,
            self.TOP_K - 1,
            f"Top-{self.TOP_K} overlap dropped to {min_overlap} — decision divergence detected",
        )


if __name__ == "__main__":
    unittest.main()
