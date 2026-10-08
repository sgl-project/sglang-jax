"""DeepSeek V4 trunk: M-owned parameters, checkpoint partition and forward graph.

MTP keys are explicitly excluded. B owns attention execution and updates, E
owns MoE loading/execution, and H owns mHC arithmetic.
"""

from __future__ import annotations

import enum
import logging
import os
import re
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import (
    DeepseekV4LayerType,
    classify_layers,
    hash_moe_layer_flags,
)
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.model_loader.weights import WeightSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Disposition",
    "KeyFacts",
    "build_weight_mappings",
    "classify_checkpoint",
    "classify_key",
    "expected_trunk_keys",
]


class Disposition(enum.Enum):
    M_OWNED = "m_owned"
    E_OWNED = "e_owned"
    DROPPED = "dropped"


@dataclass(frozen=True)
class KeyFacts:
    disposition: Disposition
    reason: str
    layer: int | None = None


_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_MTP = re.compile(r"^mtp\.\d+\.")
_EXPERT = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")

# Root-level tensors, mapped to the model's own parameters.
_ROOT = {
    "embed.weight": "model.embed_tokens.embedding",
    "norm.weight": "model.norm.scale",
    "head.weight": "lm_head.embedding",
    "hc_head_fn": "model.hc_head_fn",
    "hc_head_base": "model.hc_head_base",
    "hc_head_scale": "model.hc_head_scale",
}

# Per-layer tensors present on every trunk layer.
_EVERY_LAYER = {
    "attn_norm.weight": "attn_norm.scale",
    "ffn_norm.weight": "ffn_norm.scale",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_base": "hc_attn_base",
    "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_base": "hc_ffn_base",
    "hc_ffn_scale": "hc_ffn_scale",
    "attn.attn_sink": "self_attn.attn_sink",
    "attn.q_norm.weight": "self_attn.q_norm.scale",
    "attn.kv_norm.weight": "self_attn.kv_norm.scale",
}

# FP8 linears on every trunk layer: weight plus a block-scale sibling.
_EVERY_LAYER_FP8 = {
    "attn.wq_a": "self_attn.wq_a",
    "attn.wq_b": "self_attn.wq_b",
    "attn.wkv": "self_attn.wkv",
    "attn.wo_a": "self_attn.wo_a",
    "attn.wo_b": "self_attn.wo_b",
}

# Present only where the layer keeps compressed history (ratio > 0). Unquantised.
_COMPRESSOR = {
    "attn.compressor.ape": "self_attn.compressor.ape",
    "attn.compressor.norm.weight": "self_attn.compressor.norm.scale",
    "attn.compressor.wkv.weight": "self_attn.compressor.wkv",
    "attn.compressor.wgate.weight": "self_attn.compressor.wgate",
}

# Present only on CSA (ratio 4) layers.
_INDEXER = {
    "attn.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight": "self_attn.indexer.compressor.norm.scale",
    "attn.indexer.compressor.wkv.weight": "self_attn.indexer.compressor.wkv",
    "attn.indexer.compressor.wgate.weight": "self_attn.indexer.compressor.wgate",
    "attn.indexer.weights_proj.weight": "self_attn.indexer.weights_proj",
}
_INDEXER_FP8 = {"attn.indexer.wq_b": "self_attn.indexer.wq_b"}

_SHARED_EXPERTS_FP8 = ("ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")

# Sharding contract for the target tree. M1.4 must build parameters that accept
# these; they are recorded here because the mapping is what pins them.
_REPLICATED = (None,)
_COLUMN = (None, "tensor")
_ROW = ("tensor", None)


def _layer_families(config):
    """Which per-layer families exist on each trunk layer.

    Derived from `classify_layers` and `hash_moe_layer_flags` -- the single
    classification -- so this cannot drift from C1's or M2's view of the layers.
    """
    types = classify_layers(config)
    hashed = hash_moe_layer_flags(config)
    out = []
    for layer_type, is_hash in zip(types, hashed, strict=True):
        out.append(
            {
                "compressor": layer_type is not DeepseekV4LayerType.SWA_ONLY,
                "indexer": layer_type is DeepseekV4LayerType.C4A,
                "hash_gate": bool(is_hash),
            }
        )
    return out


def expected_trunk_keys(config) -> set[str]:
    """Every trunk key this config implies, so coverage can be checked both ways.

    A mapping is wrong if it misses a key the checkpoint has *and* if it asks for a
    key the checkpoint does not have -- e.g. `compressor` on a SWA-only layer.
    """
    keys = set(_ROOT)
    experts = int(config.n_routed_experts)
    for layer, families in enumerate(_layer_families(config)):
        prefix = f"layers.{layer}."
        keys |= {prefix + name for name in _EVERY_LAYER}
        keys.add(prefix + "ffn.gate.weight")
        for stem in _EVERY_LAYER_FP8:
            keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        for stem in _SHARED_EXPERTS_FP8:
            keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        keys.add(prefix + ("ffn.gate.tid2eid" if families["hash_gate"] else "ffn.gate.bias"))
        if families["compressor"]:
            keys |= {prefix + name for name in _COMPRESSOR}
        if families["indexer"]:
            keys |= {prefix + name for name in _INDEXER}
            for stem in _INDEXER_FP8:
                keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        for expert in range(experts):
            for w in ("w1", "w2", "w3"):
                keys |= {
                    f"{prefix}ffn.experts.{expert}.{w}.weight",
                    f"{prefix}ffn.experts.{expert}.{w}.scale",
                }
    return keys


def classify_key(config, key: str) -> KeyFacts:
    """Disposition of one checkpoint key. Never returns "unknown" -- it raises."""
    if _MTP.match(key):
        return KeyFacts(
            Disposition.DROPPED,
            "MTP draft block; three are shipped despite num_nextn_predict_layers=1, and "
            "mtp.2 carries the DSpark heads (INFERENCE-84). Out of the first version.",
        )
    if key in _ROOT:
        return KeyFacts(Disposition.M_OWNED, "root tensor")

    match = _LAYER.match(key)
    if match is None:
        raise ValueError(f"unrecognised DeepSeek-V4 checkpoint key {key!r}")
    layer = int(match.group(1))
    tail = match.group(2)

    families = _layer_families(config)
    if not 0 <= layer < len(families):
        raise ValueError(
            f"key {key!r} names layer {layer}, outside the {len(families)}-layer trunk"
        )
    present = families[layer]

    expert = _EXPERT.match(tail)
    if expert is not None:
        if int(expert.group(1)) >= int(config.n_routed_experts):
            raise ValueError(f"key {key!r} names an expert beyond n_routed_experts")
        return KeyFacts(
            Disposition.E_OWNED,
            "routed expert pair loaded by E in the selected checkpoint format",
            layer,
        )

    if tail == "ffn.gate.weight":
        return KeyFacts(Disposition.E_OWNED, "MoE gate weight", layer)
    if tail in _EVERY_LAYER:
        return KeyFacts(Disposition.M_OWNED, "every-layer tensor", layer)
    for table, needed in (
        (_EVERY_LAYER_FP8, True),
        (_INDEXER_FP8, present["indexer"]),
    ):
        for stem in table:
            if tail in (f"{stem}.weight", f"{stem}.scale"):
                if not needed:
                    raise ValueError(f"key {key!r} is present but layer {layer} should not have it")
                return KeyFacts(Disposition.M_OWNED, "FP8 linear (block scale)", layer)
    for stem in _SHARED_EXPERTS_FP8:
        if tail in (f"{stem}.weight", f"{stem}.scale"):
            return KeyFacts(Disposition.E_OWNED, "shared MoE expert FP8 pair", layer)
    if tail in _COMPRESSOR:
        if not present["compressor"]:
            raise ValueError(f"key {key!r} on a SWA-only layer, which has no compressor")
        return KeyFacts(Disposition.M_OWNED, "compressor (unquantised)", layer)
    if tail in _INDEXER:
        if not present["indexer"]:
            raise ValueError(f"key {key!r} on a layer that is not CSA")
        return KeyFacts(Disposition.M_OWNED, "indexer (unquantised)", layer)
    if tail == "ffn.gate.bias":
        if present["hash_gate"]:
            raise ValueError(
                f"key {key!r} on a hash-routed layer; bias and tid2eid are mutually exclusive"
            )
        return KeyFacts(Disposition.E_OWNED, "noaux_tc correction bias", layer)
    if tail == "ffn.gate.tid2eid":
        if not present["hash_gate"]:
            raise ValueError(f"key {key!r} on a layer that does not route by token id")
        return KeyFacts(Disposition.E_OWNED, "hash routing table", layer)

    raise ValueError(f"unrecognised DeepSeek-V4 checkpoint key {key!r}")


def classify_checkpoint(config, keys) -> dict[str, KeyFacts]:
    """Classify every key, and require the partition to be total.

    Raises on an unrecognised key rather than skipping it -- the whole point is that
    "not loaded" can never be silent.
    """
    return {key: classify_key(config, key) for key in keys}


def _add_fp8_linear(mappings, hf_stem, target, *, sharding):
    """An FP8 linear: the weight plus its ``[out/128, in/128]`` block scale.

    Checkpoint weights are ``[out, in]`` and load into ``weight_q`` without a
    transpose, with the block scale as a sidecar -- the same shape the existing
    static-FP8 path in `models/deepseek_v3.py` uses.
    """
    quant = (sharding[1], sharding[0])
    mappings[f"{hf_stem}.weight"] = WeightSpec(
        target_path=f"{target}.weight_q", sharding=quant, transpose=False
    )
    mappings[f"{hf_stem}.scale"] = WeightSpec(
        target_path=f"{target}.weight_scale", sharding=quant, transpose=False
    )


def build_weight_mappings(config, mesh=None) -> dict[str, WeightSpec]:
    """Target paths for M-owned tensors only; E maps its own MoE parameters."""
    from sgl_jax.srt.layers.lm_head_parallel import weight_spec

    mappings: dict[str, WeightSpec] = {}
    for key, target in _ROOT.items():
        if key.startswith("hc_head"):
            # mHC gates are float32 and not ``[out, in]`` projections: no transpose,
            # replicated, and the dtype must not follow the activation dtype.
            mappings[key] = WeightSpec(target_path=target, sharding=_REPLICATED, transpose=False)
        elif key == "embed.weight":
            mappings[key] = WeightSpec(
                target_path=target, sharding=("tensor", None), transpose=False
            )
        elif key == "head.weight":
            mappings[key] = WeightSpec(
                target_path=target,
                sharding=tuple(weight_spec(getattr(config, "enable_dp_lm_head", False), mesh)),
                transpose=False,
            )
        else:
            mappings[key] = WeightSpec(target_path=target, sharding=_REPLICATED, transpose=False)

    for layer, families in enumerate(_layer_families(config)):
        prefix = f"layers.{layer}."
        target = f"model.layers.{layer}."
        for name, suffix in _EVERY_LAYER.items():
            mappings[prefix + name] = WeightSpec(
                target_path=target + suffix, sharding=_REPLICATED, transpose=False
            )
        for stem, suffix in _EVERY_LAYER_FP8.items():
            # wo_b reduces G*R back to hidden, so it is the row-parallel one.
            sharding = (
                _ROW
                if stem == "attn.wo_b"
                else ((None, None) if stem in ("attn.wq_a", "attn.wkv") else _COLUMN)
            )
            _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=sharding)
        if families["compressor"]:
            for name, suffix in _COMPRESSOR.items():
                mappings[prefix + name] = WeightSpec(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name else (None, None),
                    transpose=False,
                )
        if families["indexer"]:
            for name, suffix in _INDEXER.items():
                mappings[prefix + name] = WeightSpec(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name else (None, None),
                    transpose=False,
                )
            for stem, suffix in _INDEXER_FP8.items():
                _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=(None, None))
    return mappings


def _static_fp8(config):
    quant = getattr(config, "quantization_config", None)
    return quant is not None and getattr(quant, "is_static_checkpoint", False)


def _linear(input_size, output_size, mesh, dtype, axes, name, quantized=False):
    """Build the checkpoint's resident representation before eval_shape loading."""
    if not quantized:
        return LinearBase(
            input_size=input_size,
            output_size=output_size,
            mesh=mesh,
            params_dtype=dtype,
            kernel_axes=axes,
            use_bias=False,
            scope_name=name,
        )
    from sgl_jax.srt.layers.linear import QuantizedLinear

    # Non-expert FP8 uses K128 block scales; expert MXFP4 is converted separately.
    if input_size % 128 or output_size % 128:
        raise ValueError(f"static V4 FP8 linear {name} requires dimensions divisible by 128")
    if axes[0] is not None and (input_size // 128) % mesh.shape[axes[0]]:
        # A block cannot span two devices. Match QuantizedLinear.from_linear.
        axes = (None, axes[1])
    return QuantizedLinear(
        weight_q=jnp.zeros(
            (output_size, input_size), jnp.float8_e4m3fn, out_sharding=P(axes[1], axes[0])
        ),
        weight_scale=jnp.zeros(
            (input_size // 128, 1, output_size), jnp.float32, out_sharding=P(axes[0], None, axes[1])
        ),
        bias=None,
        # W8A8 when requested: the blockwise qmm quantizes activations per token
        # (GPU serving runs every dense fp8 GEMM w8a8; default here stays w8a16).
        activation_dtype=(
            jnp.float8_e4m3fn
            if _W8A8_DENSE and (_W8A8_DENSE_NAMES is None or name in _W8A8_DENSE_NAMES)
            else None
        ),
        mesh=mesh,
        kernel_axes=axes,
        params_dtype=dtype,
        weight_block_size=(128, 128),
        # As in ModelConfig's static-FP8 loader: scales came from checkpoint,
        # not the online narrow-N quantizer the guard protects against.
        allow_narrow_n_blockwise=True,
        scope_name=name,
    )


def _checkpoint_matrix(linear):
    """Weight in [out,in] layout, used only by the grouped output contraction."""
    if hasattr(linear, "weight_q"):
        weight = linear.weight_q.value.astype(jnp.float32)
        # Kernel-ready [K/128,1,N] -> [N,K]. TP follows N for grouped wo_a.
        scale = jnp.repeat(linear.weight_scale.value[:, 0, :], 128, axis=0).T
        return weight * scale
    return linear.weight.value.T


# ``DSV4_SP_NORM_BEFORE_GATHER=1``: under sequence parallelism apply the sublayer
# RMSNorm (per row) and the bf16 cast on the T/tp rows and all-gather the result,
# instead of gathering the raw stream and normalising all T rows on every device.
# Same values; the gather carries bf16 instead of the stream dtype.
_SP_NORM_BEFORE_GATHER = os.environ.get("DSV4_SP_NORM_BEFORE_GATHER", "1") == "1"
# ``DSV4_LOWRANK_AG=1``: on CSA layers under sequence parallelism, project q_lora / kv /
# indexer weights on the local T/tp rows and all-gather those (1024+512+64 columns)
# instead of the 4096-wide hidden; the compressors then run on local rows with a
# ppermute halo (needs DSV4_COMPRESSOR_ROW_SHARD=1). HCA layers keep the full gather.
_LOWRANK_AG = os.environ.get("DSV4_LOWRANK_AG", "1") == "1"
# ``DSV4_HCA_FUSED_PROJ=1`` (default): build the HCA compressor's fused ``[Wkv|Wgate]^T`` bf16
# projection once after loading instead of converting the f32 gate weight and
# concatenating on every step in every HCA layer.
_HCA_FUSED_PROJ = os.environ.get("DSV4_HCA_FUSED_PROJ", "1") == "1"
_WGATE_F32 = os.environ.get("DSV4_COMPRESSOR_WGATE_F32", "0") == "1"
# ``DSV4_W8A8_DENSE=1``: fp8 activations for the dense fp8 linears (weights are
# already fp8); default keeps bf16 activations.
_W8A8_DENSE = os.environ.get("DSV4_W8A8_DENSE", "1") == "1"
# ``DSV4_W8A8_DENSE_NAMES=wq_b,wo_b``: restrict fp8 activations to these linears (an empty
# value means all of them).
_W8A8_DENSE_NAMES_ENV = os.environ.get(
    "DSV4_W8A8_DENSE_NAMES", "wq_a,wkv,wo_a,wo_b,indexer_wq_b,gate_proj,up_proj,down_proj"
)
_W8A8_DENSE_NAMES = (
    frozenset(x.strip() for x in _W8A8_DENSE_NAMES_ENV.split(",") if x.strip())
    if _W8A8_DENSE_NAMES_ENV
    else None
)


# DSV4_ROPE_CACHE_LANE_PAD=1: store the [max_position, cos|sin] tables with the lane axis
# padded to a multiple of 128. With 64 lanes XLA picks a column-major layout for the jit
# parameter and inserts a full-table relayout copy (2 x 268 MB) at the top of every step
# (measured 0.71 ms/step on v7x). Consumers slice the first ``rope_head_dim`` lanes.
_ROPE_CACHE_LANE_PAD = os.environ.get("DSV4_ROPE_CACHE_LANE_PAD", "1") == "1"


def _split_rope_cache(cache, rope_dim):
    """``[N, >=rope_dim]`` cos|sin table -> (cos, sin) halves, materialised once (not per step)."""
    half = rope_dim // 2
    return cache[:, :half], cache[:, half : 2 * half]


def _rope_cache(config, ratio):
    from sgl_jax.srt.layers.attention.dsv4.rope import build_dsv4_rope

    rope = build_dsv4_rope(config, ratio, dtype=jnp.float32)
    cos, sin = rope._compute_cos_sin(jnp.arange(config.max_position_embeddings, dtype=jnp.int32))
    cache = jnp.concatenate((cos, sin), axis=-1)
    if _ROPE_CACHE_LANE_PAD and cache.shape[-1] % 128:
        cache = jnp.pad(cache, ((0, 0), (0, -cache.shape[-1] % 128)))
    return cache


class DeepseekV4Compressor(nnx.Module):
    """All compressor parameters stay in the modeling file, in checkpoint layout."""

    def __init__(self, config, head_dim, ratio, dtype):
        from sgl_jax.srt.layers.layernorm import RMSNorm

        width = head_dim * (2 if ratio == 4 else 1)
        self.wkv = nnx.Param(
            jnp.zeros((width, config.hidden_size), dtype, out_sharding=P(None, None))
        )
        # The checkpoint stores wgate in BF16; keeping the parameter in BF16 is exact
        # and halves the bytes every compress projection streams per layer per step
        # (the projections cast to f32 / HIGHEST themselves). DSV4_COMPRESSOR_WGATE_F32=1
        # restores the previous f32 storage for A/B.
        self.wgate = nnx.Param(
            jnp.zeros(
                (width, config.hidden_size),
                jnp.float32 if _WGATE_F32 else dtype,
                out_sharding=P(None, None),
            )
        )
        self.ape = nnx.Param(jnp.zeros((ratio, width), jnp.float32, out_sharding=P(None, None)))
        self.norm = RMSNorm(head_dim, epsilon=config.rms_norm_eps, param_dtype=jnp.float32)
        self.ratio = ratio

    def prepare_fused_projection(self, mesh):
        """Materialise the HCA kernels' fused ``[hidden, 2*D]`` bf16 projection once.

        Without it ``fused_projection_weight`` rebuilds it in every HCA layer on every
        step: an f32->bf16 convert of ``wgate`` (8 MB prefetched through VMEM), a
        concatenation and a transpose.
        """
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            fused = jnp.concatenate(
                (self.wkv.value.astype(jnp.bfloat16), self.wgate.value.astype(jnp.bfloat16)),
                axis=0,
            ).T
        self.fused_proj = nnx.Variable(fused)

    def weights(self, cache, halves=None):
        from sgl_jax.srt.layers.attention.dsv4.execution import CompressorWeights

        cos_table, sin_table = halves if halves is not None else (None, None)
        fused = getattr(self, "fused_proj", None)
        return CompressorWeights(
            self.wkv.value,
            self.wgate.value,
            self.ape.value,
            self.norm.scale.value,
            cache,
            cos_table,
            sin_table,
            None if fused is None else fused.value,
        )


class DeepseekV4Indexer(nnx.Module):
    def __init__(self, config, mesh, dtype):
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.rope_head_dim = config.qk_rope_head_dim
        self.weight_scale = (self.head_dim * self.num_heads) ** -0.5
        self.wq_b = _linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            mesh,
            dtype,
            (None, None),
            "indexer_wq_b",
            _static_fp8(config),
        )
        self.weights_proj = nnx.Param(
            jnp.zeros((self.num_heads, config.hidden_size), dtype, out_sharding=P(None, None))
        )
        self.compressor = DeepseekV4Compressor(config, self.head_dim, 4, dtype)

    def weights_from_hidden(self, hidden):
        """``[rows, H_idx]`` f32 per-head indexer weights (row-local, no communication)."""
        return jnp.dot(hidden.astype(jnp.float32), self.weights_proj.value.T.astype(jnp.float32))

    def project(self, q_lora, weights, cos, sin, cache, dtype):
        """Indexer inputs from an already-gathered ``q_lora`` and the raw weights."""
        from sgl_jax.srt.layers.attention.dsv4.execution import IndexerInputs
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        q, _ = self.wq_b(q_lora)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        q = apply_dsv4_partial_rope(
            q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
        ).astype(dtype)
        return IndexerInputs(q, weights * self.weight_scale, self.compressor.weights(cache))

    def __call__(self, hidden, q_lora, cos, sin, cache):
        return self.project(q_lora, self.weights_from_hidden(hidden), cos, sin, cache, hidden.dtype)


class DeepseekV4Attention(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mesh = mesh
        self.layer_id = layer_id
        self.ratio = int(config.compress_ratios[layer_id])
        if self.ratio not in (0, 4, 128):
            raise ValueError(f"V4 runtime does not support physical compression ratio {self.ratio}")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.num_groups = config.o_groups
        self.scaling = config.head_dim**-0.5
        self.norm_eps = config.rms_norm_eps
        self.index_topk = config.index_topk
        self.dtype = dtype
        if self.num_heads % self.num_groups or self.num_groups % mesh.shape["tensor"]:
            raise ValueError(
                "V4 heads must divide into output groups, and groups must divide by TP"
            )
        self.wq_a = _linear(
            config.hidden_size,
            config.q_lora_rank,
            mesh,
            dtype,
            (None, None),
            "wq_a",
            _static_fp8(config),
        )
        self.q_norm = RMSNorm(config.q_lora_rank, epsilon=self.norm_eps, dtype=dtype)
        self.wq_b = _linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            mesh,
            dtype,
            (None, "tensor"),
            "wq_b",
            _static_fp8(config),
        )
        self.wkv = _linear(
            config.hidden_size, self.head_dim, mesh, dtype, (None, None), "wkv", _static_fp8(config)
        )
        self.kv_norm = RMSNorm(self.head_dim, epsilon=self.norm_eps, dtype=dtype)
        self.attn_sink = nnx.Param(
            jnp.zeros((self.num_heads,), jnp.float32, out_sharding=P("tensor"))
        )
        self.wo_a = _linear(
            self.num_heads // self.num_groups * self.head_dim,
            self.num_groups * config.o_lora_rank,
            mesh,
            dtype,
            (None, "tensor"),
            "wo_a",
            _static_fp8(config),
        )
        self.wo_b = _linear(
            self.num_groups * config.o_lora_rank,
            config.hidden_size,
            mesh,
            dtype,
            ("tensor", None),
            "wo_b",
            _static_fp8(config),
        )
        self.compressor = (
            DeepseekV4Compressor(config, self.head_dim, self.ratio, dtype) if self.ratio else None
        )
        self.indexer = DeepseekV4Indexer(config, mesh, dtype) if self.ratio == 4 else None

    def prepare_grouped_wo_a(self):
        """Materialise the grouped, dequantised wo_a once after loading.

        The forward used to dequantise (scale broadcast + multiply) and regroup wo_a on
        every step; at bs=1 that was ~0.9 ms per decode step across the layers.
        """
        from sgl_jax.srt.layers.attention.dsv4.o_projection import (
            fuse_wo_a_weights,
            group_wo_a,
            use_fused_wo_a,
        )

        with jax.sharding.use_abstract_mesh(self.mesh.abstract_mesh):
            weights = group_wo_a(
                _checkpoint_matrix(self.wo_a),
                num_groups=self.num_groups,
                out_sharding=NamedSharding(self.mesh, P("tensor", None, None)),
            )
            if use_fused_wo_a():
                # The fused kernel wants [8*head_dim, G*R]; keep only that copy.
                self.wo_a_fused = nnx.Param(fuse_wo_a_weights(weights, mesh=self.mesh))
                return
        self.wo_a_grouped = nnx.Param(weights)

    def __call__(
        self,
        hidden,
        batch,
        pools,
        rope_cache,
        rope_halves=None,
        wo_out_sharding=None,
        sp_local=False,
    ):
        from sgl_jax.srt.layers.attention.dsv4.o_projection import group_wo_a
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        indexer_weights = None
        if sp_local:
            # ``hidden`` is this device's T/tp rows (DSV4_LOWRANK_AG): project locally,
            # gather the narrow results once. Row-wise norms commute with the gather.
            rows_sh = NamedSharding(self.mesh, P("tensor", None))
            q_lora_l, _ = self.wq_a(hidden, out_sharding=rows_sh)
            q_lora_l = self.q_norm(q_lora_l)
            kv_l, _ = self.wkv(hidden, out_sharding=rows_sh)
            kv_l = self.kv_norm(kv_l)
            parts = [q_lora_l.astype(self.dtype), kv_l.astype(self.dtype)]
            if self.indexer is not None:
                # f32 weights ride along as two bf16 halves (exact round trip)
                w32 = jax.lax.bitcast_convert_type(
                    self.indexer.weights_from_hidden(hidden), jnp.uint32
                )
                parts.append(
                    jax.lax.bitcast_convert_type((w32 >> 16).astype(jnp.uint16), jnp.bfloat16)
                )
                parts.append(
                    jax.lax.bitcast_convert_type((w32 & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
                )
            packed = _sp_gather(self.mesh, jnp.concatenate(parts, axis=-1))
            widths = [p.shape[-1] for p in parts]
            cuts = np.cumsum(widths)[:-1].tolist()
            pieces = jnp.split(packed, cuts, axis=-1)
            q_lora, kv = pieces[0], pieces[1]
            if self.indexer is not None:
                hi = jax.lax.bitcast_convert_type(pieces[2], jnp.uint16).astype(jnp.uint32)
                lo = jax.lax.bitcast_convert_type(pieces[3], jnp.uint16).astype(jnp.uint32)
                indexer_weights = jax.lax.bitcast_convert_type((hi << 16) | lo, jnp.float32)
        else:
            q_lora, _ = self.wq_a(hidden)
            q_lora = self.q_norm(q_lora)
        q, _ = self.wq_b(q_lora)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        # V4 normalizes q again per head after wq_b, with no learned weight.
        q = (
            q.astype(jnp.float32)
            * jax.lax.rsqrt(
                jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True) + self.norm_eps
            )
        ).astype(self.dtype)
        if not sp_local:
            kv, _ = self.wkv(hidden)
            kv = self.kv_norm(kv)
        positions = batch.positions
        selected = rope_cache.at[positions].get(
            out_sharding=NamedSharding(self.mesh, P("data", None))
        )
        cos, sin = jnp.split(selected[:, : self.rope_head_dim], 2, axis=-1)
        q = apply_dsv4_partial_rope(
            q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
        ).astype(self.dtype)
        kv = apply_dsv4_partial_rope(kv, cos, sin, rope_head_dim=self.rope_head_dim).astype(
            self.dtype
        )
        if self.indexer is None:
            indexer = None
        elif sp_local:
            indexer = self.indexer.project(
                q_lora, indexer_weights, cos, sin, rope_cache, self.dtype
            )
        else:
            indexer = self.indexer(hidden, q_lora, cos, sin, rope_cache)
        output, updates = batch.attn_backend(
            q,
            kv,
            kv,
            self,
            batch,
            pools.token_to_kv_pool,
            compressor_state_pool=pools.compressor_state_pool,
            compressor_input=hidden,
            compressor_input_local=sp_local,
            compressor=(
                None
                if self.compressor is None
                else self.compressor.weights(rope_cache, rope_halves)
            ),
            indexer=indexer,
            attention_sink=self.attn_sink.value,
            rope_head_dim=self.rope_head_dim,
            norm_eps=self.norm_eps,
            index_topk=self.index_topk,
        )
        if getattr(self, "wo_a_fused", None) is not None:
            from sgl_jax.srt.layers.attention.dsv4.o_projection import (
                fused_wo_a_projection,
            )

            reduced = fused_wo_a_projection(
                output,
                cos,
                sin,
                self.wo_a_fused.value,
                mesh=self.mesh,
                rope_head_dim=self.rope_head_dim,
                dtype=self.dtype,
            )
            output, _ = self.wo_b(reduced, out_sharding=wo_out_sharding)
            return output, updates
        output = apply_dsv4_partial_rope(
            output, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim, inverse=True
        )
        grouped = jax.lax.reshape(
            output,
            (output.shape[0], self.num_groups, self.num_heads // self.num_groups * self.head_dim),
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        if getattr(self, "wo_a_grouped", None) is not None:
            weights = self.wo_a_grouped.value
        else:
            weights = group_wo_a(
                _checkpoint_matrix(self.wo_a),
                num_groups=self.num_groups,
                out_sharding=NamedSharding(self.mesh, P("tensor", None, None)),
            )
        reduced = jnp.einsum("tgd,gdr->tgr", grouped, weights, preferred_element_type=jnp.float32)
        reduced = reduced.reshape(reduced.shape[0], -1).astype(self.dtype)
        output, _ = self.wo_b(reduced, out_sharding=wo_out_sharding)
        return output, updates


# ``DSV4_SEQ_PARALLEL=1``: sequence parallelism for the mHC / norm / residual work.
# Today every TP rank runs the mHC pre/post/seam kernels, the layer norms and the
# residual adds over all T rows (P("data", ...) = replicated across "tensor"):
# ~55 ms of an 8K prefill step on v7x that each of the 8 ranks repeats. With the
# flag the streams live row-sharded over the tensor axis (P("tensor", ...)), the
# row-parallel wo_b returns a reduce-scatter instead of an all-reduce, and the
# hidden states are all-gathered only where a full row set is needed (the
# attention projections, the MoE dispatch, the final norm). Same bytes on the
# ICI (RS + AG == AR); the replicated compute shrinks by the tensor size. Only
# prefill buckets take the path (rows >= DSV4_SEQ_PARALLEL_MIN_TOKENS and
# divisible by the tensor axis); decode keeps the replicated form.
_SEQ_PARALLEL = os.environ.get("DSV4_SEQ_PARALLEL", "1") == "1"
_SEQ_PARALLEL_MIN_TOKENS = int(os.environ.get("DSV4_SEQ_PARALLEL_MIN_TOKENS", "256"))


def _sp_active(mesh, rows: int) -> bool:
    if not _SEQ_PARALLEL:
        return False
    tp = int(mesh.shape.get("tensor", 1))
    return tp > 1 and rows >= _SEQ_PARALLEL_MIN_TOKENS and rows % tp == 0


def _sp_gather(mesh, x):
    """Row-sharded ``[T/tp, ...]`` -> replicated ``[T, ...]`` (all-gather over tensor)."""
    return jax.sharding.reshard(x, NamedSharding(mesh, P("data", *([None] * (x.ndim - 1)))))


def _sp_rows(mesh, x):
    """Replicated ``[T, ...]`` -> row-sharded over the tensor axis (a local slice)."""
    return jax.sharding.reshard(x, NamedSharding(mesh, P("tensor", *([None] * (x.ndim - 1)))))


class DeepseekV4DecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        from sgl_jax.srt.configs.deepseek_v4 import mhc_param_shapes
        from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC
        from sgl_jax.srt.layers.deepseek_v4_moe import DeepseekV4MoE
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mhc = DeepseekV4MHC(config)
        self.mesh = mesh
        for kind in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                shape = mhc_param_shapes(config)[part]
                setattr(
                    self,
                    f"hc_{kind}_{part}",
                    nnx.Param(
                        jnp.zeros(shape, jnp.float32, out_sharding=P(*([None] * len(shape))))
                    ),
                )
        self.attn_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.self_attn = DeepseekV4Attention(config, mesh, layer_id, dtype)
        self.mlp = DeepseekV4MoE(config, mesh, layer_id, dtype)
        self.dtype = dtype

    def _mhc_pre(self, streams, fn, base, scale):
        if self.mhc.backend != "pallas":
            return self.mhc.pre(streams, fn, base, scale)
        row = "tensor" if _sp_active(self.mesh, streams.shape[0] * self._sp_tp(streams)) else "data"
        specs = (P(row, None), P(row, None), P(row, None, None))
        compute = jax.shard_map(
            self.mhc.pre,
            mesh=None,
            in_specs=(P(row, None, None), P(), P(), P()),
            out_specs=specs,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute,
            axes=self.mesh.axis_names,
            out_sharding=tuple(NamedSharding(self.mesh, spec) for spec in specs),
        )
        return compute(streams, fn, base, scale)

    def _mhc_post(self, output, residual, post, comb):
        if self.mhc.backend != "pallas":
            return self.mhc.post(output, residual, post, comb)
        row = (
            "tensor" if _sp_active(self.mesh, residual.shape[0] * self._sp_tp(residual)) else "data"
        )
        spec = P(row, None, None)
        compute = jax.shard_map(
            self.mhc.post,
            mesh=None,
            in_specs=(P(row, None), spec, P(row, None), spec),
            out_specs=spec,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute, axes=self.mesh.axis_names, out_sharding=NamedSharding(self.mesh, spec)
        )
        return compute(output, residual, post, comb)

    def _sp_tp(self, x):
        # Global arrays carry the global row count whether replicated or sharded; the
        # gate below only needs T itself, so this is 1 (kept for readability).
        return 1

    def _lowrank_attn(self, hidden, batch) -> bool:
        """DSV4_LOWRANK_AG applies on CSA layers, under SP, for single-request batches
        (the row-local compressors have no multi-request path)."""
        if not _LOWRANK_AG or getattr(self.self_attn, "ratio", None) != 4:
            return False
        tp = int(self.mesh.shape.get("tensor", 1))
        if not _sp_active(self.mesh, hidden.shape[0] * tp):
            return False
        if os.environ.get("DSV4_COMPRESSOR_ROW_SHARD", "1") != "1":
            raise ValueError("DSV4_LOWRANK_AG needs DSV4_COMPRESSOR_ROW_SHARD=1")
        return int(batch.seq_lens.shape[0]) == 1

    def _sp_full(self, hidden):
        """All-gather a row-sharded pre-output before the attention / MoE projections."""
        if _sp_active(self.mesh, hidden.shape[0]):
            return _sp_gather(self.mesh, hidden)
        return hidden

    def _sp_shard(self, x):
        """Bring a sublayer output onto the streams' row sharding (no-op if already there)."""
        if _sp_active(self.mesh, x.shape[0]):
            return _sp_rows(self.mesh, x)
        return x

    def _sp_wo_sharding(self, rows: int):
        if _sp_active(self.mesh, rows):
            return NamedSharding(self.mesh, P("tensor", None))
        return None

    def __call__(self, streams, batch, pools, rope_cache, rope_halves=None):
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_attn_fn.value, self.hc_attn_base.value, self.hc_attn_scale.value
        )
        lowrank = self._lowrank_attn(hidden, batch)
        if lowrank:
            # DSV4_LOWRANK_AG: hand the attention the local rows; it gathers q_lora/kv
            full_rows = hidden.shape[0] * int(self.mesh.shape["tensor"])
            attn_in = self.attn_norm(hidden.astype(self.dtype))
        elif _SP_NORM_BEFORE_GATHER:
            # RMSNorm is per row: normalise (and cast) the T/tp rows, then gather bf16.
            hidden_full = self._sp_full(self.attn_norm(hidden.astype(self.dtype)))
            attn_in = hidden_full
            full_rows = hidden_full.shape[0]
        else:
            hidden_full = self._sp_full(hidden)
            attn_in = self.attn_norm(hidden_full.astype(self.dtype))
            full_rows = hidden_full.shape[0]
        attn, updates = self.self_attn(
            attn_in,
            batch,
            pools,
            rope_cache,
            rope_halves,
            wo_out_sharding=self._sp_wo_sharding(full_rows),
            sp_local=lowrank,
        )
        attn = self._sp_shard(attn)
        streams = self._mhc_post(attn, streams, post, comb).astype(self.dtype)
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_ffn_fn.value, self.hc_ffn_base.value, self.hc_ffn_scale.value
        )
        if _SP_NORM_BEFORE_GATHER:
            rows = self.ffn_norm(hidden.astype(self.dtype))
            hidden_full = self._sp_full(rows)
            ffn_in = hidden_full
        else:
            hidden_full = self._sp_full(hidden)
            ffn_in = self.ffn_norm(hidden_full.astype(self.dtype))
        ffn, ids = self.mlp(
            ffn_in,
            batch.input_ids,
            token_valid_mask=batch.get_token_valid_mask(hidden_full.shape[0]),
            dispatch_info=batch.expert_location_metadata,
            out_sharding=NamedSharding(self.mesh, P("data", None)),
            output_sharding=self._sp_wo_sharding(hidden_full.shape[0]),
            return_expert_ids=True,
        )
        ffn = self._sp_shard(ffn)
        streams = self._mhc_post(ffn, streams, post, comb).astype(self.dtype)
        return streams, updates, ids


class DeepseekV4Model(nnx.Module):
    def __init__(self, config, mesh, dtype):
        from sgl_jax.srt.configs.deepseek_v4 import mhc_param_shapes
        from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC
        from sgl_jax.srt.layers.embeddings import Embed
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mhc = DeepseekV4MHC(config)
        self.mesh = mesh
        self.dtype = dtype
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.List(
            [
                DeepseekV4DecoderLayer(config, mesh, i, dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        for part in ("fn", "base", "scale"):
            shape = mhc_param_shapes(config)["head_" + part]
            setattr(
                self,
                "hc_head_" + part,
                nnx.Param(jnp.zeros(shape, jnp.float32, out_sharding=P(*([None] * len(shape))))),
            )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.rope_plain = nnx.Variable(_rope_cache(config, 0))
        self.rope_compressed = nnx.Variable(_rope_cache(config, 4))
        cos, sin = _split_rope_cache(self.rope_compressed.value, config.qk_rope_head_dim)
        self.rope_compressed_cos = nnx.Variable(cos)
        self.rope_compressed_sin = nnx.Variable(sin)

    def _collapse_head(self, streams):
        params = (self.hc_head_fn.value, self.hc_head_base.value, self.hc_head_scale.value)
        if self.mhc.backend != "pallas":
            return self.mhc.collapse_head(streams, *params)
        row = "tensor" if _sp_active(self.mesh, streams.shape[0]) else "data"
        spec = P(row, None)
        compute = jax.shard_map(
            self.mhc.collapse_head,
            mesh=None,
            in_specs=(P(row, None, None), P(), P(), P()),
            out_specs=spec,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute, axes=self.mesh.axis_names, out_sharding=NamedSharding(self.mesh, spec)
        )
        return compute(streams, *params)

    def __call__(self, batch, pools):
        from sgl_jax.srt.layers.deepseek_v4_mhc import expand_streams

        hidden = self.embed_tokens(batch.input_ids)
        if batch.input_embedding is not None:
            hidden = batch.input_embedding
        streams = expand_streams(hidden, self.mhc.hc_mult).astype(hidden.dtype)
        if _sp_active(self.mesh, streams.shape[0]):
            streams = _sp_rows(self.mesh, streams)
        updates, ids = {}, []
        for i, layer in enumerate(self.layers):
            cache = self.rope_compressed if layer.self_attn.ratio else self.rope_plain
            halves = (
                (self.rope_compressed_cos.value, self.rope_compressed_sin.value)
                if layer.self_attn.ratio
                else None
            )
            streams, updates[i], route_ids = layer(streams, batch, pools, cache.value, halves)
            ids.append(route_ids)
        hidden = self._collapse_head(streams)
        if _sp_active(self.mesh, hidden.shape[0]):
            hidden = _sp_gather(self.mesh, hidden)
        return (
            self.norm(hidden.astype(self.dtype)),
            batch.attn_backend.pack_pool_updates(
                updates, pools.token_to_kv_pool, pools.compressor_state_pool
            ),
            ids,
        )


class DeepseekV4ForCausalLM(nnx.Module):
    owns_quantization_structure = True

    @classmethod
    def patch_model_config(cls, mc):
        from sgl_jax.srt.configs.model_config import AttentionArch

        mc.attention_arch = AttentionArch.MLA
        mc.head_dim = mc.hf_text_config.head_dim

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        from sgl_jax.srt.layers.embeddings import ParallelLMHead
        from sgl_jax.srt.layers.logits_processor import LogitsProcessor

        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = DeepseekV4Model(config, mesh, dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        hidden, updates, ids = self.model(forward_batch, memory_pools)
        output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        return output, updates, True, ids

    def load_weights(self, model_config):
        """Load M's tensors and hand the exact MoE partition to E."""
        from sgl_jax.srt.layers.deepseek_v4_moe_loader import (
            STATIC_EXPERT_FORMAT,
            expected_moe_keys,
        )
        from sgl_jax.srt.model_loader.deepseek_v4_static import (
            CONFIG_KEY,
            FORMAT,
            validate_static_checkpoint,
        )
        from sgl_jax.srt.model_loader.weights.source import LocalSource

        if getattr(model_config, "_dummy_mode", False):
            state = nnx.state(self, nnx.Param)
            for i, (path, variable) in enumerate(state.flat_state()):
                old = variable.value
                mesh = self.mesh
                if "experts" in path:
                    mesh = self.model.layers[int(path[2])].mlp.experts.moe_mesh
                if getattr(model_config, "_abstract_mode", False):
                    # DummyLoader traces post-load preparation during CPU AOT.
                    # Expert parameters use their own mesh even while tracing.
                    spec = (
                        old.sharding.spec
                        if isinstance(old, jax.ShapeDtypeStruct)
                        else jax.typeof(old).sharding.spec
                    )
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        variable.value = jnp.zeros(
                            old.shape, old.dtype, out_sharding=NamedSharding(mesh, spec)
                        )
                    continue
                sharding = NamedSharding(mesh, old.sharding.spec)

                def initialize(index, shape=old.shape, dtype=old.dtype, seed=i):
                    dims = tuple(len(range(*sl.indices(n))) for sl, n in zip(index, shape))
                    rng = np.random.default_rng(seed)
                    return (rng.standard_normal(dims) * 0.02).astype(dtype)

                variable.value = jax.make_array_from_callback(old.shape, sharding, initialize)
            nnx.update(self, state)
            for layer in self.model.layers:
                if layer.mlp.is_hash_layer:
                    table = (
                        np.arange(self.config.vocab_size)[:, None]
                        + np.arange(self.config.num_experts_per_tok)
                    ) % self.config.n_routed_experts
                    layer.mlp.load_hash_table(table)
        else:
            expert_format = getattr(self.config, CONFIG_KEY, None)
            if expert_format not in (None, FORMAT) or FORMAT != STATIC_EXPERT_FORMAT:
                raise ValueError(f"Unsupported V4 expert format: {expert_format}")
            if expert_format == FORMAT:
                if not _static_fp8(self.config):
                    raise ValueError(
                        "static expert-FP8 export requires static FP8 model parameters"
                    )
                validate_static_checkpoint(model_config.model_path)
            with LocalSource(model_config) as source:
                info = source.metadata
                if any(len(entries) != 1 for entries in info.values()):
                    raise ValueError("V4 tensors must occur exactly once across checkpoint shards")
                facts = classify_checkpoint(self.config, info)
                required = expected_trunk_keys(self.config)
                missing = required - info.keys()
                if missing:
                    raise ValueError(
                        f"V4 checkpoint missing {len(missing)} trunk tensors: {sorted(missing)[:8]}"
                    )
                m_keys = {
                    key for key, fact in facts.items() if fact.disposition is Disposition.M_OWNED
                }
                e_keys = {
                    key for key, fact in facts.items() if fact.disposition is Disposition.E_OWNED
                }
                if m_keys | e_keys != required or m_keys & e_keys:
                    raise ValueError("V4 checkpoint ownership does not cover the required trunk")
                started = time.monotonic()
                m_consumed = self._load_regular_weights(info)
                logger.info(
                    "Loaded DeepSeek V4 M-owned parameters in %.1fs", time.monotonic() - started
                )
                e_consumed = set()
                for layer in self.model.layers:
                    layer_keys = expected_moe_keys(layer.mlp)
                    assigned = {key: info[key] for key in layer_keys}
                    if layer_keys != {
                        key for key in e_keys if facts[key].layer == layer.mlp.layer_id
                    }:
                        raise ValueError(f"V4 layer {layer.mlp.layer_id} MoE ownership mismatch")
                    report = layer.mlp.load_owned_weights(assigned, expert_format=expert_format)
                    if e_consumed & report.consumed_keys:
                        raise ValueError("V4 MoE source tensor was consumed by two layers")
                    e_consumed.update(report.consumed_keys)
                if m_consumed != m_keys or e_consumed != e_keys or m_consumed & e_consumed:
                    raise ValueError("V4 M/E consumed-key report does not match source ownership")
                if m_consumed | e_consumed != required:
                    raise ValueError("V4 required checkpoint tensor was not consumed")
        # eval_shape creates placeholders for these non-parameter tables too.
        with jax.sharding.use_abstract_mesh(self.mesh.abstract_mesh):
            self.model.rope_plain.value = _rope_cache(self.config, 0)
            self.model.rope_compressed.value = _rope_cache(self.config, 4)
            cos, sin = _split_rope_cache(
                self.model.rope_compressed.value, self.config.qk_rope_head_dim
            )
            self.model.rope_compressed_cos.value = cos
            self.model.rope_compressed_sin.value = sin
        for layer in self.model.layers:
            layer.self_attn.prepare_grouped_wo_a()
            compressor = getattr(layer.self_attn, "compressor", None)
            if _HCA_FUSED_PROJ and compressor is not None and compressor.ratio == 128:
                compressor.prepare_fused_projection(self.mesh)

    def _load_regular_weights(self, info):
        """Populate only M-owned parameters from the assigned byte ranges."""
        import ml_dtypes

        dtype_map = {
            "F32": np.float32,
            "BF16": ml_dtypes.bfloat16,
            "F8_E4M3": ml_dtypes.float8_e4m3fn,
        }

        def read(key):
            entry = info[key][0]
            shape = tuple(entry["shape"])
            dtype = entry["dtype"]
            if dtype not in dtype_map and dtype != "F8_E8M0":
                raise ValueError(f"{key}: unsupported checkpoint dtype {dtype}")
            itemsize = 1 if dtype == "F8_E8M0" else np.dtype(dtype_map[dtype]).itemsize
            if entry["byte_size"] != int(np.prod(shape)) * itemsize:
                raise ValueError(f"{key}: invalid checkpoint dtype, shape, or byte count")
            with open(entry["file"], "rb") as source:
                source.seek(entry["byte_offset"])
                raw = source.read(entry["byte_size"])
            if len(raw) != entry["byte_size"]:
                raise ValueError(f"{key}: truncated checkpoint tensor")
            if dtype == "F8_E8M0":
                codes = np.frombuffer(raw, np.uint8).reshape(shape)
                if np.any(codes == 255):
                    raise ValueError(f"{key}: reserved E8M0 scale code 255")
                return np.ldexp(np.ones(shape, np.float32), codes.astype(np.int16) - 127)
            return np.frombuffer(raw, dtype=dtype_map[dtype]).reshape(shape)

        def parameter(path):
            obj = self
            for component in path.split("."):
                obj = obj[int(component)] if component.isdigit() else getattr(obj, component)
            return obj

        def assign(param, array, key):
            old = param.value
            if array.shape != old.shape:
                raise ValueError(
                    f"{key}: checkpoint {array.shape} does not match model {old.shape}"
                )
            if not np.isfinite(np.asarray(array, np.float32)).all():
                raise ValueError(f"{key}: non-finite checkpoint tensor")
            param.value = jax.device_put(
                np.asarray(array, dtype=old.dtype), NamedSharding(self.mesh, old.sharding.spec)
            )
            param.value.block_until_ready()

        mappings = build_weight_mappings(self.config, self.mesh)
        consumed = set()
        for key, mapping in mappings.items():
            path = mapping.target_path
            if path.endswith(".weight_scale"):
                continue  # Paired with its FP8 weight below.
            if path.endswith(".weight_q"):
                linear = parameter(path.rsplit(".", 1)[0])
                if info[key][0]["dtype"] != "F8_E4M3":
                    raise ValueError(f"{key}: expected F8_E4M3 checkpoint weight")
                scale_key = key.removesuffix(".weight") + ".scale"
                if info[scale_key][0]["dtype"] != "F8_E8M0":
                    raise ValueError(f"{scale_key}: expected F8_E8M0 block scales")
                weight = read(key)
                scale = read(scale_key)
                if scale.shape != ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128):
                    raise ValueError(f"{key}: invalid K128/N128 FP8 block scales")
                if hasattr(linear, "weight_q"):
                    assign(linear.weight_q, weight, key)
                    expanded = np.repeat(scale, 128, axis=0)[: weight.shape[0], :].T[:, None, :]
                    assign(linear.weight_scale, expanded, scale_key)
                else:
                    expanded = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
                    assign(
                        linear.weight,
                        (
                            weight.astype(np.float32)
                            * expanded[: weight.shape[0], : weight.shape[1]]
                        ).T,
                        key,
                    )
                consumed.update((key, scale_key))
                continue
            if key.startswith("hc_") or ".hc_" in key:
                if info[key][0]["dtype"] != "F32":
                    raise ValueError(f"{key}: mHC gate parameters must be F32")
            elif info[key][0]["dtype"] not in ("BF16", "F32"):
                raise ValueError(f"{key}: expected BF16 or F32 checkpoint tensor")
            value = read(key)
            assign(parameter(path), value.T if mapping.transpose else value, key)
            consumed.add(key)
        return consumed


EntryClass = [DeepseekV4ForCausalLM]
