"""InternVL 3 (HF ``InternVLForConditionalGeneration``, e.g. OpenGVLab/InternVL3-2B-hf).

InternViT encoder (CLS + learned absolute position embeddings, pre-LN blocks
with layer scale) -> drop CLS -> 0.5x pixel shuffle -> LayerNorm/MLP projector
-> Qwen2 language model. The image processor tiles each image into 448x448
crops (plus a thumbnail); every crop yields 256 LM tokens.

The host side flattens every crop into 1024 row-major patches of
``3 * 14 * 14`` values (conv-weight order), so a crop is one fixed-size block
of the lane-packed encoder input and ``merge_unit`` (pixel-shuffle factor
squared) is 4.
"""

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    VisionInputSpec,
)
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs

logger = logging.getLogger(__name__)


def _as_pair(value) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _gelu(x: jax.Array) -> jax.Array:
    # HF "gelu" is the exact (erf) GELU.
    return jax.nn.gelu(x, approximate=False)


class _Linear(nnx.Module):
    """Replicated dense layer for the (small) vision tower."""

    def __init__(self, in_dim: int, out_dim: int, mesh, dtype, use_bias: bool = True):
        self.inner = LinearBase(
            input_size=in_dim,
            output_size=out_dim,
            mesh=mesh,
            use_bias=use_bias,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.replicated = NamedSharding(mesh, PartitionSpec())

    def __call__(self, x: jax.Array) -> jax.Array:
        out, _ = self.inner(x, out_sharding=self.replicated)
        return out


class _LayerNorm(nnx.Module):
    def __init__(self, dim: int, eps: float, dtype):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        orig = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        x = x * self.weight.value.astype(jnp.float32) + self.bias.value.astype(jnp.float32)
        return x.astype(orig)


class InternVisionLayer(nnx.Module):
    def __init__(self, config, mesh, dtype):
        if getattr(config, "use_qk_norm", False):
            raise NotImplementedError("InternViT qk-norm (InternViT-6B) is not supported yet")
        if getattr(config, "norm_type", "layer_norm") != "layer_norm":
            raise NotImplementedError("Only layer_norm InternViT variants are supported")
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = d // self.num_heads
        bias = getattr(config, "attention_bias", True)
        self.q_proj = _Linear(d, d, mesh, dtype, bias)
        self.k_proj = _Linear(d, d, mesh, dtype, bias)
        self.v_proj = _Linear(d, d, mesh, dtype, bias)
        self.projection_layer = _Linear(d, d, mesh, dtype)
        self.fc1 = _Linear(d, config.intermediate_size, mesh, dtype)
        self.fc2 = _Linear(config.intermediate_size, d, mesh, dtype)
        self.layernorm_before = _LayerNorm(d, config.layer_norm_eps, dtype)
        self.layernorm_after = _LayerNorm(d, config.layer_norm_eps, dtype)
        self.lambda_1 = nnx.Param(jnp.ones((d,), dtype=dtype))
        self.lambda_2 = nnx.Param(jnp.ones((d,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        """*x*: ``[crops, seq, hidden]`` with full (bidirectional) attention per crop."""
        b, s, d = x.shape
        h = self.layernorm_before(x).reshape(b * s, d)
        q = self.q_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        attn = jax.nn.dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        attn = self.projection_layer(attn.reshape(b * s, d)).reshape(b, s, d)
        x = x + self.lambda_1.value * attn

        h = self.layernorm_after(x).reshape(b * s, d)
        h = self.fc2(_gelu(self.fc1(h))).reshape(b, s, d)
        return x + self.lambda_2.value * h


class InternVisionTower(nnx.Module):
    """InternViT + pixel shuffle + projector, consuming lane-packed crop patches."""

    def __init__(self, config, text_hidden_size: int, mesh, dtype):
        vc = config.vision_config
        self.mesh = mesh
        self.dtype = dtype
        self.vision_tp = False
        self.specs = VisionShardSpecs(mesh, False)
        self.hidden_size = vc.hidden_size
        ph, pw = _as_pair(vc.patch_size)
        ih, iw = _as_pair(vc.image_size)
        self.grid = (ih // ph, iw // pw)
        self.patches_per_crop = self.grid[0] * self.grid[1]
        self.in_channels = getattr(vc, "num_channels", 3)
        self.patch_dim = self.in_channels * ph * pw
        ratio = float(config.downsample_ratio)
        self.shuffle = int(round(1 / ratio))
        self.merge_unit = self.shuffle**2
        if config.vision_feature_select_strategy != "default" or config.vision_feature_layer != -1:
            raise NotImplementedError(
                "Only last-layer, CLS-dropped InternVL features are supported"
            )

        d = vc.hidden_size
        self.patch_embedding = _Linear(self.patch_dim, d, mesh, dtype)
        self.cls_token = nnx.Param(jnp.zeros((1, d), dtype=dtype))
        self.position_embeddings = nnx.Param(jnp.zeros((self.patches_per_crop + 1, d), dtype=dtype))
        self.layers = nnx.List(
            [InternVisionLayer(vc, mesh, dtype) for _ in range(vc.num_hidden_layers)]
        )
        merged = d * self.merge_unit
        self.mm_norm = _LayerNorm(merged, 1e-5, dtype)
        self.mm_fc1 = _Linear(merged, text_hidden_size, mesh, dtype)
        self.mm_fc2 = _Linear(text_hidden_size, text_hidden_size, mesh, dtype)

    def prepare_metadata(self, grid_thw, *, capacity: int, sharding) -> dict:
        # Crops are fixed-size and self-contained; no per-item metadata needed.
        return {}

    def _pixel_shuffle(self, x: jax.Array) -> jax.Array:
        """HF ``InternVLModel.pixel_shuffle`` for x: ``[n, w, h, c]``."""
        n, w, h, c = x.shape
        r = self.shuffle
        x = x.reshape(n, w, h // r, c * r)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = x.reshape(n, h // r, w // r, c * r * r)
        return jnp.transpose(x, (0, 2, 1, 3))

    def __call__(self, patches: jax.Array, **_) -> jax.Array:
        return self.encode(patches)

    @jax.jit
    def encode(self, patches: jax.Array) -> jax.Array:
        replicated = NamedSharding(self.mesh, PartitionSpec())
        num_lanes = encoder_num_lanes(self.mesh, self.vision_tp)
        x = jax.sharding.reshard(patches, replicated).astype(self.dtype)
        x = x.reshape(num_lanes, -1, self.patch_dim)
        capacity = x.shape[1]

        # Pad each lane to whole crops (precompile buckets need not be crop-aligned).
        ppc = self.patches_per_crop
        crops_per_lane = -(-capacity // ppc)
        pad = crops_per_lane * ppc - capacity
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        crops = num_lanes * crops_per_lane
        x = x.reshape(crops * ppc, self.patch_dim)

        x = self.patch_embedding(x).reshape(crops, ppc, self.hidden_size)
        cls = jnp.broadcast_to(self.cls_token.value[None], (crops, 1, self.hidden_size))
        x = jnp.concatenate([cls, x], axis=1) + self.position_embeddings.value[None]
        for layer in self.layers:
            x = layer(x)

        x = x[:, 1:, :].reshape(crops, self.grid[0], self.grid[1], self.hidden_size)
        x = self._pixel_shuffle(x)
        tokens_per_crop = ppc // self.merge_unit
        x = x.reshape(crops * tokens_per_crop, self.hidden_size * self.merge_unit)
        x = self.mm_fc2(_gelu(self.mm_fc1(self.mm_norm(x))))

        x = x.reshape(num_lanes, crops_per_lane * tokens_per_crop, -1)
        return x[:, : capacity // self.merge_unit, :]


class InternVLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    """InternVL 3: InternViT vision tower + Qwen2 language model."""

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config.text_config
        self.dtype = dtype or jnp.bfloat16
        text_arch = (getattr(self.text_config, "architectures", None) or ["Qwen2ForCausalLM"])[0]
        if text_arch != "Qwen2ForCausalLM":
            raise NotImplementedError(
                f"InternVL with a {text_arch} language model is not supported yet"
            )

        self.model = Qwen2Model(self.text_config, mesh=mesh, dtype=self.dtype)
        # HF ties on the top-level flag, but only when the checkpoint has no
        # separate lm_head (InternVL3-*-hf ships one that differs from the
        # embedding, so HF keeps it untied). Decided again at load time.
        self.tie_word_embeddings = getattr(
            config, "tie_word_embeddings", getattr(self.text_config, "tie_word_embeddings", False)
        )
        self.lm_head = ParallelLMHead(
            self.text_config.vocab_size,
            self.text_config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.logits_processor = LogitsProcessor(
            self.text_config.vocab_size,
            mesh=self.mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.visual = InternVisionTower(config, self.text_config.hidden_size, mesh, self.dtype)
        # spatial_merge_size**2 must equal the pixel-shuffle merge unit (4).
        self.vision_input_spec = VisionInputSpec(
            patch_dim=self.visual.patch_dim,
            spatial_merge_size=self.visual.shuffle,
            dtype=np.dtype("float32"),
        )

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def get_image_feature(self, items_per_lane: list[list[MultimodalDataItem]]) -> jax.Array:
        return run_mrope_vision_model(
            self.visual,
            items_per_lane,
            mesh=self.mesh,
            num_lanes=encoder_num_lanes(self.mesh, self.visual.vision_tp),
            merge_unit=self.visual.merge_unit,
            rope_type="rope_2d",
            input_sharding=self.visual.specs.sharding(self.visual.specs.batch_axis),
            output_sharding=self.visual.specs.sharding(),
        )

    def get_multimodal_encode_funcs(self):
        return {
            Modality.IMAGE: self.get_image_feature,
            Modality.MULTI_IMAGES: self.get_image_feature,
        }

    # ------------------------------------------------------------------ weights
    def load_weights(self, model_config: ModelConfig) -> None:
        mappings = {}
        for key, spec in self._language_weight_mappings().items():
            # HF checkpoints use "language_model.model.*" (hub *-hf repos) or
            # "model.language_model.*" (transformers>=4.52 save_pretrained).
            mappings[f"language_model.{key}"] = spec
            mappings[key.replace("model.", "model.language_model.", 1)] = spec
        for key, spec in self._vision_weight_mappings().items():
            mappings[key] = spec
            mappings[f"model.{key}"] = spec
        head = self.lm_head.weight_mapping("lm_head.embedding")
        for key in ("language_model.lm_head.weight", "lm_head.weight"):
            mappings[key] = head
        loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        if not loader.dummy_mode:
            present = loader.metadata
            mappings = {k: v for k, v in mappings.items() if k in present}
            has_head = any(spec is head for spec in mappings.values())
            self.tie_word_embeddings = not has_head
            if not has_head:
                self.lm_head.embedding = self.model.embed_tokens.embedding
        loader.load(mappings)
        logger.info("InternVL weights loaded (tied lm_head: %s).", self.tie_word_embeddings)

    def _language_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        for layer_idx in range(self.text_config.num_hidden_layers):
            mappings.update(self._language_layer_mappings(layer_idx))
        return mappings

    def _language_layer_mappings(self, layer_idx: int) -> dict:
        return Qwen2_5_VLForConditionalGeneration._language_layer_mappings(self, layer_idx)

    def _vision_weight_mappings(self) -> dict:
        rep2, rep1 = (None, None), (None,)

        def lin(src: str, dst: str) -> dict:
            return {
                f"{src}.weight": WeightSpec(
                    target_path=f"{dst}.inner.weight", sharding=rep2, transpose=True
                ),
                f"{src}.bias": WeightSpec(target_path=f"{dst}.inner.bias", sharding=rep1),
            }

        def norm(src: str, dst: str) -> dict:
            return {
                f"{src}.weight": WeightSpec(target_path=f"{dst}.weight", sharding=rep1),
                f"{src}.bias": WeightSpec(target_path=f"{dst}.bias", sharding=rep1),
            }

        vt = "vision_tower"
        d = self.visual.hidden_size
        mappings = {
            # Conv2d [out, C, kh, kw] -> flattened-patch linear [C*kh*kw, out].
            f"{vt}.embeddings.patch_embeddings.projection.weight": WeightSpec(
                target_path="visual.patch_embedding.inner.weight",
                sharding=rep2,
                transpose_axes=(1, 2, 3, 0),  # applied before reshape
                reshape=(self.visual.patch_dim, d),
            ),
            f"{vt}.embeddings.patch_embeddings.projection.bias": WeightSpec(
                target_path="visual.patch_embedding.inner.bias", sharding=rep1
            ),
            f"{vt}.embeddings.cls_token": WeightSpec(
                target_path="visual.cls_token", sharding=rep2, reshape=(1, d)
            ),
            f"{vt}.embeddings.position_embeddings": WeightSpec(
                target_path="visual.position_embeddings",
                sharding=rep2,
                reshape=(self.visual.patches_per_crop + 1, d),
            ),
            **norm("multi_modal_projector.layer_norm", "visual.mm_norm"),
            **lin("multi_modal_projector.linear_1", "visual.mm_fc1"),
            **lin("multi_modal_projector.linear_2", "visual.mm_fc2"),
        }
        for i in range(len(self.visual.layers)):
            src, dst = f"{vt}.encoder.layer.{i}", f"visual.layers.{i}"
            for name in ("q_proj", "k_proj", "v_proj", "projection_layer"):
                mappings.update(lin(f"{src}.attention.{name}", f"{dst}.{name}"))
            mappings.update(lin(f"{src}.mlp.fc1", f"{dst}.fc1"))
            mappings.update(lin(f"{src}.mlp.fc2", f"{dst}.fc2"))
            mappings.update(norm(f"{src}.layernorm_before", f"{dst}.layernorm_before"))
            mappings.update(norm(f"{src}.layernorm_after", f"{dst}.layernorm_after"))
            for lam in ("lambda_1", "lambda_2"):
                mappings[f"{src}.{lam}"] = WeightSpec(target_path=f"{dst}.{lam}", sharding=rep1)
        return mappings

    def get_embed_and_head(self):
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = InternVLForConditionalGeneration
