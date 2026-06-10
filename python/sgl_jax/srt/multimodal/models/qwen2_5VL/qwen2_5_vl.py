"""Qwen2.5-VL in-model (refactor M3).

A single model registered on the standard srt LLM runtime (no staged ViT/AR pipeline).
Its forward encodes vision in-forward and merges via the canonical mm_core.merge():

    text_embed = self.model.embed_tokens(input_ids)
    mod_embeds = encode_image/encode_video(self.visual, pixel_values, grid_thw)
    fused = mm_core.merge(text_embed, mod_embeds, pad_values, input_ids)   # single mask-scatter
    forward_batch.input_embedding = fused                                  # AR body reads this
    -> self.model(forward_batch) -> logits

Reuses the existing ViT tower (`Qwen2_5_VL_VisionTransformer`) and the AR LLM body
(`Qwen2_5_VL_Model`), replacing the staged `vit_model_runner` cumsum-gather with merge().
See design doc §3.3 and tmp/refactor/m3-plan.md.

NOTE (validation status): import / construct(eval_shape) / weight-load are validated on
the TPU dev pod. The vision config is derived from the checkpoint's vision_config via
qwen_vl_vision_config_from_hf (hidden_size=1280 for 7B; the bare QwenVLModelVitConfig()
default 3584 is the post-merger LLM dim and breaks the patch_embed reshape). ViT
precompile/HBM behaviour (m3-plan risk 1) is observed via the encode+merge forward smoke.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.mm_core.merge import merge
from sgl_jax.srt.mm_core.pad_value import MM_PAD_SHIFT_VALUE
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.configs.config_registry import qwen_vl_vision_config_from_hf
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vit import (
    Qwen2_5_VL_VisionTransformer,
)
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vl_generation import (
    Qwen2_5_VL_Model,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Qwen2_5_VLForConditionalGeneration(nnx.Module):
    """In-model Qwen2.5-VL: owns the ViT + LLM and merges in-forward (refactor M3)."""

    # Multimodal capability declarations (mm_core.capability / U3).
    audio_kind = None
    has_deepstack = False

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # ViT tower (reused as-is). Vision config is derived from the checkpoint's
        # vision_config (hidden_size=1280 for 7B); the bare QwenVLModelVitConfig() default
        # (hidden_size=3584) is the post-merger LLM dim and breaks the patch_embed reshape.
        self.vision_config = qwen_vl_vision_config_from_hf(config)
        self.visual = Qwen2_5_VL_VisionTransformer(
            config=self.vision_config,
            dtype=self.dtype,
            rngs=None,
            mesh=mesh,
            norm_eps=getattr(self.vision_config, "rms_norm_eps", 1e-6),
        )

        # AR LLM body (reused as-is; reads forward_batch.input_embedding).
        self.model = Qwen2_5_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=mesh)

        self.image_token_id = getattr(config, "image_token_id", None)
        self.video_token_id = getattr(config, "video_token_id", None)
        self.spatial_merge_size = getattr(self.vision_config, "spatial_merge_size", 2)

    # ---- per-modality encoders (model-owned tower; capability U3) ----

    def _encode(self, pixel_values: jax.Array, grid_thw: tuple) -> jax.Array:
        """Encode one modality's pixels into [N, hidden], one ViT call per item."""
        embeds = []
        cur = 0
        for thw in grid_thw:
            t, h, w = thw
            size = int(t) * int(h) * int(w)
            embeds.append(self.visual(pixel_values[cur : cur + size, :], (thw,)))
            cur += size
        return jnp.concatenate(embeds, axis=0)

    def encode_image(self, pixel_values: jax.Array, grid_thw: tuple) -> jax.Array:
        return self._encode(pixel_values, grid_thw)

    def encode_video(self, pixel_values: jax.Array, grid_thw: tuple) -> jax.Array:
        return self._encode(pixel_values, grid_thw)

    # ---- forward: in-forward encode + merge, then reuse the AR body ----

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        is_extend = forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        if is_extend and forward_batch.contains_mm_inputs():
            # Placeholder rows carry pad_values (>= MM_PAD_SHIFT_VALUE, out of vocab range
            # after pad_input_tokens). Clamp them to a safe in-vocab id for the embed_tokens
            # lookup -- merge() overwrites those exact rows with the real modality features,
            # so the dummy text embedding there is discarded. merge() still receives the
            # ORIGINAL pad-value-laden input_ids to build its isin() placeholder mask.
            safe_ids = jnp.where(
                forward_batch.input_ids >= MM_PAD_SHIFT_VALUE, 0, forward_batch.input_ids
            )
            text_embed = self.model.embed_tokens(safe_ids)
            mod_embeds = []
            if forward_batch.mm_pixel_values is not None:
                mod_embeds.append(
                    self.encode_image(forward_batch.mm_pixel_values, forward_batch.mm_grid_thw)
                )
            if forward_batch.mm_pixel_values_videos is not None:
                mod_embeds.append(
                    self.encode_video(
                        forward_batch.mm_pixel_values_videos, forward_batch.mm_video_grid_thw
                    )
                )
            pad_values = list(forward_batch.mm_pad_values or ())
            fused = merge(
                text_embed,
                mod_embeds,
                pad_values,
                forward_batch.input_ids,
                mesh=self.mesh,
            ).embed
            # Stamp the fused embedding; the AR body reads forward_batch.input_embedding
            # in extend mode (no AR signature change needed -- m3-plan step1 decision).
            forward_batch.input_embedding = fused

        token_to_kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None

    # ---- weight loading: merge ViT (visual.*) + LLM (model.* / lm_head) mappings ----

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = {}
        mappings.update(self._vit_weight_mappings())
        mappings.update(self._llm_weight_mappings())
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen2.5-VL (in-model) weights loaded: %d mappings", len(mappings))

    def _vit_weight_mappings(self) -> dict:
        # visual.* keys from the HF checkpoint -> self.visual.* (replicated; no TP).
        m = {
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.kernel", sharding=(None, None), transpose=True
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.kernel", sharding=(None, None), transpose=True
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias", sharding=(None,), transpose=False
            ),
        }
        for i in range(getattr(self.vision_config, "depth", 0)):
            p = f"visual.blocks.{i}"
            m.update(
                {
                    f"{p}.norm1.weight": WeightMapping(
                        target_path=f"{p}.norm1.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.norm2.weight": WeightMapping(
                        target_path=f"{p}.norm2.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.attn.qkv.weight": WeightMapping(
                        target_path=f"{p}.attn.qkv_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.attn.qkv.bias": WeightMapping(
                        target_path=f"{p}.attn.qkv_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.attn.proj.weight": WeightMapping(
                        target_path=f"{p}.attn.proj.kernel", sharding=(None, None), transpose=True
                    ),
                    f"{p}.attn.proj.bias": WeightMapping(
                        target_path=f"{p}.attn.proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.mlp.gate_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.kernel", sharding=(None, None), transpose=True
                    ),
                    f"{p}.mlp.up_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.bias", sharding=(None,), transpose=False
                    ),
                    f"{p}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{p}.mlp.down_proj.bias": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.bias", sharding=(None,), transpose=False
                    ),
                }
            )
        return m

    def _llm_weight_mappings(self) -> dict:
        tc = self.text_config
        m = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not getattr(tc, "tie_word_embeddings", False):
            m["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )
        attn_bias = getattr(tc, "attention_bias", True)
        for i in range(tc.num_hidden_layers):
            p = f"model.layers.{i}"
            m.update(
                {
                    f"{p}.input_layernorm.weight": WeightMapping(
                        target_path=f"{p}.input_layernorm.scale", sharding=(None,), transpose=False
                    ),
                    f"{p}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{p}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{p}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                    ),
                    f"{p}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{p}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{p}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{p}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                        head_dim_padding=True,
                    ),
                    f"{p}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{p}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{p}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{p}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )
            if attn_bias:
                for proj, kvpad in (("q", False), ("k", True), ("v", True)):
                    m[f"{p}.self_attn.{proj}_proj.bias"] = WeightMapping(
                        target_path=f"{p}.self_attn.{proj}_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=kvpad,
                    )
        return m


EntryClass = [Qwen2_5_VLForConditionalGeneration]
