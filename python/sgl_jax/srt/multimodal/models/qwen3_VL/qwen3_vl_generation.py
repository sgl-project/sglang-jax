import logging


import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vl_generation import Qwen2_5_VL_Model
from sgl_jax.srt.multimodal.models.qwen3_VL.qwen3_vl_vit import (
    Qwen3VLVisionTransformer,
    Qwen3VLVisionConfig,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Qwen3_VL_Generation(nnx.Module):
    """
    Qwen3-VL model for conditional generation.
    Combines Qwen2.5-VL text backbone (with M-RoPE) and Qwen3-VL Vision Transformer.
    """

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Adapter for Qwen2.5-VL compatibility
        # Qwen3-VL config has `mrope_section` as attribute, but Qwen2.5-VL expects it in `rope_scaling` dict.
        if hasattr(self.text_config, "mrope_section") and (
            not hasattr(self.text_config, "rope_scaling")
            or self.text_config.rope_scaling is None
            or "mrope_section" not in self.text_config.rope_scaling
        ):
            # Create a wrapper or modify if mutable
            # If it's a frozen dataclass, we can't modify it. We use a proxy.
            class ConfigProxy:
                def __init__(self, cfg):
                    self._cfg = cfg

                def __getattr__(self, name):
                    if name == "rope_scaling":
                        # Return minimal dict with mrope_section
                        return {"mrope_section": self._cfg.mrope_section}
                    return getattr(self._cfg, name)

            self.text_config = ConfigProxy(self.text_config)

        # Use Qwen2.5-VL Model as the text backbone since it supports M-RoPE
        # We ensure the config passed to it has the necessary M-RoPE parameters
        self.model = Qwen2_5_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)

        # Initialize Vision Transformer
        # We need to extract vision config from the main config
        vision_config_dict = getattr(self.config, "vision_config", None)
        if vision_config_dict is None:
            # Fallback or error if strictly required, but for now assumption is it's there
            # Some HG configs might have it as a dict or object
            pass

        # Create Vision Config object (assuming it matches the dataclass in qwen3_vl_vit)
        # We might need to map from dict to dataclass if it's a dict
        if isinstance(vision_config_dict, dict):
            self.vision_config = Qwen3VLVisionConfig(**vision_config_dict)
        else:
            self.vision_config = vision_config_dict

        if self.vision_config:
            self.visual = Qwen3VLVisionTransformer(self.vision_config, dtype=self.dtype, mesh=mesh)
        else:
            self.visual = None
            logger.warning("No vision config found, Vision Transformer not initialized.")

        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)

        # Multimodal token ids
        self.image_token_id = getattr(self.config, "image_token_id", 151655)
        self.video_token_id = getattr(self.config, "video_token_id", 151656)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3-VL weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        # Start with Qwen2 text mappings
        mappings = self._create_qwen2_weight_mappings()

        if self.visual:
            mappings.update(self._create_vision_weight_mappings())

        return mappings

    def _create_qwen2_weight_mappings(self) -> dict:
        # Reuse logic from Qwen2.5-VL / Qwen2

        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.text_config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        # Same as Qwen2.5-VL
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.text_config, "attention_bias", True):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.o_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                }
            )

        return mappings

    def _create_vision_weight_mappings(self) -> dict:
        mappings = {}
        # Mappings for Vision Transformer
        # "model.visual.patch_embed.proj.weight" -> "visual.patch_embed.proj.kernel"
        # "model.visual.patch_embed.proj.bias" -> "visual.patch_embed.proj.bias"
        # "model.visual.pos_embed" -> "visual.pos_embed.embedding"
        # "model.visual.blocks.N..." -> "visual.blocks.N..."
        # "model.visual.merger..." -> "visual.merger..."

        # Patch Embed
        mappings["model.visual.patch_embed.proj.weight"] = WeightMapping(
            target_path="visual.patch_embed.proj.kernel",
            sharding=(None, None, None, None),  # Conv weights usually not sharded or replicated
            transpose=True,  # Conv weights might need transpose depending on formatting
        )
        mappings["model.visual.patch_embed.proj.bias"] = WeightMapping(
            target_path="visual.patch_embed.proj.bias", sharding=(None,), transpose=False
        )

        # Pos Embed
        mappings["model.visual.pos_embed"] = WeightMapping(
            target_path="visual.pos_embed.embedding", sharding=(None, None), transpose=False
        )

        # Blocks
        if self.visual:
            num_blocks = self.visual.config.depth
            for i in range(num_blocks):
                mappings.update(self._create_vision_block_mappings(i))

        # Merger
        mappings["model.visual.merger.norm.weight"] = WeightMapping(
            target_path="visual.merger.norm.scale", sharding=(None,), transpose=False
        )
        mappings["model.visual.merger.linear_fc1.weight"] = WeightMapping(
            target_path="visual.merger.linear_fc1.weight", sharding=(None, None), transpose=True
        )
        mappings["model.visual.merger.linear_fc1.bias"] = WeightMapping(
            target_path="visual.merger.linear_fc1.bias", sharding=(None,), transpose=False
        )
        mappings["model.visual.merger.linear_fc2.weight"] = WeightMapping(
            target_path="visual.merger.linear_fc2.weight", sharding=(None, None), transpose=True
        )
        mappings["model.visual.merger.linear_fc2.bias"] = WeightMapping(
            target_path="visual.merger.linear_fc2.bias", sharding=(None,), transpose=False
        )

        return mappings

    def _create_vision_block_mappings(self, idx: int) -> dict:
        prefix = f"model.visual.blocks.{idx}"
        target = f"visual.blocks.{idx}"
        m = {}

        # Norm1
        m[f"{prefix}.norm1.weight"] = WeightMapping(
            target_path=f"{target}.norm1.scale", sharding=(None,), transpose=False
        )
        # Norm2
        m[f"{prefix}.norm2.weight"] = WeightMapping(
            target_path=f"{target}.norm2.scale", sharding=(None,), transpose=False
        )

        # Attn
        m[f"{prefix}.attn.qkv.weight"] = WeightMapping(
            target_path=f"{target}.attn.qkv.weight", sharding=(None, None), transpose=True
        )
        m[f"{prefix}.attn.qkv.bias"] = WeightMapping(
            target_path=f"{target}.attn.qkv.bias", sharding=(None,), transpose=False
        )
        m[f"{prefix}.attn.proj.weight"] = WeightMapping(
            target_path=f"{target}.attn.proj.weight", sharding=(None, None), transpose=True
        )
        m[f"{prefix}.attn.proj.bias"] = WeightMapping(
            target_path=f"{target}.attn.proj.bias", sharding=(None,), transpose=False
        )

        # MLP
        m[f"{prefix}.mlp.linear_fc1.weight"] = WeightMapping(
            target_path=f"{target}.mlp.linear_fc1.weight", sharding=(None, None), transpose=True
        )
        m[f"{prefix}.mlp.linear_fc1.bias"] = WeightMapping(
            target_path=f"{target}.mlp.linear_fc1.bias", sharding=(None,), transpose=False
        )
        m[f"{prefix}.mlp.linear_fc2.weight"] = WeightMapping(
            target_path=f"{target}.mlp.linear_fc2.weight", sharding=(None, None), transpose=True
        )
        m[f"{prefix}.mlp.linear_fc2.bias"] = WeightMapping(
            target_path=f"{target}.mlp.linear_fc2.bias", sharding=(None,), transpose=False
        )
        return m

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            weight = self.model.embed_tokens.embedding.value
            return (weight, weight)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None

    def encode_vision(self, pixel_values: jax.Array, image_grid_thw: jax.Array):
        """
        Encode vision inputs.
        Args:
            pixel_values: [N, ...]
            image_grid_thw: [N, 3]
        Returns:
            vision_embeddings: [N_vis_tokens, hidden_size]
        """
        if self.visual is None:
            raise ValueError("Vision model is not initialized.")

        # Call ViT
        # Note: Qwen3VLVisionTransformer returns (hidden_states, deepstack_features)
        # We generally only use the last hidden states for generation input for now
        # unless deepstack features are mixed in (Q3VL source ignores them in generation!)
        hidden_states, _ = self.visual(pixel_values, image_grid_thw)
        return hidden_states


Qwen3VLForConditionalGeneration = Qwen3_VL_Generation
EntryClass = [Qwen3_VL_Generation, Qwen3VLForConditionalGeneration]
