from typing import List
from transformers import PretrainedConfig
import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.utils import logger


class Qwen2_5_VLForConditionalGeneration(nnx.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.mesh = mesh
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
        )

        self.model = Qwen2Model(
            self.config, dtype=self.dtype, rngs=rngs, mesh=mesh
        )

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        image_grid_thw = jnp.concatenate([item.image_grid_thw for item in items], axis=0)
        assert pixel_values.ndim == 2, pixel_values.ndim
        assert image_grid_thw.ndim == 2, image_grid_thw.ndim
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        video_grid_thw = jnp.concatenate([item.video_grid_thw for item in items], axis=0)
        assert pixel_values.ndim == 2, pixel_values.ndim
        assert video_grid_thw.ndim == 2, video_grid_thw.ndim
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states, layers_kv_fused, layers_callback_flag = general_mm_embed_routine(
            forward_batch=forward_batch,
            language_model=self.model,
            token_to_kv_pool=token_to_kv_pool,
            multimodal_model=self,
        )
        
        return self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata), layers_kv_fused, layers_callback_flag

    def load_weights(self, model_config, rng_key: jax.Array):
        """Load weights for Qwen2.5-VL model.
        
        Args:
            model_config: Model configuration containing model path and settings
            rng_key: JAX random key for initialization
        """
        self.rng = nnx.Rngs(rng_key)
        
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        
        weight_mappings = self._create_qwen2_5_vl_weight_mappings()
        
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen2.5-VL weights loaded successfully!")
    
    def _create_qwen2_5_vl_weight_mappings(self) -> dict:
        """Create weight mappings for Qwen2.5-VL model.
        
        Returns:
            Dictionary mapping HuggingFace weight names to model parameter paths
        """        
        mappings = {}
        
        # Vision transformer weights
        mappings.update(self._create_vision_transformer_mappings())
        
        # Language model embeddings
        mappings["model.embed_tokens.weight"] = WeightMapping(
            target_path="model.embed_tokens.embedding",
            sharding=("tensor", None),
            transpose=False,
        )
        
        # Language model norm
        mappings["model.norm.weight"] = WeightMapping(
            target_path="model.norm.scale",
            sharding=(None,),
            transpose=False,
        )
        
        # LM head
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )
        
        # Language model layers
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)
        
        return mappings
    
    def _create_vision_transformer_mappings(self) -> dict:
        """Create weight mappings for the vision transformer.
        
        Returns:
            Dictionary mapping vision transformer weight names to model paths
        """        
        mappings = {}
        
        # Vision embeddings
        mappings["visual.patch_embed.proj.weight"] = WeightMapping(
            target_path="visual.patch_embed.proj.weight",
            sharding=(None, None, None, None),
            transpose=False,
        )
        
        if hasattr(self.visual, "patch_embed") and hasattr(self.visual.patch_embed, "proj"):
            if hasattr(self.visual.patch_embed.proj, "bias"):
                mappings["visual.patch_embed.proj.bias"] = WeightMapping(
                    target_path="visual.patch_embed.proj.bias",
                    sharding=(None,),
                    transpose=False,
                )
        
        # Vision transformer layers
        if hasattr(self.config, "vision_config"):
            num_vision_layers = getattr(self.config.vision_config, "num_hidden_layers", 0)
            for layer_idx in range(num_vision_layers):
                vision_layer_mappings = self._create_vision_layer_mappings(layer_idx)
                mappings.update(vision_layer_mappings)
        
        return mappings
    
    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single vision transformer layer.
        
        Args:
            layer_idx: Index of the vision layer
            
        Returns:
            Dictionary mapping vision layer weight names to model paths
        """
        from sgl_jax.srt.utils.weight_utils import WeightMapping
        
        prefix = f"visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"
        
        mappings = {
            # Attention norm
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            # Attention QKV projection
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # Attention output projection
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # MLP norm
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLP layers
            f"{prefix}.mlp.fc1.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.fc1.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.fc1.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.fc1.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            f"{prefix}.mlp.fc2.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.fc2.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.mlp.fc2.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.fc2.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        
        return mappings
    
    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single language model layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Dictionary mapping layer weight names to model paths
        """
        from sgl_jax.srt.utils.weight_utils import WeightMapping
        
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
        
        # Add bias mappings if attention_bias is enabled
        if getattr(self.config, "attention_bias", True):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
            }
            mappings.update(bias_mappings)
        
        return mappings


EntryClass = [Qwen2_5_VLForConditionalGeneration]