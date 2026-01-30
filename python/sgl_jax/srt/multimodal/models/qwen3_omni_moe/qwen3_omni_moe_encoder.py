import logging

import jax
from flax import nnx
from jax import numpy as jnp
from transformers import Qwen3OmniMoeThinkerConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig, \
    Qwen3OmniMoeVisionEncoderConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.audio_encoder import Qwen3OmniMoeAudioEncoder
from sgl_jax.srt.multimodal.models.qwen3_omni_moe.vision_encoder import Qwen3OmniMoeVisionEncoder
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.srt.layers.embeddings import Embed

logger = logging.getLogger(__name__)


class Qwen3OmniMoeThinkerEmbedding(nnx.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mesh = mesh
        self.dtype = dtype
        self.config = config
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config, mesh=mesh, dtype=dtype, rngs=rngs)
        self.visual = Qwen3OmniMoeVisionEncoder(config.vision_config, mesh=mesh, dtype=dtype, rngs=rngs)
        self.text_embed_tokens = Embed(
            num_embeddings=config.text_config.vocab_size,
            features=config.text_config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = {
            **self._create_audio_tower_weight_mappings(self.config.audio_config),
            **self._create_visual_weight_mappings(self.config.vision_config),
            **self._create_text_embed_tokens_mappings(self.config.text_config),
        }

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3OmniMoeThinkerEmbedding weights loaded successfully!")

    @staticmethod
    def _create_audio_tower_weight_mappings(config: Qwen3OmniMoeAudioEncoderConfig) -> dict:
        mappings = {}
        prefix = "thinker.audio_tower"
        target_prefix = "audio_tower"

        # 1. conv2d layer: (conv2d1, conv2d2, conv2d3)
        for i in range(1, 4):
            mappings[f"{prefix}.conv2d{i}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.conv2d{i}.kernel",
                transpose_axes=(2, 3, 1, 0),  # PT [O, I, H, W] -> JAX [H, W, I, O]
                sharding=(None, None, None, None),
            )
            mappings[f"{prefix}.conv2d{i}.bias"] = WeightMapping(
                target_path=f"{target_prefix}.conv2d{i}.bias",
                sharding=(None,),
            )

        # 2. conv_out layer:
        mappings[f"{prefix}.conv_out.weight"] = WeightMapping(
            target_path=f"{target_prefix}.conv_out.weight",
            transpose=True,  # PT [O, I] -> JAX [I, O]
            sharding=(None, None),
        )

        # 3. Transformer layer: (0-31)
        for i in range(config.num_hidden_layers):
            l_pre = f"{prefix}.layers.{i}"
            l_targ = f"{target_prefix}.layers.{i}"

            # Self Attention: q_proj, k_proj, v_proj, out_proj
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                mappings[f"{l_pre}.self_attn.{proj}.weight"] = WeightMapping(
                    target_path=f"{l_targ}.self_attn.{proj}.weight",
                    transpose=True,  # PT [O, I] -> JAX [I, O]
                    sharding=(None, None),
                )
                mappings[f"{l_pre}.self_attn.{proj}.bias"] = WeightMapping(
                    target_path=f"{l_targ}.self_attn.{proj}.bias",
                    sharding=(None,),
                )

            # Attention LayerNorm (weight -> scale)
            mappings[f"{l_pre}.self_attn_layer_norm.weight"] = WeightMapping(
                target_path=f"{l_targ}.self_attn_layer_norm.scale",
                sharding=(None,),
            )
            mappings[f"{l_pre}.self_attn_layer_norm.bias"] = WeightMapping(
                target_path=f"{l_targ}.self_attn_layer_norm.bias",
                sharding=(None,),
            )

            # MLP layer: (fc1, fc2)
            for fc in ["fc1", "fc2"]:
                mappings[f"{l_pre}.{fc}.weight"] = WeightMapping(
                    target_path=f"{l_targ}.{fc}.weight",
                    transpose=True,  # PT [O, I] -> JAX [I, O]
                    sharding=(None, None),
                )
                mappings[f"{l_pre}.{fc}.bias"] = WeightMapping(
                    target_path=f"{l_targ}.{fc}.bias",
                    sharding=(None,),
                )

            # Final LayerNorm (weight -> scale)
            mappings[f"{l_pre}.final_layer_norm.weight"] = WeightMapping(
                target_path=f"{l_targ}.final_layer_norm.scale",
                sharding=(None,),
            )
            mappings[f"{l_pre}.final_layer_norm.bias"] = WeightMapping(
                target_path=f"{l_targ}.final_layer_norm.bias",
                sharding=(None,),
            )

        # 4. post process: (ln_post, proj1, proj2)
        # ln_post (weight -> scale)
        mappings[f"{prefix}.ln_post.weight"] = WeightMapping(
            target_path=f"{target_prefix}.ln_post.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.ln_post.bias"] = WeightMapping(
            target_path=f"{target_prefix}.ln_post.bias",
            sharding=(None,),
        )

        # proj1 & proj2
        for p in ["proj1", "proj2"]:
            mappings[f"{prefix}.{p}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.{p}.weight",
                transpose=True,  # PT [O, I] -> JAX [I, O]
                sharding=(None, None),
            )
            mappings[f"{prefix}.{p}.bias"] = WeightMapping(
                target_path=f"{target_prefix}.{p}.bias",
                sharding=(None,),
            )

        return mappings

    @staticmethod
    def _create_visual_weight_mappings(config: Qwen3OmniMoeVisionEncoderConfig) -> dict:
        prefix: str = "thinker.visual."
        target_prefix = "visual."
        mappings = {}

        # Helper functions to reduce repetition
        def add_linear(src: str, dst: str, tp_col: bool = False, tp_row: bool = False):
            """Add linear layer mapping with optional TP sharding."""
            w_sharding = (None, "tensor") if tp_col else ("tensor", None) if tp_row else (None, None)
            b_sharding = ("tensor",) if tp_col else (None,)
            mappings[f"{prefix}{src}.weight"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.weight", sharding=w_sharding, transpose=True
            )
            mappings[f"{prefix}{src}.bias"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.bias", sharding=b_sharding, transpose=False
            )

        def add_layernorm(src: str, dst: str):
            """Add layernorm mapping."""
            mappings[f"{prefix}{src}.weight"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.scale", sharding=(None,), transpose=False
            )
            mappings[f"{prefix}{src}.bias"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.bias", sharding=(None,), transpose=False
            )

        # ==================== Patch Embedding ====================
        # Conv3d: PyTorch (out, in, T, H, W) -> JAX (T, H, W, in, out)
        mappings[f"{prefix}patch_embed.proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.kernel",
            sharding=(None, None, None, None, None),
            transpose=False,
            transpose_axes=(2, 3, 4, 1, 0),
        )
        mappings[f"{prefix}patch_embed.proj.bias"] = WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.bias", sharding=(None,), transpose=False
        )

        # ==================== Position Embedding ====================
        mappings[f"{prefix}pos_embed.weight"] = WeightMapping(
            target_path=f"{target_prefix}pos_embed.embedding", sharding=(None, None), transpose=False
        )

        # ==================== Transformer Blocks ====================
        for i in range(config.depth):
            block = f"blocks.{i}"

            # LayerNorm
            add_layernorm(f"{block}.norm1", f"{block}.norm1")
            add_layernorm(f"{block}.norm2", f"{block}.norm2")

            # Attention: QKV (column-wise TP), Output (row-wise TP)
            add_linear(f"{block}.attn.qkv", f"{block}.attn.qkv_proj", tp_col=True)
            add_linear(f"{block}.attn.proj", f"{block}.attn.o_proj", tp_row=True)

            # MLP: fc1 (column-wise TP), fc2 (row-wise TP)
            add_linear(f"{block}.mlp.linear_fc1", f"{block}.mlp.fc1", tp_col=True)
            add_linear(f"{block}.mlp.linear_fc2", f"{block}.mlp.fc2", tp_row=True)

        # ==================== Final Merger ====================
        add_layernorm("merger.ln_q", "merger.ln_q")
        # Merger MLP: [0]=Linear, [1]=GELU, [2]=Linear
        add_linear("merger.mlp.0", "merger.mlp_fc1", tp_col=True)
        add_linear("merger.mlp.2", "merger.mlp_fc2", tp_row=True)

        # ==================== Deepstack Mergers ====================
        deepstack_indexes = getattr(config, "deepstack_visual_indexes", [8, 16, 24])
        for idx in range(len(deepstack_indexes)):
            src = f"merger_list.{idx}"
            dst = f"deepstack_mergers.{idx}"

            add_layernorm(f"{src}.ln_q", f"{dst}.ln_q")
            add_linear(f"{src}.mlp.0", f"{dst}.mlp_fc1", tp_col=True)
            add_linear(f"{src}.mlp.2", f"{dst}.mlp_fc2", tp_row=True)

        return mappings

    @staticmethod
    def _create_text_embed_tokens_mappings(config: Qwen3OmniMoeAudioEncoderConfig) -> dict:
        mappings = {}
        prefix = "thinker.model"
        target_prefix = "text_embed_tokens"

        mappings[f"{prefix}.embed_tokens.weight"] = WeightMapping(
            target_path=f"{target_prefix}.embedding",
            sharding=("tensor", None),
        )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
    ):
        # 1. Extract the input embeddings
        inputs_embeds = self.text_embed_tokens(forward_batch.input_ids)
