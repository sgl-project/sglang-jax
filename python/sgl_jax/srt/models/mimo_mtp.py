import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
import logging

logger = logging.getLogger(__name__)

class MiMoMTPLayer(nnx.Module):
    """Container for MiMo multi-token prediction (MTP) block weights."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.layer_id = layer_id
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.token_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = LinearBase(
            input_size=config.hidden_size * 2,
            output_size=config.hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
           
        self.mtp_layers = Qwen2DecoderLayer(layer_id, dtype=dtype, rngs=rngs, mesh=mesh)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        hidden_states.at[forward_batch.positions == 0].set(0)
        layers_kv_fused = []

        hidden_states = self.input_proj(
            jnp.concatenate((self.hidden_layernorm(forward_batch.spec_info.hidden_states), self.token_layernorm(hidden_states)), axis = -1)
        )
        hidden_states, residual, kv_fused, _ = self.mtp_layers(forward_batch.positions, hidden_states, forward_batch, token_to_kv_pool, None)

        layers_kv_fused.append(kv_fused)
        hidden_states = hidden_states + residual
        
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, layers_kv_fused

class MiMoMTPForCausalLM(nnx.module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.model = MiMoMTPLayer(config, dtype=self.dtype, rngs=rngs, mesh=mesh)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.load_lm_head_from_target = True
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)
        
    def _create_mimo_weight_mappings(self) -> dict:
        mappings = {}

        num_mtp_layers = getattr(self, "_num_mtp_layers", 0)
        if not num_mtp_layers:
            num_mtp_layers = len(getattr(self.model, "mtp_layers", []))

        for layer_idx in range(num_mtp_layers):
            mappings.update(self._create_mtp_layer_mappings(layer_idx))

        return mappings

    def _create_mtp_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.mtp_layers.{layer_idx}"
        target_prefix = "model.mtp_layers"

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
            f"{prefix}.hidden_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.hidden_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.token_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.token_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.final_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.final_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.input_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.input_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
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

        if getattr(self.config, "attention_bias", True):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=False,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=False,
                        kv_head_padding=True,
                    ),
                }
            )

        return mappings

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = super()._create_qwen2_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMo MTP weights loaded successfully!")

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=None
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=None,
            )

        return output, layers_kv_fused, []
    
    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """

        # Set embedding weight
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight

        # Set LM Head weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight


EntryClass = MiMoMTPForCausalLM