import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class DeepseekV2MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class DeepseekV2Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id
        assert (
            num_heads % num_kv_heads == 0
        ), "Please use other tp partition strategy for this model."
        # DeepSeek-V2 MLA: Q/K have larger head_dim, V has smaller head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # Q/K use full head_dim (qk_nope + qk_rope), V uses v_head_dim
        self.head_dim = head_dim or (qk_nope_head_dim + qk_rope_head_dim)  # 192
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * v_head_dim
        self.scaling = v_head_dim**-0.5  # Use v_head_dim for scaling

        # Q projection (standard)
        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,  # 16 * 192 = 3072
            use_bias=False,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * v_head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        # RoPE is applied only to the rope part (64 dims for DeepSeek-V2)
        self.rotary_emb = RotaryEmbedding(
            head_size=qk_rope_head_dim,  # 64 dims for RoPE
            rotary_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        # MLA (Multi-Head Latent Attention) layers for KV compression
        kv_lora_rank = 512  # Compressed KV dimension
        self.kv_lora_rank = kv_lora_rank
        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=hidden_size,
            output_size=kv_lora_rank + qk_rope_head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )
        # kv_b_proj outputs: k_nope (num_kv_heads * qk_nope_head_dim) + v (num_kv_heads * v_head_dim)
        # = 16 * 128 + 16 * 128 = 4096 for DeepSeek-V2-Lite
        self.kv_b_proj = LinearBase(
            input_size=kv_lora_rank,
            output_size=num_kv_heads * (qk_nope_head_dim + v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
            mesh=mesh,
        )

        # For attention, we use v_head_dim since that's what gets cached and output
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=v_head_dim,  # Use v_head_dim for KV cache
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        batch_size = hidden_states.shape[0]
        
        # Q projection: output (B, num_heads * (qk_nope_head_dim + qk_rope_head_dim))
        q, _ = self.q_proj(hidden_states)
        q = q.reshape(batch_size, self.q_head_num, self.head_dim)  # (B, 16, 192)
        
        # MLA: Compressed KV projection
        compressed_kv, _ = self.kv_a_proj_with_mqa(hidden_states)  # (B, kv_lora_rank + qk_rope_head_dim)
        
        # Split: kv_lora_rank for compression + qk_rope_head_dim for RoPE
        compressed = compressed_kv[:, :self.kv_lora_rank]  # (B, 512)
        k_pe = compressed_kv[:, self.kv_lora_rank:]  # (B, 64) for positional encoding
        
        # Normalize and expand compressed KV
        compressed = self.kv_a_layernorm(compressed)
        kv_full, _ = self.kv_b_proj(compressed)  # (B, num_kv_heads * (qk_nope_head_dim + v_head_dim))
        
        # Split k_nope and v from kv_b_proj output
        # kv_full shape: (B, num_kv_heads * (qk_nope_head_dim + v_head_dim))
        kv_full = kv_full.reshape(batch_size, self.kv_head_num, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_full[:, :, :self.qk_nope_head_dim]  # (B, 16, 128)
        v = kv_full[:, :, self.qk_nope_head_dim:]  # (B, 16, 128)
        
        # Apply RoPE to q_rope and k_pe
        k_pe = k_pe.reshape(batch_size, 1, self.qk_rope_head_dim)  # (B, 1, 64) - MQA style
        k_pe = jnp.repeat(k_pe, self.kv_head_num, axis=1)  # (B, 16, 64)
        
        # Split Q into nope and rope parts
        q_nope = q[:, :, :self.qk_nope_head_dim]  # (B, 16, 128)
        q_rope = q[:, :, self.qk_nope_head_dim:]  # (B, 16, 64)
        
        # Apply RoPE to rope parts
        q_rope, k_pe = self.rotary_emb(positions, q_rope, k_pe)
        
        # For simplified MLA attention: 
        # Q = [q_nope, q_rope], K = [k_nope, k_pe], V = v
        # But FlashAttention expects same dimension for Q/K/V
        # Use v_head_dim for all, truncating Q and K
        q_attn = q_nope[:, :, :self.v_head_dim]  # (B, 16, 128)
        k_attn = k_nope[:, :, :self.v_head_dim]  # (B, 16, 128)
        v_attn = v  # (B, 16, 128)
        
        attn_output, kv_fused = self.attn(q_attn, k_attn, v_attn, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class DeepseekV2DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 128)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        v_head_dim = getattr(config, "v_head_dim", 128)
        self.self_attn = DeepseekV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        # Layer 0 uses standard MLP, other layers use MoE
        num_experts = getattr(config, "n_routed_experts", 64)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
        ep_size = getattr(config, "ep_size", 1)
        shared_intermediate_size = getattr(config, "shared_expert_intermediate_size", config.intermediate_size)
        
        # Always create shared_experts for all layers to ensure nnx.eval_shape can properly
        # evaluate the model structure. Layer 0 won't use it in forward pass.
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=shared_intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        
        if layer_id == 0:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            )
            # Layer 0 doesn't use MoE components
            self.moe_gate = None
            self.topk = None
        else:
            # MoE gate and topk
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=num_experts,
                weight_dtype=dtype,
            )
            self.topk = TopK(
                topk=num_experts_per_tok,
                renormalize=getattr(config, "norm_topk_prob", True),
            )
            
            # MoE experts
            moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)
            self.mlp = EPMoE(
                config=config,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                ep_size=ep_size,
                mesh=mesh,
                intermediate_dim=moe_intermediate_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ):
        layer_callback_flag = []
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Layer 0 uses standard MLP, other layers use MoE
        if self.layer_id == 0:
            # Layer 0 doesn't use shared_experts even though it's created for structure consistency
            hidden_states = self.mlp(hidden_states)
        else:
            # MoE forward pass
            router_logits = self.moe_gate(hidden_states)
            topk_weights, topk_ids = self.topk(router_logits)
            
            # Shared experts output
            shared_output = self.shared_experts(hidden_states)
            
            # MoE experts output
            moe_output = self.mlp(hidden_states, topk_weights, topk_ids)
            
            # Combine shared and MoE outputs
            hidden_states = shared_output + moe_output

        return hidden_states, residual, kv_fused, layer_callback_flag


class DeepseekV2Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                DeepseekV2DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    rngs=rngs,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag


class DeepseekV2ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("DeepseekV2ForCausalLM config dtype: %s", self.dtype)
        self.model = DeepseekV2Model(config, dtype=self.dtype, rngs=rngs, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                rngs=rngs,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_deepseek2_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("DeepseekV2 weights loaded successfully!")

    def _create_deepseek2_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
            "model.lm_head.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            ),
        }

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
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
            f"{prefix}.self_attn.kv_a_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.self_attn.kv_b_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
        }
        
        # DeepSeek Layer 0 FNN - Standard MLP (no mqa) 
        prefix_mlp = f"{prefix}.mlp"
        target_prefix_mlp = f"{target_prefix}.mlp"
        if (layer_idx == 0):
            mappings.update({
                f"{prefix_mlp}.down_proj.weight": WeightMapping(
                    target_path=f"{target_prefix_mlp}.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
                f"{prefix_mlp}.gate_proj.weight": WeightMapping(
                    target_path=f"{target_prefix_mlp}.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
            mappings.update({
                f"{prefix_mlp}.up_proj.weight": WeightMapping(
                    target_path=f"{target_prefix_mlp}.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
            })
        else:
            # Add shared_experts mappings for layers > 0
            # Note: shared_experts is a direct attribute of DecoderLayer, not under mlp
            if layer_idx > 0:
                mappings.update({
                    f"{prefix_mlp}.shared_experts.down_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                    f"{prefix_mlp}.shared_experts.gate_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix_mlp}.shared_experts.up_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                })
            
            # MoE experts mappings - use __MOE_EXPERTS__ prefix for grouped loading
            num_experts = getattr(self.config, "n_routed_experts", 64)
            for expert_type in ["gate_proj", "up_proj", "down_proj"]:
                target_name = {
                    "gate_proj": "wi_0",
                    "up_proj": "wi_1",
                    "down_proj": "wo",
                }[expert_type]
                
                expert_keys = [
                    f"{prefix}.mlp.experts.{i}.{expert_type}.weight" for i in range(num_experts)
                ]
                
                if expert_type == "down_proj":
                    sharding = ("expert", "tensor", None)
                else:
                    sharding = ("expert", None, "tensor")
                
                mappings[f"__MOE_EXPERTS__{prefix}.mlp.{target_name}"] = WeightMapping(
                    target_path=[f"{target_prefix}.mlp.{target_name}"] + expert_keys,
                    sharding=sharding,
                    transpose=True,
                )
            
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )

        return mappings

    
    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag


EntryClass = DeepseekV2ForCausalLM
