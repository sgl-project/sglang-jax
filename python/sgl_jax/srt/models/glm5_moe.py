import logging
from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    EPMoE,
    FusedEPMoE,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class GlmNorm(nnx.Module):
    def __init__(self, dim: int, dtype: jnp.dtype = jnp.bfloat16):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        eps = 1e-5
        normalized = (x - mean) / jnp.sqrt(variance + eps)
        return normalized * self.weight.value + self.bias.value



def get_hadamard_matrix(n):
    if n == 1:
        return jnp.array([[1.0]])
    h = get_hadamard_matrix(n // 2)
    return jnp.block([[h, h], [h, -h]])

class GlmDsaIndexer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        self.head_dim = index_head_dim
        self.n_head = index_n_heads
        self.mesh = mesh


        self.wq_b = LinearBase(
            input_size=q_lora_rank,
            output_size=index_head_dim * index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wq_b",
        )
        self.wk = LinearBase(
            input_size=hidden_size,
            output_size=index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wk",
        )
        self.k_norm = GlmNorm(index_head_dim, dtype)

        self.weights_proj = LinearBase(
            input_size=hidden_size,
            output_size=index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="weights_proj",
        )

    def __call__(self, hidden_states: jax.Array, qr: jax.Array, positions: jax.Array, rotary_emb: Any) -> jax.Array:
        # 1. Project Query and Key
        query, _ = self.wq_b(qr)
        query = query.reshape(-1, self.n_head, self.head_dim)
        
        key, _ = self.wk(hidden_states)
        key = self.k_norm(key)
        
        # Apply RoPE
        rope_dim = 64
        q_rope = query[:, :, :rope_dim]
        k_rope = key[:, :rope_dim]
        k_rope = k_rope[:, None, :] # Add head dim for RoPE
        
        q_rope, k_rope = rotary_emb(positions, q_rope, k_rope)
        k_rope = k_rope.squeeze(1) # Remove head dim
        
        query = query.at[:, :, :rope_dim].set(q_rope)
        key = key.at[:, :rope_dim].set(k_rope)
        
        # Apply Hadamard Transform
        h_matrix = get_hadamard_matrix(128)
        h_matrix = h_matrix * (128**-0.5)
        
        query = jnp.einsum("thd,de->the", query, h_matrix)
        key = jnp.einsum("td,de->te", key, h_matrix)
        
        # 2. Compute Logits (simplified dense dot product)
        key_replicated = jax.sharding.reshard(key, jax.sharding.NamedSharding(self.mesh, P(None, None)))
        logits = jnp.einsum("thd,sd->ths", query, key_replicated)
        
        # 3. Apply weights_proj
        weights, _ = self.weights_proj(hidden_states)
        
        # Scale and apply weights
        scaling = self.head_dim**-0.5
        logits = logits * scaling * weights[:, :, None]
        
        # 4. Top-K Selection (Top-1 for now to match dummy shape [T, n_head])
        _, topk_ids = jax.lax.top_k(logits, 1)
        topk_ids = topk_ids.squeeze(-1)
        
        return topk_ids



class Glm5Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.qk_nope_head_dim = 192
        self.qk_rope_head_dim = 64
        self.qk_head_dim = 256
        self.v_head_dim = 256
        self.kv_lora_rank = 512
        self.q_lora_rank = 2048

        self.scaling = 256**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm")
            self.k_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm")
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_a_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_lora_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_a_layernorm"
        )
        self.q_b_proj = LinearBase(
            input_size=self.q_lora_rank,
            output_size=num_heads * self.qk_head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="kv_a_layernorm"
        )

        # kv_b_proj is initialized but will be dropped in post_load_weights
        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            input_size=num_heads * self.v_head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        self.indexer = GlmDsaIndexer(
            hidden_size=hidden_size,
            q_lora_rank=self.q_lora_rank,
            index_head_dim=128,
            index_n_heads=32,
            mesh=mesh,
            dtype=dtype,
            scope_name="indexer",
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype=dtype,
            mesh=mesh,
        )

        # Absorbed MLA placeholders
        uk_axes = (None, "tensor", None)
        self.w_uk = nnx.Param(
            jnp.zeros(
                (self.kv_lora_rank, num_heads, self.qk_nope_head_dim),
                dtype=dtype,
                out_sharding=P(*uk_axes),
            )
        )
        self.w_uv = nnx.Param(
            jnp.zeros(
                (self.kv_lora_rank, num_heads, self.v_head_dim),
                dtype=dtype,
                out_sharding=P(*uk_axes),
            )
        )

        # MQA attention on latent states
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,  # 576
            scaling=self.scaling,
            num_kv_heads=1,
            v_head_dim=self.kv_lora_rank,  # 512
            layer_id=layer_id,
        )

    def post_load_weights(self):
        """Split kv_b_proj.weight into absorbed-MLA folded projections."""
        if self.kv_b_proj is None:
            return
        w_kv = self.kv_b_proj.weight.value.reshape(
            self.kv_lora_rank,
            self.q_head_num,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.w_uk.value = w_kv[:, :, : self.qk_nope_head_dim]
        self.w_uv.value = w_kv[:, :, self.qk_nope_head_dim :]
        self.kv_b_proj = None

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        # 1. Q projection
        q_compressed, _ = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.q_head_num, self.qk_head_dim)
        
        # Call indexer (result not used yet)
        _ = self.indexer(hidden_states, q_compressed, positions, self.rotary_emb)

        q_nope = q[:, :, : self.qk_nope_head_dim]

        q_rope = q[:, :, self.qk_nope_head_dim :]

        # 2. KV projection (latent)
        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed, k_rope = jnp.split(latent_cache, [self.kv_lora_rank], axis=-1)
        compressed = self.kv_a_layernorm(compressed)

        k_rope = k_rope.reshape(-1, 1, self.qk_rope_head_dim)

        # 3. Apply RoPE
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        # ql_nope[t, h, r] = sum_d q_nope[t, h, d] * w_uk[r, h, d]
        ql_nope = jnp.einsum("thd,rhd->thr", q_nope, self.w_uk.value)

        # Latent K/V are a single shared head — pack into [T, 1, *] for MQA.
        c_kv_3d = compressed[:, None, :]

        attn_output, kv_fused = self.attn(
            ql_nope,
            c_kv_3d,
            c_kv_3d,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            q_rope=q_rope,
            k_rope=k_rope,
        )

        # o_v[t, h, d] = sum_r attn_output[t, h, r] * w_uv[r, h, d]
        o_v = jnp.einsum("thr,rhd->thd", attn_output, self.w_uv.value)

        attn_output = o_v.reshape(-1, self.q_head_num * self.v_head_dim)

        output, _ = self.o_proj(attn_output)

        return output, kv_fused


class Glm5MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Glm5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.head_dim = getattr(config, "head_dim", None) or 128
        use_qk_norm = getattr(config, "use_qk_norm", True)

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        self.self_attn = Glm5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

        if layer_id < first_k_dense_replace:
            self.mlp = Glm5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            router_dtype = jnp.float32
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                enable_expert_bias=True,
                weight_dtype=router_dtype,
                score_func=getattr(config, "scoring_func", "sigmoid"),
            )

            self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
            self.use_fused = self.moe_backend == MoEBackend.FUSED

            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_id,
            )

            if self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 1) > 1,
                    num_groups=getattr(config, "n_group", 1),
                    top_k_groups=getattr(config, "topk_group", 1),
                    num_shared_experts=getattr(config, "n_shared_experts", 0),
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            num_shared_experts = getattr(config, "n_shared_experts", 0)
            if num_shared_experts > 0 and not self.use_fused:
                self.shared_experts = Glm5MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array]:
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

        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class Glm5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Glm5DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids


class Glm5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Glm5Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )

        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_glm5_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
        logger.info("Absorbed MLA weights split successfully!")

        # Skipping scale inversion for BF16
        logger.info("Skipping scale inversion for BF16 model.")

    def _create_glm5_weight_mappings(self, model_config: ModelConfig) -> dict:
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

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        hf_layer_indices = list(range(num_layers))
        for layer_idx in range(num_layers):
            target_idx = hf_layer_indices[layer_idx]
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx,
                target_idx,
                target_idx < first_k_dense_replace,
                is_static_quant=is_static_quant,
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(
        self, layer_idx: int, target_idx: int, is_mlp_layer: bool, is_static_quant: bool = False
    ) -> dict:
        prefix = f"model.layers.{target_idx}"
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
        }

        w_name = "weight_q" if is_static_quant else "weight"

        # Attention mappings (separate Q, K, V in checkpoint)
        # Attention mappings (MLA)
        mappings[f"{prefix}.self_attn.q_a_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_a_proj.{w_name}",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.q_b_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_b_proj.{w_name}",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.{w_name}",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_b_proj.{w_name}",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.o_proj.{w_name}",
            sharding=("tensor", None),
            transpose=True,
        )

        # Indexer mappings
        mappings[f"{prefix}.self_attn.indexer.wq_b.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.wq_b.{w_name}",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.indexer.wk.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.wk.{w_name}",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.indexer.weights_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.weights_proj.{w_name}",
            sharding=(None, None),
            transpose=True,
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.weight",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.bias"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.bias",
            sharding=(None,),
        )

        if is_static_quant:
            mappings[f"{prefix}.self_attn.q_a_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.q_b_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_b_proj.weight_scale",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_b_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight_scale",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.indexer.wk.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.indexer.wk.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.indexer.wq_b.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.indexer.wq_b.weight_scale",
                sharding=(None,),
                transpose=False,
            )

        # DSA Indexer Norm
        mappings[f"{prefix}.self_attn.indexer.k_norm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.weight", sharding=(None,)
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.bias"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.bias", sharding=(None,)
        )

        if is_mlp_layer:
            mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.{w_name}",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.{w_name}",
                sharding=(None, "tensor"),
                transpose=True,
            )
            mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.{w_name}",
                sharding=("tensor", None),
                transpose=True,
            )
            if is_static_quant:
                mappings[f"{prefix}.mlp.gate_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.up_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.down_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # GLM-4 uses e_score_correction_bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias", sharding=(None,)
            )

            num_logical_experts = self.config.n_routed_experts
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("gate_proj", "up_proj", "down_proj"),
                moe_backend=moe_backend,
                physical_to_logical_map=None,  # Handle physical mapping if needed later
            )

            if is_static_quant:
                new_moe_mappings = {}

                for key, mapping in moe_mappings.items():
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]

                    new_moe_mappings[key] = WeightMapping(
                        target_path=[target_param] + src_paths,
                        sharding=mapping.sharding,
                        transpose=True,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    scale_src_paths = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]

                    # For GLM-5 FP8, scales are stored as [num_experts, in_blocks, out_blocks]
                    # We need to transpose them to [num_experts, out_blocks, in_blocks] for moe.py
                    new_moe_mappings[scale_key] = WeightMapping(
                        target_path=[target_scale_param] + scale_src_paths,
                        sharding=None,
                        transpose=False,
                        transpose_axes=(0, 2, 1),
                        reshape=None,
                        repeat=None,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = new_moe_mappings

            mappings.update(moe_mappings)

            num_shared = getattr(self.config, "n_shared_experts", 0)
            if num_shared > 0:
                mappings[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.{w_name}",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.shared_experts.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.{w_name}",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{prefix}.mlp.shared_experts.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.{w_name}",
                    sharding=("tensor", None),
                    transpose=True,
                )
                if is_static_quant:
                    mappings[f"{prefix}.mlp.shared_experts.gate_proj.weight_scale_inv"] = (
                        WeightMapping(
                            target_path=f"{target_prefix}.shared_experts.gate_proj.weight_scale",
                            sharding=(None,),
                            transpose=False,
                        )
                    )
                    mappings[f"{prefix}.mlp.shared_experts.up_proj.weight_scale_inv"] = (
                        WeightMapping(
                            target_path=f"{target_prefix}.shared_experts.up_proj.weight_scale",
                            sharding=(None,),
                            transpose=False,
                        )
                    )
                    mappings[f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv"] = (
                        WeightMapping(
                            target_path=f"{target_prefix}.shared_experts.down_proj.weight_scale",
                            sharding=(None,),
                            transpose=False,
                        )
                    )

        return mappings


class GlmMoeDsaForCausalLM(Glm5ForCausalLM):
    pass


EntryClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM]
