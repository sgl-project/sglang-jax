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


class Qwen2MoeMLP(nnx.Module):
    """Qwen2 MoE MLP layer with gate, up, and down projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        gate_up_down_bias: bool | None = False,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=gate_up_down_bias,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Qwen2MoeAttention(nnx.Module):
    """Qwen2 MoE attention layer with QKV projections and output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        qkv_bias: bool | None = True,
        o_bias: bool | None = False,
    ):
        self.layer_id = layer_id
        assert (
            num_heads % num_kv_heads == 0
        ), "Please use other tp partition strategy for this model."
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=qkv_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=o_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
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
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Qwen2MoeDecoderLayer(nnx.Module):
    """Qwen2 MoE decoder layer with attention and MoE/MLP."""

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
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)

        self.self_attn = Qwen2MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            layer_id=layer_id,
            dtype=dtype,
            qkv_bias=getattr(config, "qkv_bias", True),
            o_bias=getattr(config, "o_bias", False),
            mesh=mesh,
        )

        # Use MoE for all layers
        num_experts = getattr(config, "num_experts", 8)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=num_experts,
            weight_dtype=dtype,
        )

        self.moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = self.moe_backend == "fused"

        if self.use_fused:
            from sgl_jax.srt.layers.fused_moe import FusedEPMoE

            self.mlp = FusedEPMoE(
                config=config,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=getattr(config, "norm_topk_prob", True),
            )
        else:
            self.topk = TopK(
                topk=num_experts_per_tok,
                renormalize=getattr(config, "norm_topk_prob", True),
            )
            self.mlp = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                intermediate_dim=moe_intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
            )

        # Optional shared expert path
        shared_sz = getattr(config, "shared_expert_intermediate_size", 0)
        if shared_sz and shared_sz > 0:
            self.shared_experts = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_sz,
                layer_id=layer_id,
                dtype=dtype,
                gate_up_down_bias=False,
                mesh=mesh,
            )
            self.shared_expert_gate = LinearBase(
                input_size=config.hidden_size,
                output_size=1,
                use_bias=False,
                kernel_axes=(None, None),
                params_dtype=dtype,
                mesh=mesh,
            )
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:

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

        # optional shared expert output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            if self.shared_expert_gate is not None:
                gate, _ = self.shared_expert_gate(hidden_states)
                shared_output = jax.nn.sigmoid(gate) * shared_output
        else:
            shared_output = None

        router_logits = self.moe_gate(hidden_states)
        if self.use_fused:
            mlp_output = self.mlp(hidden_states, router_logits)
        else:
            topk_weights, topk_ids = self.topk(router_logits)
            mlp_output = self.mlp(hidden_states, topk_weights, topk_ids)

        hidden_states = mlp_output if shared_output is None else (mlp_output + shared_output)

        return hidden_states, residual, kv_fused


class Qwen2MoeModel(nnx.Module):
    """Qwen2 MoE model with embedding, layers, and normalization."""

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
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Qwen2MoeDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        layers_kv_fused = []

        for layer in self.layers:
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused


class Qwen2MoeForCausalLM(nnx.Module):
    """Qwen2 MoE model for causal language modeling."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("Qwen2MoeForCausalLM config dtype: %s", self.dtype)
        self.model = Qwen2MoeModel(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", True):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_qwen2_moe_weight_mappings()

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen2Moe weights loaded successfully!")

    def _create_qwen2_moe_weight_mappings(self) -> dict:
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

        if not getattr(self.config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int) -> dict:
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
        }

        # Add bias mappings if attention_bias is True
        if getattr(self.config, "attention_bias", True):
            bias_mappings = {
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
            mappings.update(bias_mappings)

        # MoE mappings for expert layers
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target_prefix}.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )

        # Optional shared expert weight mapping (singular in source naming)
        if (
            getattr(self.config, "shared_expert_intermediate_size", 0)
            and getattr(self.config, "shared_expert_intermediate_size", 0) > 0
        ):
            shared_expert_mappings = {
                f"{prefix}.mlp.shared_expert.gate_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.shared_expert.up_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.shared_expert.down_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
                f"{prefix}.mlp.shared_expert_gate.weight": WeightMapping(
                    target_path=f"{target_prefix}.shared_expert_gate.weight",
                    sharding=(None, None),
                    transpose=True,
                ),
            }
            mappings.update(shared_expert_mappings)

        num_experts = getattr(self.config, "num_experts", 8)
        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        use_fused = moe_backend == "fused"

        if use_fused:
            # Fused MoE Mapping
            # w1: gate_proj -> (num_experts, hidden, intermediate)
            # w3: up_proj   -> (num_experts, hidden, intermediate)
            # w2: down_proj -> (num_experts, intermediate, hidden)

            target_path_w1 = [f"{target_prefix}.mlp.w1"]
            target_path_w1.extend(
                [f"{prefix}.mlp.experts.{i}.gate_proj.weight" for i in range(num_experts)]
            )
            mappings[f"__MOE_EXPERTS__{prefix}.mlp.w1"] = WeightMapping(
                target_path=target_path_w1,
                sharding=("tensor", None, None),  # (E, H, I)
                transpose=True,
            )

            target_path_w3 = [f"{target_prefix}.mlp.w3"]
            target_path_w3.extend(
                [f"{prefix}.mlp.experts.{i}.up_proj.weight" for i in range(num_experts)]
            )
            mappings[f"__MOE_EXPERTS__{prefix}.mlp.w3"] = WeightMapping(
                target_path=target_path_w3,
                sharding=("tensor", None, None),  # (E, H, I)
                transpose=True,
            )

            # 2. w2 (down)
            target_path_w2 = [f"{target_prefix}.mlp.w2"]
            target_path_w2.extend(
                [f"{prefix}.mlp.experts.{i}.down_proj.weight" for i in range(num_experts)]
            )

            mappings[f"__MOE_EXPERTS__{prefix}.mlp.w2"] = WeightMapping(
                target_path=target_path_w2,
                sharding=("tensor", None, None),  # (E, I, H)
                transpose=True,
            )
        else:
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

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)
        if not getattr(self.config, "tie_word_embeddings", True):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True


EntryClass = Qwen2MoeForCausalLM
