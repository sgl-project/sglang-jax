import logging
from typing import Any

from flax import nnx
from jax import jax
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import Qwen3MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class QWen3MoeAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_norm = RMSNorm(
            self.head_dim,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            params_dtype=dtype,
        )
        self.c_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            rngs=rngs,
            params_dtype=dtype,
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
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class QWen3MoeDecoderLayer(nnx.Module):
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
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        head_dim = getattr(config, "head_dim", None)

        self.self_attn = QWen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            rngs=rngs,
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])

        if layer_id in mlp_only_layers:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                rngs=rngs,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            num_experts = getattr(config, "num_experts", 128)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
            moe_intermediate_size = getattr(config, "moe_intermediate_size", 768)
            expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                features=num_experts,
                model_name=getattr(config, "model_name", "qwen3_moe"),
                use_bias=False,
                kernel_axes=(None, ("data", "tensor")),
                dtype=dtype,
                layer_id=layer_id,
                rngs=rngs,
            )
            with mesh:
                self.mlp = EPMoE(
                    config=config,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_dim=moe_intermediate_size,
                    mesh=mesh,
                    expert_parallel_size=expert_parallel_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    rngs=rngs,
                )
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
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
            router_logits = self.moe_gate(hidden_states)
            mlp_output = self.mlp(hidden_states, router_logits=router_logits)
            hidden_states = mlp_output
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, kv_fused


class QWen3MoeModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = nnx.data(
            [
                QWen3MoeDecoderLayer(
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
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
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
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused


class Qwen3MoeForCausalLM(nnx.Module):
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
        logger.info("QWen3MoeForCausalLMModel config dtype: %s", self.dtype)
        self.transformer = QWen3MoeModel(config, dtype=self.dtype, rngs=rngs, mesh=mesh)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, rngs=rngs)
        self.logits_processor = LogitsProcessor(config.vocab_size, self.lm_head, self.mesh)

    def load_weights(self, model_config: ModelConfig, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen3_moe_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3Moe weights loaded successfully!")

    def _create_qwen3_moe_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="transformer.embed_tokens.embedding",
                sharding=(None, None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="transformer.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=(None, None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        mlp_only_layers = getattr(self.config, "mlp_only_layers", [])

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx, layer_idx in mlp_only_layers
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int, is_mlp_layer: bool) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"transformer.layers.{layer_idx}"

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
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.c_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(bias_mappings)

        if is_mlp_layer:
            mlp_mappings = {
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
            mappings.update(mlp_mappings)
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, ("data", "tensor")),
                transpose=True,
            )

            num_experts = getattr(self.config, "num_experts", 128)
            for expert_type in ["gate_proj", "up_proj", "down_proj"]:
                target_name = {
                    "gate_proj": "wi_0",
                    "up_proj": "wi_1",
                    "down_proj": "wo",
                }[expert_type]
                expert_keys = [
                    f"{prefix}.mlp.experts.{i}.{expert_type}.weight" for i in range(num_experts)
                ]

                mappings[f"__MOE_EXPERTS__{prefix}.mlp.{target_name}"] = WeightMapping(
                    target_path=[f"{target_prefix}.mlp.{target_name}"] + expert_keys,
                    sharding=(("data", "tensor"), None, None),
                    transpose=True,
                )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.transformer(forward_batch, token_to_kv_pool)
        output = self.logits_processor(hidden_states, logits_metadata)
        return output, layers_kv_fused, True


EntryClass = Qwen3MoeForCausalLM
