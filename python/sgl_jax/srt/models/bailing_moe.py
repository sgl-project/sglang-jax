import logging
from typing import Any, Dict, Optional, Tuple

from flax import nnx
from jax import jax
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class BailingMoEAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
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

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = nnx.RMSNorm(
                self.head_dim,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None,)),
                rngs=rngs,
            )
            self.k_norm = nnx.RMSNorm(
                self.head_dim,
                epsilon=rms_norm_eps,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None,)),
                rngs=rngs,
            )
        else:
            self.q_norm = None
            self.k_norm = None

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
            rotary_dim=rotary_dim,
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

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )

        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class BailingMoEMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            rngs=rngs,
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jnp.ndarray):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class BailingMoEGate(nnx.Module):
    def __init__(
        self,
        input_size: int,
        weight_dtype: jnp.dtype = jnp.float32,
        enable_expert_bias: bool = False,
        num_experts: int = 0,
        score_func: str = "",
        rngs: nnx.Rngs = None,
    ):
        self.weight_dtype = weight_dtype
        self.enable_expert_bias = enable_expert_bias
        self.score_func = score_func

        self.kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), (None, None))(
                rngs.params(), (input_size, num_experts), self.weight_dtype
            )
        )
        if enable_expert_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.zeros_init(), (None,))(
                    rngs.params(), (num_experts,), self.weight_dtype
                )
            )
        else:
            self.bias = None

    def __call__(
        self, hidden_states: jax.Array
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        logits = jnp.dot(hidden_states.astype(self.weight_dtype), self.kernel.value)
        return logits


class BailingMoEDecoderLayer(nnx.Module):
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
        use_qk_norm = getattr(config, "use_qk_norm", False)
        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = int(head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            rotary_dim = config.rotary_dim
        else:
            rotary_dim = head_dim

        self.self_attn = BailingMoEAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            rngs=rngs,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

        if layer_id < first_k_dense_replace:
            self.mlp = BailingMoEMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                rngs=rngs,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            num_shared_experts = getattr(config, "num_shared_experts", 0)
            expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get(
                "tensor", 1
            )
            router_dtype = getattr(config, "router_dtype", None)
            if router_dtype is None:
                router_dtype = None
            elif router_dtype == "fp32":
                router_dtype = jnp.float32
            else:
                router_dtype = jnp.bfloat16
            self.moe_gate = BailingMoEGate(
                input_size=config.hidden_size,
                num_experts=config.num_experts,
                enable_expert_bias=getattr(
                    config, "moe_router_enable_expert_bias", False
                ),
                weight_dtype=router_dtype,
                score_func=getattr(config, "score_function", "sigmoid"),
                rngs=rngs,
            )
            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                routed_scaling_factor=config.routed_scaling_factor,
            )
            self.mlp = EPMoE(
                config=config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                expert_parallel_size=expert_parallel_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                rngs=rngs,
            )
            if num_shared_experts > 0:
                self.shared_experts = BailingMoEMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=getattr(
                        config,
                        "moe_shared_expert_intermediate_size",
                        config.moe_intermediate_size,
                    )
                    * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    rngs=rngs,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = nnx.RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rngs,
        )
        self.post_attention_layernorm = nnx.RMSNorm(
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
        residual: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
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
            correction_bias = (
                self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            )
            topk_ids, topk_weights = self.topk(router_logits, correction_bias)
            hidden_states = self.mlp(hidden_states, topk_ids, topk_weights)
            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual, kv_fused


class BailingMoEModel(nnx.Module):
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

        self.layers = [
            BailingMoEDecoderLayer(
                config=config,
                layer_id=i,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.norm = nnx.RMSNorm(
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


class BailingMoEForCausalLM(nnx.Module):
    def __init__(
        self,
        config: ModelConfig,
        rngs: nnx.Rngs = None,
        mesh: jax.sharding.Mesh = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = config.dtype
        self.model = BailingMoEModel(
            config.hf_config, dtype=self.dtype, rngs=rngs, mesh=mesh
        )
        self.lm_head = ParallelLMHead(
            config.hf_config.vocab_size, config.hidden_size, rngs=rngs
        )
        self.logits_processor = LogitsProcessor(
            config.hf_config.vocab_size, self.lm_head, self.mesh
        )

    def load_weights(self, rng_key: jax.Array):
        self.rng = nnx.Rngs(rng_key)
        loader = WeightLoader(
            model=self,
            model_config=self.config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_bailing_moe_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Weights loaded successfully!")

    def _create_bailing_moe_weight_mappings(self) -> dict:
        mappings = {
            "model.word_embeddings.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=(None, None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config.hf_config, "tie_word_embeddings", True):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=(None, None), transpose=False
            )

        num_layers = self.config.hf_config.num_hidden_layers
        first_k_dense_replace = self.config.hf_config.first_k_dense_replace

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx, layer_idx < first_k_dense_replace
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int, is_mlp_layer: bool) -> dict:
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
            f"{prefix}.attention.query_key_value.weight": WeightMapping(
                target_path=[
                    f"{target_prefix}.self_attn.q_proj.weight",
                    f"{target_prefix}.self_attn.k_proj.weight",
                    f"{target_prefix}.self_attn.v_proj.weight",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.attention.dense.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.c_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.config.hf_config, "use_qk_norm", True):
            mappings[f"{prefix}.attention.query_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.attention.key_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            )

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
                sharding=(None, None),
                transpose=True,
            )
            if getattr(self.config.hf_config, "moe_router_enable_expert_bias", False):
                mappings[f"{prefix}.mlp.gate.expert_bias"] = WeightMapping(
                    target_path=f"{target_prefix}.moe_gate.bias",
                    sharding=(None,),
                    transpose=False,
                )

            if getattr(self.config.hf_config, "num_shared_experts", 0) > 0:
                shared_experts_mappings = {
                    f"{prefix}.mlp.shared_experts.gate_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.shared_experts.up_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.shared_experts.down_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
                mappings.update(shared_experts_mappings)

            num_experts = getattr(self.config.hf_config, "num_experts", 256)
            for expert_type in ["gate_proj", "up_proj", "down_proj"]:
                target_name = {
                    "gate_proj": "wi_0",
                    "up_proj": "wi_1",
                    "down_proj": "wo",
                }[expert_type]
                expert_keys = [
                    f"{prefix}.mlp.experts.{i}.{expert_type}.weight"
                    for i in range(num_experts)
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
        hidden_states, layers_kv_fused = self.model(forward_batch, token_to_kv_pool)
        output = self.logits_processor(hidden_states, logits_metadata)
        return output, layers_kv_fused, True


class BailingMoeForCausalLM(BailingMoEForCausalLM):
    pass


class BailingMoeV2ForCausalLM(BailingMoEForCausalLM):
    pass


EntryClass = [BailingMoEForCausalLM, BailingMoeForCausalLM, BailingMoeV2ForCausalLM]
