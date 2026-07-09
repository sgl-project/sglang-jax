from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class DFlashAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // config.num_attention_heads))
        self.q_head_num = int(config.num_attention_heads)
        self.kv_head_num = int(getattr(config, "num_key_value_heads", self.q_head_num))
        self.q_size = self.q_head_num * self.head_dim
        self.kv_size = self.kv_head_num * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.q_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )
        self.k_norm = RMSNorm(
            self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.q_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 32768)),
            base=float(getattr(config, "rope_theta", 1000000)),
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
        )

    @named_scope
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

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        output, _ = self.o_proj(attn_output)
        return output, kv_fused

    def dense_prefix_attention(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        prefix_cache_loc: jax.Array,
        prefix_lens: jax.Array,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        """Non-causal DFlash block attention using gathered prefix K/V.

        The TPU RPA path currently collapses DFlash mask-query outputs when a
        non-causal TARGET_VERIFY block has a prefix. DFlash only needs a small
        fixed query block, so this dense path gathers the already-materialized
        prefix K/V and performs qK attention directly.
        """
        q, _ = self.q_proj(hidden_states)
        k_noise, _ = self.k_proj(hidden_states)
        v_noise, _ = self.v_proj(hidden_states)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k_noise = k_noise.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v_noise = v_noise.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q = self.q_norm(q)
        k_noise = self.k_norm(k_noise)
        q, k_noise = self.rotary_emb(positions, q, k_noise)

        bs = prefix_lens.shape[0]
        block = hidden_states.shape[0] // bs
        prefix_cache_loc = prefix_cache_loc.reshape(bs, -1)
        prefix_width = prefix_cache_loc.shape[1]

        fused = token_to_kv_pool.get_fused_kv_buffer(self.layer_id)
        fused_flat = jax.lax.reshape(
            fused,
            (
                fused.shape[0] * fused.shape[1],
                fused.shape[2] * fused.shape[3],
                fused.shape[4],
            ),
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        prefix_fused = fused_flat.at[prefix_cache_loc.reshape(-1)].get(
            out_sharding=NamedSharding(self.mesh, P(None, None, None))
        )
        prefix_fused = prefix_fused.reshape(
            bs, prefix_width, prefix_fused.shape[-2], prefix_fused.shape[-1]
        )
        prefix_k = prefix_fused[:, :, 0 : self.kv_head_num * 2 : 2, : self.head_dim]
        prefix_v = prefix_fused[:, :, 1 : self.kv_head_num * 2 : 2, : self.head_dim]
        kv_block_sharding = NamedSharding(self.mesh, P(None, "data", "tensor", None))
        prefix_k = jax.sharding.reshard(prefix_k, kv_block_sharding)
        prefix_v = jax.sharding.reshard(prefix_v, kv_block_sharding)

        q = q.reshape(bs, block, self.q_head_num, self.head_dim)
        k_noise = k_noise.reshape(bs, block, self.kv_head_num, self.head_dim)
        v_noise = v_noise.reshape(bs, block, self.kv_head_num, self.head_dim)
        k_noise = jax.sharding.reshard(k_noise, kv_block_sharding)
        v_noise = jax.sharding.reshard(v_noise, kv_block_sharding)

        k_all = jnp.concatenate([prefix_k, k_noise], axis=1)
        v_all = jnp.concatenate([prefix_v, v_noise], axis=1)
        if self.q_head_num != self.kv_head_num:
            repeat = self.q_head_num // self.kv_head_num
            q_head_sharding = NamedSharding(self.mesh, P(None, "data", "tensor", None))
            k_all = jnp.repeat(k_all, repeat, axis=2, out_sharding=q_head_sharding)
            v_all = jnp.repeat(v_all, repeat, axis=2, out_sharding=q_head_sharding)

        dense_sharding = NamedSharding(self.mesh, P(None, None, None, None))
        q = jax.sharding.reshard(q, dense_sharding)
        k_all = jax.sharding.reshard(k_all, dense_sharding)
        v_all = jax.sharding.reshard(v_all, dense_sharding)

        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k_all) * self.scaling
        prefix_valid = jnp.arange(prefix_width, dtype=jnp.int32)[None, :] < prefix_lens[:, None]
        block_valid = jnp.ones((bs, block), dtype=bool)
        kv_valid = jnp.concatenate([prefix_valid, block_valid], axis=1)
        scores = jnp.where(kv_valid[:, None, None, :], scores, jnp.array(-1.0e30, scores.dtype))
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", probs, v_all)
        attn_output = attn_output.reshape(bs * block, self.q_size)
        output, _ = self.o_proj(attn_output)
        return output, token_to_kv_pool.get_fused_kv_buffer(self.layer_id)

    def kv_proj_only(self, positions: jax.Array, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array]:
        num_tokens = positions.shape[0]
        hidden_states = hidden_states[:num_tokens]
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = self.k_norm(k)
        dummy_q = jnp.zeros((num_tokens, 1, self.head_dim), dtype=k.dtype)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k, v


class DFlashMLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)

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

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = self.act_fn(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class DFlashDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.self_attn = DFlashAttention(config=config, mesh=mesh, layer_id=layer_id, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )
        self.mlp = DFlashMLP(config=config, mesh=mesh, layer_id=layer_id, dtype=dtype)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        prefix_cache_loc: jax.Array | None = None,
        prefix_lens: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if prefix_cache_loc is not None and prefix_lens is not None:
            hidden_states, kv_fused = self.self_attn.dense_prefix_attention(
                positions=positions,
                hidden_states=hidden_states,
                prefix_cache_loc=prefix_cache_loc,
                prefix_lens=prefix_lens,
                token_to_kv_pool=token_to_kv_pool,
            )
        else:
            hidden_states, kv_fused = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, kv_fused


class DFlashBackbone(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layers = nnx.data(
            [
                DFlashDecoderLayer(config=config, mesh=mesh, layer_id=i, dtype=dtype)
                for i in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(
            int(config.hidden_size),
            epsilon=float(getattr(config, "rms_norm_eps", 1e-6)),
            param_dtype=dtype,
            scope_name="norm",
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array]]:
        hidden_states = forward_batch.input_embedding
        if hidden_states is None:
            raise ValueError("DFlashDraftModel requires forward_batch.input_embedding.")

        spec_info = getattr(forward_batch, "spec_info", None)
        prefix_cache_loc = None
        prefix_lens = None
        if bool(getattr(spec_info, "dense_draft", False)):
            prefix_cache_loc = getattr(spec_info, "prefix_cache_loc", None)
            prefix_lens = getattr(spec_info, "prefix_lens", None)

        residual = None
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, residual, kv_fused = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                prefix_cache_loc,
                prefix_lens,
            )
            layers_kv_fused.append(kv_fused)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class DFlashDraftModel(nnx.Module):
    """DFlash draft transformer without owned embedding or LM head."""

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = DFlashBackbone(config=config, mesh=mesh, dtype=dtype)

        hidden_size = int(config.hidden_size)
        target_hidden_size = int(getattr(config, "target_hidden_size", hidden_size))
        target_layer_ids = getattr(config, "target_layer_ids", None)
        if target_layer_ids is None:
            dflash_config = getattr(config, "dflash_config", {}) or {}
            target_layer_ids = dflash_config.get("target_layer_ids", None)
        num_context_features = (
            len(target_layer_ids) if target_layer_ids is not None else int(config.num_hidden_layers)
        )
        self.num_context_features = int(num_context_features)
        self.target_hidden_size = target_hidden_size
        self.block_size = int(getattr(config, "block_size", 16))

        self.fc = LinearBase(
            input_size=self.num_context_features * self.target_hidden_size,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="fc",
        )
        self.hidden_norm = RMSNorm(
            hidden_size,
            epsilon=float(getattr(config, "rms_norm_eps", 1e-6)),
            param_dtype=dtype,
            scope_name="hidden_norm",
        )
        self.embed_weight = None
        self.lm_head_weight = None

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        from flax import nnx

        self.embed_weight = nnx.data(embed_weight) if embed_weight is not None else None
        self.lm_head_weight = nnx.data(head_weight) if head_weight is not None else None

    def project_target_hidden(self, target_hidden: jax.Array) -> jax.Array:
        expected = self.num_context_features * self.target_hidden_size
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                "DFLASH target hidden feature mismatch: "
                f"expected last dim {expected}, got {target_hidden.shape[-1]}."
            )
        hidden_states, _ = self.fc(target_hidden)
        return self.hidden_norm(hidden_states)

    def materialize_kv(
        self,
        target_hidden: jax.Array,
        positions: jax.Array,
    ) -> list[tuple[jax.Array, jax.Array]]:
        ctx = self.project_target_hidden(target_hidden)
        return [layer.self_attn.kv_proj_only(positions, ctx) for layer in self.model.layers]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: Any = None,
    ):
        hidden_states, layers_kv_fused = self.model(
            forward_batch,
            memory_pools.token_to_kv_pool,
        )
        output = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)
        return output, {"token_to_kv_pool": layers_kv_fused}, [], None

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("DFlash draft weights loaded successfully.")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        mappings: dict[str, WeightMapping] = {
            "fc.weight": WeightMapping(
                target_path="fc.weight",
                sharding=(None, None),
                transpose=True,
            ),
            "hidden_norm.weight": WeightMapping(
                target_path="hidden_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        for layer_idx in range(int(self.config.num_hidden_layers)):
            prefix = f"layers.{layer_idx}"
            target = f"model.layers.{layer_idx}"
            mappings.update(
                {
                    f"{prefix}.input_layernorm.weight": WeightMapping(
                        target_path=f"{target}.input_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{target}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                        target_path=f"{target}.self_attn.q_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                        target_path=f"{target}.self_attn.k_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{target}.mlp.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{target}.mlp.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{target}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )
            if getattr(self.config, "attention_bias", False):
                for proj, kv_padding in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
                    mappings[f"{prefix}.self_attn.{proj}.bias"] = WeightMapping(
                        target_path=f"{target}.self_attn.{proj}.bias",
                        sharding=(None,),
                        transpose=False,
                        kv_head_padding=kv_padding,
                    )
                mappings[f"{prefix}.self_attn.o_proj.bias"] = WeightMapping(
                    target_path=f"{target}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                )

        return mappings


EntryClass = DFlashDraftModel
