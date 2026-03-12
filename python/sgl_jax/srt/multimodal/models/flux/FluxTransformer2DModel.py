import inspect
import logging
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.dits.flux_model_config import FluxModelConfig
from sgl_jax.srt.multimodal.layers.attention.layer import USPAttention
from sgl_jax.srt.multimodal.models.flux.flux_weights_mapping import to_mappings
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)
_SUPPORTED_ATTENTION_IMPLS = ("usp", "naive")


def _resolve_mesh(mesh: Mesh | None) -> Mesh:
    if mesh is None:
        devices = np.array(jax.devices()).reshape((1, -1))
        try:
            mesh = Mesh(
                devices,
                ("data", "tensor"),
                axis_types=(
                    jax.sharding.AxisType.Explicit,
                    jax.sharding.AxisType.Explicit,
                ),
            )
        except TypeError:
            mesh = Mesh(devices, ("data", "tensor"))
    try:
        jax.set_mesh(mesh)
    except AttributeError:
        pass
    return mesh


def _no_shard(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))


def _layer_norm(
    x: jax.Array,
    weight: jax.Array | None = None,
    bias: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    origin_dtype = x.dtype
    y = x.astype(jnp.float32)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.var(y, axis=-1, keepdims=True)
    y = (y - mean) * jax.lax.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.astype(jnp.float32)
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    return y.astype(origin_dtype)


def _gelu_tanh(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x, approximate=True)


def _rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)


def _apply_flux_rotary_emb(
    x: jax.Array,
    image_rotary_emb: tuple[jax.Array, jax.Array] | None,
    sequence_dim: int = 1,
) -> jax.Array:
    if image_rotary_emb is None:
        return x
    cos, sin = image_rotary_emb
    if sequence_dim != 1:
        raise NotImplementedError("Flux rotary embedding currently expects sequence_dim=1")
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    return x * cos + _rotate_half(x) * sin


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: jax.Array,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = pos.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)
    return cos, sin


def _naive_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float | None = None,
    causal: bool = False,
) -> jax.Array:
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k) * scale
    if causal:
        seq_len = query.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        attn_weights = jnp.where(mask[None, None, :, :], attn_weights, -jnp.inf)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v)
    return jnp.transpose(output, (0, 2, 1, 3))


def _get_projections(
    attn: "FluxAttention",
    hidden_states: jax.Array,
    encoder_hidden_states: jax.Array | None = None,
):
    query, _ = attn.to_q(hidden_states)
    key, _ = attn.to_k(hidden_states)
    value, _ = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query, _ = attn.add_q_proj(encoder_hidden_states)
        encoder_key, _ = attn.add_k_proj(encoder_hidden_states)
        encoder_value, _ = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(
    attn: "FluxAttention",
    hidden_states: jax.Array,
    encoder_hidden_states: jax.Array | None = None,
):
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class Timesteps(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * jnp.arange(half_dim, dtype=jnp.float32)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = jnp.exp(exponent)
        emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
        emb = self.scale * emb
        if self.flip_sin_to_cos:
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        if self.num_channels % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


class FluxTimestepEmbedding(nnx.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, mesh: Mesh):
        self.linear_1 = LinearBase(
            input_size=in_channels,
            output_size=time_embed_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.linear_2 = LinearBase(
            input_size=time_embed_dim,
            output_size=time_embed_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

    def __call__(self, sample: jax.Array) -> jax.Array:
        sample, _ = self.linear_1(sample)
        sample = jax.nn.silu(sample)
        sample, _ = self.linear_2(sample)
        return sample


class FluxTextProjection(nnx.Module):
    def __init__(self, in_features: int, hidden_size: int, mesh: Mesh, act_fn: str = "silu"):
        self.linear_1 = LinearBase(
            input_size=in_features,
            output_size=hidden_size,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.linear_2 = LinearBase(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.act_fn = act_fn

    def __call__(self, caption: jax.Array) -> jax.Array:
        hidden_states, _ = self.linear_1(caption)
        if self.act_fn == "silu":
            hidden_states = jax.nn.silu(hidden_states)
        else:
            hidden_states = _gelu_tanh(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepTextProjEmbeddings(nnx.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int, mesh: Mesh):
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = FluxTimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, mesh=mesh
        )
        self.text_embedder = FluxTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            mesh=mesh,
            act_fn="silu",
        )

    def __call__(self, timestep: jax.Array, pooled_projection: jax.Array) -> jax.Array:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        pooled_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_emb


class CombinedTimestepGuidanceTextProjEmbeddings(nnx.Module):
    def __init__(self, embedding_dim: int, pooled_projection_dim: int, mesh: Mesh):
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = FluxTimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, mesh=mesh
        )
        self.guidance_embedder = FluxTimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, mesh=mesh
        )
        self.text_embedder = FluxTextProjection(
            in_features=pooled_projection_dim,
            hidden_size=embedding_dim,
            mesh=mesh,
            act_fn="silu",
        )

    def __call__(
        self,
        timestep: jax.Array,
        guidance: jax.Array,
        pooled_projection: jax.Array,
    ) -> jax.Array:
        timesteps_proj = self.time_proj(timestep)
        guidance_proj = self.time_proj(guidance)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        guidance_emb = self.guidance_embedder(guidance_proj)
        pooled_emb = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_emb


class FluxAdaLayerNormZero(nnx.Module):
    def __init__(self, dim: int, mesh: Mesh, eps: float = 1e-6):
        self.mesh = mesh
        self.linear = LinearBase(
            input_size=dim,
            output_size=6 * dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps

    def __call__(
        self,
        hidden_states: jax.Array,
        emb: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        hidden_states = _layer_norm(hidden_states, eps=self.eps)
        emb, _ = self.linear(jax.nn.silu(emb))
        emb = _no_shard(emb, self.mesh)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            emb, 6, axis=-1
        )
        hidden_states = hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class FluxAdaLayerNormZeroSingle(nnx.Module):
    def __init__(self, dim: int, mesh: Mesh, eps: float = 1e-6):
        self.mesh = mesh
        self.linear = LinearBase(
            input_size=dim,
            output_size=3 * dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps

    def __call__(self, hidden_states: jax.Array, emb: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden_states = _layer_norm(hidden_states, eps=self.eps)
        emb, _ = self.linear(jax.nn.silu(emb))
        emb = _no_shard(emb, self.mesh)
        shift_msa, scale_msa, gate_msa = jnp.split(emb, 3, axis=-1)
        hidden_states = hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa


class FluxAdaLayerNormContinuous(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        mesh: Mesh,
        eps: float = 1e-6,
    ):
        self.mesh = mesh
        self.weight = nnx.Param(jnp.ones((embedding_dim,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((embedding_dim,), dtype=jnp.float32))
        self.linear = LinearBase(
            input_size=conditioning_embedding_dim,
            output_size=2 * embedding_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps

    def __call__(self, x: jax.Array, conditioning_embedding: jax.Array) -> jax.Array:
        emb, _ = self.linear(jax.nn.silu(conditioning_embedding).astype(x.dtype))
        emb = _no_shard(emb, self.mesh)
        scale, shift = jnp.split(emb, 2, axis=-1)
        x = _layer_norm(x, self.weight.value, self.bias.value, eps=self.eps)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class FluxFeedForward(nnx.Module):
    def __init__(self, dim: int, dim_out: int, mesh: Mesh):
        self.fc1 = LinearBase(
            input_size=dim,
            output_size=dim_out,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.fc2 = LinearBase(
            input_size=dim_out,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.fc1(x)
        x = _gelu_tanh(x)
        x, _ = self.fc2(x)
        return x


class FluxDropout(nnx.Module):
    def __init__(self, rate: float):
        self.dropout = nnx.Dropout(rate)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        return self.dropout(x, deterministic=deterministic)


class FluxAttnProcessor:
    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        image_rotary_emb: tuple[jax.Array, jax.Array] | None = None,
        req=None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if attention_mask is not None:
            logger.warning("attention_mask is currently ignored in FluxAttnProcessor")

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.reshape(query.shape[0], query.shape[1], attn.heads, attn.head_dim)
        key = key.reshape(key.shape[0], key.shape[1], attn.heads, attn.head_dim)
        value = value.reshape(value.shape[0], value.shape[1], attn.heads, attn.head_dim)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.reshape(
                encoder_query.shape[0], encoder_query.shape[1], attn.heads, attn.head_dim
            )
            encoder_key = encoder_key.reshape(
                encoder_key.shape[0], encoder_key.shape[1], attn.heads, attn.head_dim
            )
            encoder_value = encoder_value.reshape(
                encoder_value.shape[0], encoder_value.shape[1], attn.heads, attn.head_dim
            )

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = jnp.concatenate([encoder_query, query], axis=1)
            key = jnp.concatenate([encoder_key, key], axis=1)
            value = jnp.concatenate([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            query = _apply_flux_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _apply_flux_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = attn.run_attention(query, key, value, req=req)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        hidden_states = hidden_states.astype(query.dtype)

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :context_len, :]
            hidden_states = hidden_states[:, context_len:, :]
            if not attn.pre_only:
                hidden_states, _ = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states, deterministic=True)
            encoder_hidden_states, _ = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        if not attn.pre_only:
            hidden_states, _ = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states, deterministic=True)
        return hidden_states


class FluxAttention(nnx.Module):
    _default_processor_cls = FluxAttnProcessor

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor: FluxAttnProcessor | None = None,
        attention_impl: str = "usp",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        mesh: Mesh | None = None,
    ):
        mesh = _resolve_mesh(mesh)
        if attention_impl not in _SUPPORTED_ATTENTION_IMPLS:
            raise ValueError(
                f"Unsupported attention_impl {attention_impl!r}. "
                f"Expected one of {_SUPPORTED_ATTENTION_IMPLS}."
            )

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.attention_impl = attention_impl
        self.mesh = mesh
        self.params_dtype = params_dtype

        self.norm_q = RMSNorm(dim_head, epsilon=eps, use_scale=elementwise_affine)
        self.norm_k = RMSNorm(dim_head, epsilon=eps, use_scale=elementwise_affine)

        self.to_q = LinearBase(
            input_size=query_dim,
            output_size=self.inner_dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.to_k = LinearBase(
            input_size=query_dim,
            output_size=self.inner_dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.to_v = LinearBase(
            input_size=query_dim,
            output_size=self.inner_dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        if not self.pre_only:
            self.to_out = nnx.List(
                [
                    LinearBase(
                        input_size=self.inner_dim,
                        output_size=self.out_dim,
                        use_bias=out_bias,
                        mesh=mesh,
                        params_dtype=params_dtype,
                        kernel_axes=("tensor", None),
                    ),
                    FluxDropout(dropout),
                ]
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, epsilon=eps)
            self.norm_added_k = RMSNorm(dim_head, epsilon=eps)
            self.add_q_proj = LinearBase(
                input_size=added_kv_proj_dim,
                output_size=self.inner_dim,
                use_bias=added_proj_bias,
                mesh=mesh,
                params_dtype=params_dtype,
                kernel_axes=(None, "tensor"),
            )
            self.add_k_proj = LinearBase(
                input_size=added_kv_proj_dim,
                output_size=self.inner_dim,
                use_bias=added_proj_bias,
                mesh=mesh,
                params_dtype=params_dtype,
                kernel_axes=(None, "tensor"),
            )
            self.add_v_proj = LinearBase(
                input_size=added_kv_proj_dim,
                output_size=self.inner_dim,
                use_bias=added_proj_bias,
                mesh=mesh,
                params_dtype=params_dtype,
                kernel_axes=(None, "tensor"),
            )
            self.to_add_out = LinearBase(
                input_size=self.inner_dim,
                output_size=query_dim,
                use_bias=out_bias,
                mesh=mesh,
                params_dtype=params_dtype,
                kernel_axes=("tensor", None),
            )

        if self.attention_impl == "usp":
            self.attn = USPAttention(
                num_heads=heads,
                head_size=dim_head,
                causal=False,
                mesh=mesh,
            )
        else:
            self.attn = None
        self.processor = processor or self._default_processor_cls()

    def run_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        req=None,
    ) -> jax.Array:
        if self.attention_impl == "naive":
            del req
            return _naive_attention(query, key, value, causal=False)
        return self.attn(query, key, value, req)

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        image_rotary_emb: tuple[jax.Array, jax.Array] | None = None,
        req=None,
        **kwargs,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k in kwargs if k not in attn_parameters]
        if unused_kwargs:
            logger.warning("Ignoring unsupported joint_attention_kwargs in FluxAttention: %s", unused_kwargs)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            req=req,
            **filtered_kwargs,
        )


class FluxSingleTransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mesh: Mesh,
        mlp_ratio: float = 4.0,
        attention_impl: str = "usp",
    ):
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = FluxAdaLayerNormZeroSingle(dim, mesh=mesh)
        self.proj_mlp = LinearBase(
            input_size=dim,
            output_size=self.mlp_hidden_dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )
        self.proj_out = LinearBase(
            input_size=dim + self.mlp_hidden_dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            processor=FluxAttnProcessor(),
            attention_impl=attention_impl,
            mesh=mesh,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        image_rotary_emb: tuple[jax.Array, jax.Array] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        req=None,
    ) -> tuple[jax.Array, jax.Array]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states, _ = self.proj_mlp(norm_hidden_states)
        mlp_hidden_states = _gelu_tanh(mlp_hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            req=req,
            **joint_attention_kwargs,
        )

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=-1)
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = residual + gate[:, None, :] * hidden_states

        if hidden_states.dtype == jnp.float16:
            hidden_states = jnp.clip(hidden_states, -65504, 65504)

        encoder_hidden_states = hidden_states[:, :text_seq_len, :]
        hidden_states = hidden_states[:, text_seq_len:, :]
        return encoder_hidden_states, hidden_states


class FluxTransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mesh: Mesh,
        eps: float = 1e-6,
        attention_impl: str = "usp",
    ):
        self.norm1 = FluxAdaLayerNormZero(dim, mesh=mesh, eps=eps)
        self.norm1_context = FluxAdaLayerNormZero(dim, mesh=mesh, eps=eps)
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            processor=FluxAttnProcessor(),
            attention_impl=attention_impl,
            mesh=mesh,
        )
        self.eps = eps
        self.ff = FluxFeedForward(dim=dim, dim_out=4 * dim, mesh=mesh)
        self.ff_context = FluxFeedForward(dim=dim, dim_out=4 * dim, mesh=mesh)

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array,
        temb: jax.Array,
        image_rotary_emb: tuple[jax.Array, jax.Array] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        req=None,
    ) -> tuple[jax.Array, jax.Array]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            req=req,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        hidden_states = hidden_states + gate_msa[:, None, :] * attn_output

        norm_hidden_states = _layer_norm(hidden_states, eps=self.eps)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        )
        hidden_states = hidden_states + gate_mlp[:, None, :] * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None, :] * context_attn_output
        norm_encoder_hidden_states = _layer_norm(encoder_hidden_states, eps=self.eps)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp[:, None, :] * self.ff_context(
            norm_encoder_hidden_states
        )
        if encoder_hidden_states.dtype == jnp.float16:
            encoder_hidden_states = jnp.clip(encoder_hidden_states, -65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nnx.Module):
    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        self.theta = float(theta)
        self.axes_dim = tuple(axes_dim)

    def __call__(self, ids: jax.Array) -> tuple[jax.Array, jax.Array]:
        pos = ids.astype(jnp.float32)
        n_axes = pos.shape[-1]
        cos_out = []
        sin_out = []
        for axis_idx in range(n_axes):
            cos, sin = _get_1d_rotary_pos_embed(
                self.axes_dim[axis_idx],
                pos[:, axis_idx],
                theta=self.theta,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        return jnp.concatenate(cos_out, axis=-1), jnp.concatenate(sin_out, axis=-1)


class FluxTransformer2DModel(nnx.Module):
    def __init__(
        self,
        config: FluxModelConfig,
        *,
        dtype: jnp.dtype | None = None,
        mesh: Mesh | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        del rngs
        self.model_config = config
        self.dtype = dtype or config.dtype
        self.mesh = _resolve_mesh(mesh)
        self.attention_impl = config.attention_impl
        self.out_channels = config.out_channels or config.in_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=config.axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if config.guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=config.pooled_projection_dim,
            mesh=self.mesh,
        )

        self.context_embedder = LinearBase(
            input_size=config.joint_attention_dim,
            output_size=self.inner_dim,
            use_bias=True,
            mesh=self.mesh,
            kernel_axes=(None, "tensor"),
        )
        self.x_embedder = LinearBase(
            input_size=config.in_channels,
            output_size=self.inner_dim,
            use_bias=True,
            mesh=self.mesh,
            kernel_axes=(None, "tensor"),
        )

        self.transformer_blocks = nnx.List(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    mesh=self.mesh,
                    eps=config.epsilon,
                    attention_impl=self.attention_impl,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.single_transformer_blocks = nnx.List(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    mesh=self.mesh,
                    attention_impl=self.attention_impl,
                )
                for _ in range(config.num_single_layers)
            ]
        )

        self.norm_out = FluxAdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            mesh=self.mesh,
            eps=config.epsilon,
        )
        self.proj_out = LinearBase(
            input_size=self.inner_dim,
            output_size=config.patch_size * config.patch_size * self.out_channels,
            use_bias=True,
            mesh=self.mesh,
            kernel_axes=("tensor", None),
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        pooled_projections: jax.Array | None = None,
        timestep: jax.Array | None = None,
        img_ids: jax.Array | None = None,
        txt_ids: jax.Array | None = None,
        guidance: jax.Array | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        req=None,
    ) -> jax.Array | tuple[jax.Array]:
        if encoder_hidden_states is None or pooled_projections is None or timestep is None:
            raise ValueError(
                "encoder_hidden_states, pooled_projections, and timestep are required for Flux"
            )

        hidden_states, _ = self.x_embedder(hidden_states)

        timestep = timestep.astype(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.astype(hidden_states.dtype) * 1000

        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        temb = _no_shard(temb, self.mesh)
        encoder_hidden_states, _ = self.context_embedder(encoder_hidden_states)

        image_rotary_emb = None
        if txt_ids is not None and img_ids is not None:
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]
            ids = jnp.concatenate((txt_ids, img_ids), axis=0)
            image_rotary_emb = self.pos_embed(ids)

        joint_attention_kwargs = joint_attention_kwargs or {}
        if "ip_adapter_image_embeds" in joint_attention_kwargs:
            logger.warning("ip_adapter_image_embeds is not implemented in sglang-jax Flux yet")

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                req=req,
            )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(math.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[
                        index_block % len(controlnet_block_samples)
                    ]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[
                        index_block // interval_control
                    ]

        for index_block, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                req=req,
            )

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(math.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[
                    index_block // interval_control
                ]

        hidden_states = self.norm_out(hidden_states, temb)
        output, _ = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return output

    def load_weights(self, model_path: str | FluxModelConfig | None = None) -> None:
        resolved_model_path = None
        if model_path is not None and not isinstance(model_path, str):
            resolved_model_path = getattr(model_path, "model_path", None)
        else:
            resolved_model_path = model_path

        if resolved_model_path is None:
            resolved_model_path = self.model_config.model_path

        if resolved_model_path is None:
            raise ValueError("model_path must be provided either in config or as an argument")

        self.model_config.model_path = resolved_model_path
        loader = WeightLoader(
            model=self,
            model_config=self.model_config,
            mesh=self.mesh,
            dtype=self.model_config.weights_dtype,
        )
        loader.load_weights_from_safetensors(
            to_mappings(has_guidance_embeds=self.model_config.guidance_embeds)
        )


EntryClass = FluxTransformer2DModel
