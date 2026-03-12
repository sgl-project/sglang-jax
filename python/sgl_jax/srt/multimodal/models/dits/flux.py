import logging
import math
from contextlib import suppress
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.dits.flux_model_config import FluxModelConfig
from sgl_jax.srt.multimodal.layers.attention.layer import USPAttention
from sgl_jax.srt.multimodal.models.dits.flux_weights_mapping import to_mappings
from sgl_jax.srt.multimodal.models.layers.adalayernorm import (
    FluxAdaLayerNormContinuous,
    FluxAdaLayerNormZero,
    FluxAdaLayerNormZeroSingle,
)
from sgl_jax.srt.multimodal.models.layers.visual_embedding import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    _apply_flux_rotary_emb,
    _get_1d_rotary_pos_embed,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader

logger = logging.getLogger(__name__)
_SUPPORTED_ATTENTION_IMPLS = ("usp", "sdpa")


def _resolve_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    return rngs or nnx.Rngs(0)


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
    with suppress(AttributeError):
        jax.set_mesh(mesh)
    return mesh


def _no_shard(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))


def _sdpa_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attn_mask: jax.Array | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **_: Any,
) -> jax.Array:
    del dropout_p
    q = query.astype(jnp.float32)
    k = key.astype(jnp.float32)
    v = value.astype(jnp.float32)

    mask = None
    bias = None
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        elif attn_mask.ndim != 4:
            raise ValueError(
                "attn_mask must have rank 2, 3, or 4 for Flux attention. "
                f"Got shape {attn_mask.shape}."
            )

        if attn_mask.dtype == jnp.bool_:
            mask = attn_mask
        else:
            bias = attn_mask.astype(jnp.float32)

    output = jax.nn.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        scale=scale,
        is_causal=is_causal,
        implementation="xla",
    )
    return output.astype(query.dtype)


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


class FluxFeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        mesh: Mesh,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        if activation_fn == "gelu":
            self.act = modeling_flax_utils.ACT2FN["gelu"]
        elif activation_fn == "gelu-approximate":
            self.act = modeling_flax_utils.ACT2FN["gelu_pytorch_tanh"]
        else:
            raise ValueError(f"Unsupported activation_fn {activation_fn!r} for FluxFeedForward.")

        self.net = nnx.List(
            [
                LinearBase(
                    input_size=dim,
                    output_size=inner_dim,
                    use_bias=bias,
                    mesh=mesh,
                    params_dtype=params_dtype,
                    kernel_axes=(None, "tensor"),
                ),
                nnx.Dropout(dropout, rngs=_rngs),
                LinearBase(
                    input_size=inner_dim,
                    output_size=dim_out,
                    use_bias=bias,
                    mesh=mesh,
                    params_dtype=params_dtype,
                    kernel_axes=("tensor", None),
                ),
            ]
        )
        if final_dropout:
            self.net.append(nnx.Dropout(dropout, rngs=_rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.net[0](x)
        x = self.act(x)
        x = self.net[1](x, deterministic=True)
        x, _ = self.net[2](x)
        if len(self.net) == 4:
            x = self.net[3](x, deterministic=True)
        return x


class FluxAttention(nnx.Module):
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
        attention_impl: str = "usp",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        mesh: Mesh | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """
        NOTE: Align with the implementation of HF.
        attention_impl: support usp and sdpa. usp dont support attention_mask yet.
        """
        mesh = _resolve_mesh(mesh)
        _rngs = _resolve_rngs(rngs)
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
                    )
                ]
            )
            if dropout != 0.0:
                self.to_out.append(nnx.Dropout(dropout, rngs=_rngs))

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
                num_heads=self.heads,
                head_size=dim_head,
                causal=False,
                mesh=mesh,
            )
        else:
            self.attn = None

    def __call__(
        self,
        hidden_states: jax.Array,
        encoder_hidden_states: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        image_rotary_emb: tuple[jax.Array, jax.Array] | None = None,
        req=None,
        **kwargs,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        if kwargs:
            logger.warning(
                "Ignoring unsupported joint_attention_kwargs in FluxAttention: %s",
                list(kwargs),
            )

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            self, hidden_states, encoder_hidden_states
        )

        query = query.reshape(query.shape[0], query.shape[1], self.heads, self.head_dim)
        key = key.reshape(key.shape[0], key.shape[1], self.heads, self.head_dim)
        value = value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query = encoder_query.reshape(
                encoder_query.shape[0], encoder_query.shape[1], self.heads, self.head_dim
            )
            encoder_key = encoder_key.reshape(
                encoder_key.shape[0], encoder_key.shape[1], self.heads, self.head_dim
            )
            encoder_value = encoder_value.reshape(
                encoder_value.shape[0], encoder_value.shape[1], self.heads, self.head_dim
            )

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = jnp.concatenate([encoder_query, query], axis=1)
            key = jnp.concatenate([encoder_key, key], axis=1)
            value = jnp.concatenate([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            query = _apply_flux_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = _apply_flux_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if self.attention_impl == "sdpa":
            hidden_states = _sdpa_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=False,
            )
        else:
            if attention_mask is not None:
                logger.warning(
                    "attention_mask is not supported by USPAttention yet; falling back to sdpa attention."
                )
                hidden_states = _sdpa_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    is_causal=False,
                )
            else:
                hidden_states = self.attn(query, key, value, req)

        hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        hidden_states = hidden_states.astype(query.dtype)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :context_len, :]
            hidden_states = hidden_states[:, context_len:, :]
            if not self.pre_only:
                hidden_states, _ = self.to_out[0](hidden_states)
                if len(self.to_out) == 2:
                    hidden_states = self.to_out[1](hidden_states, deterministic=True)
            encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        if not self.pre_only:
            hidden_states, _ = self.to_out[0](hidden_states)
            if len(self.to_out) == 2:
                hidden_states = self.to_out[1](hidden_states, deterministic=True)
        return hidden_states


class FluxSingleTransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mesh: Mesh,
        mlp_ratio: float = 4.0,
        attention_impl: str = "usp",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = FluxAdaLayerNormZeroSingle(
            dim, mesh=mesh, params_dtype=params_dtype, rngs=_rngs
        )
        self.proj_mlp = LinearBase(
            input_size=dim,
            output_size=self.mlp_hidden_dim,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.proj_out = LinearBase(
            input_size=dim + self.mlp_hidden_dim,
            output_size=dim,
            use_bias=True,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=("tensor", None),
        )
        self.act_mlp = modeling_flax_utils.ACT2FN["gelu_pytorch_tanh"]
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            attention_impl=attention_impl,
            params_dtype=params_dtype,
            mesh=mesh,
            rngs=_rngs,
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
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            req=req,
            **joint_attention_kwargs,
        )

        hidden_states = jnp.concatenate([attn_output, mlp_hidden_states], axis=-1)
        gate = gate[:, None, :]
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states

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
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.norm1 = FluxAdaLayerNormZero(
            dim, mesh=mesh, eps=eps, params_dtype=params_dtype, rngs=_rngs
        )
        self.norm1_context = FluxAdaLayerNormZero(
            dim, mesh=mesh, eps=eps, params_dtype=params_dtype, rngs=_rngs
        )
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            attention_impl=attention_impl,
            params_dtype=params_dtype,
            mesh=mesh,
            rngs=_rngs,
        )
        self.eps = eps
        self.norm2 = nnx.LayerNorm(
            num_features=dim,
            epsilon=eps,
            use_bias=False,
            use_scale=False,
            use_fast_variance=False,
            rngs=_rngs,
        )
        self.norm2_context = nnx.LayerNorm(
            num_features=dim,
            epsilon=eps,
            use_bias=False,
            use_scale=False,
            use_fast_variance=False,
            rngs=_rngs,
        )
        self.ff = FluxFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
        )
        self.ff_context = FluxFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            mesh=mesh,
            params_dtype=params_dtype,
            rngs=_rngs,
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

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        )
        hidden_states = hidden_states + gate_mlp[:, None, :] * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None, :] * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
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
    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": {"split_dim": 1, "expected_dims": 3, "split_output": False},
            "encoder_hidden_states": {"split_dim": 1, "expected_dims": 3, "split_output": False},
            "img_ids": {"split_dim": 0, "expected_dims": 2, "split_output": False},
            "txt_ids": {"split_dim": 0, "expected_dims": 2, "split_output": False},
        },
        "proj_out": {"gather_dim": 1, "expected_dims": 3},
    }

    def __init__(
        self,
        config: FluxModelConfig,
        *,
        dtype: jnp.dtype | None = None,
        mesh: Mesh | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        self.config = config
        self.model_config = config
        self.dtype = dtype or config.dtype
        self.params_dtype = config.weights_dtype
        self.mesh = _resolve_mesh(mesh)
        self.attention_impl = config.attention_impl
        self.out_channels = config.out_channels or config.in_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.gradient_checkpointing = False

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
            params_dtype=self.params_dtype,
            rngs=_rngs,
        )

        self.context_embedder = LinearBase(
            input_size=config.joint_attention_dim,
            output_size=self.inner_dim,
            use_bias=True,
            mesh=self.mesh,
            params_dtype=self.params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.x_embedder = LinearBase(
            input_size=config.in_channels,
            output_size=self.inner_dim,
            use_bias=True,
            mesh=self.mesh,
            params_dtype=self.params_dtype,
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
                    params_dtype=self.params_dtype,
                    rngs=_rngs,
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
                    params_dtype=self.params_dtype,
                    rngs=_rngs,
                )
                for _ in range(config.num_single_layers)
            ]
        )

        self.norm_out = FluxAdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            mesh=self.mesh,
            eps=config.epsilon,
            params_dtype=self.params_dtype,
            rngs=_rngs,
        )
        self.proj_out = LinearBase(
            input_size=self.inner_dim,
            output_size=config.patch_size * config.patch_size * self.out_channels,
            use_bias=True,
            mesh=self.mesh,
            params_dtype=self.params_dtype,
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
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block // interval_control]
                    )

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
                hidden_states = (
                    hidden_states + controlnet_single_block_samples[index_block // interval_control]
                )

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
