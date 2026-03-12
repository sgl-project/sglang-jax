import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import modeling_flax_utils

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.models.flux.embeddings import CombinedTimestepLabelEmbeddings


def _no_shard(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))


class _FP32LayerNorm(nnx.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        self.norm = nnx.LayerNorm(
            num_features=num_features,
            epsilon=eps,
            use_bias=False,
            use_scale=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x).astype(x.dtype)


def _build_zero_norm(dim: int, eps: float, norm_type: str):
    if norm_type == "layer_norm":
        return nnx.LayerNorm(
            num_features=dim,
            epsilon=eps,
            use_bias=False,
            use_scale=False,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )
    if norm_type == "fp32_layer_norm":
        return _FP32LayerNorm(dim, eps=eps)
    raise ValueError(
        f"Unsupported `norm_type` ({norm_type}) provided. "
        "Supported ones are: 'layer_norm', 'fp32_layer_norm'."
    )


def _build_continuous_norm(
    embedding_dim: int,
    eps: float,
    elementwise_affine: bool,
    norm_type: str,
):
    if norm_type == "layer_norm":
        return nnx.LayerNorm(
            num_features=embedding_dim,
            epsilon=eps,
            use_bias=elementwise_affine,
            use_scale=elementwise_affine,
            use_fast_variance=False,
            rngs=nnx.Rngs(0),
        )
    if norm_type == "rms_norm":
        return RMSNorm(
            num_features=embedding_dim,
            epsilon=eps,
            use_scale=elementwise_affine,
            use_fast_variance=False,
        )
    raise ValueError(f"unknown norm_type {norm_type}")


class FluxAdaLayerNormZero(nnx.Module):
    def __init__(
        self,
        dim: int,
        mesh: Mesh,
        num_embeddings: int | None = None,
        class_dropout_prob: float = 0.1,
        norm_type: str = "layer_norm",
        eps: float = 1e-6,
        bias: bool = True,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.emb = (
            CombinedTimestepLabelEmbeddings(
                num_classes=num_embeddings,
                embedding_dim=dim,
                mesh=mesh,
                class_dropout_prob=class_dropout_prob,
                params_dtype=params_dtype,
            )
            if num_embeddings is not None
            else None
        )
        self.norm_type = norm_type
        self.norm = _build_zero_norm(dim, eps, norm_type)
        self.act = modeling_flax_utils.ACT2FN["silu"]
        self.linear = LinearBase(
            input_size=dim,
            output_size=6 * dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps

    def __call__(
        self,
        hidden_states: jax.Array,
        timestep: jax.Array | None = None,
        class_labels: jax.Array | None = None,
        hidden_dtype: jnp.dtype | None = None,
        emb: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        if self.emb is not None:
            if timestep is None or class_labels is None:
                raise ValueError(
                    "`timestep` and `class_labels` must be provided when `num_embeddings` is set."
                )
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        if emb is None:
            raise ValueError("`emb` must be provided for FluxAdaLayerNormZero.")
        hidden_states = self.norm(hidden_states)
        emb, _ = self.linear(self.act(emb))
        emb = _no_shard(emb, self.mesh)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            emb, 6, axis=-1
        )
        hidden_states = hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class FluxAdaLayerNormZeroSingle(nnx.Module):
    def __init__(
        self,
        dim: int,
        mesh: Mesh,
        norm_type: str = "layer_norm",
        eps: float = 1e-6,
        bias: bool = True,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.norm_type = norm_type
        self.norm = _build_zero_norm(dim, eps, norm_type)
        self.act = modeling_flax_utils.ACT2FN["silu"]
        self.linear = LinearBase(
            input_size=dim,
            output_size=3 * dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps

    def __call__(self, hidden_states: jax.Array, emb: jax.Array) -> tuple[jax.Array, jax.Array]:
        hidden_states = self.norm(hidden_states)
        emb, _ = self.linear(self.act(emb))
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
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        bias: bool = True,
        norm_type: str = "layer_norm",
        params_dtype: jnp.dtype | None = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.norm_type = norm_type
        self.norm = _build_continuous_norm(
            embedding_dim=embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
            norm_type=norm_type,
        )
        self.act = modeling_flax_utils.ACT2FN["silu"]
        self.linear = LinearBase(
            input_size=conditioning_embedding_dim,
            output_size=2 * embedding_dim,
            use_bias=bias,
            mesh=mesh,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
        )
        self.eps = eps
        self.weight = getattr(self.norm, "scale", None)
        self.bias = getattr(self.norm, "bias", None)

    def __call__(self, x: jax.Array, conditioning_embedding: jax.Array) -> jax.Array:
        emb, _ = self.linear(self.act(conditioning_embedding).astype(x.dtype))
        emb = _no_shard(emb, self.mesh)
        scale, shift = jnp.split(emb, 2, axis=-1)
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]
