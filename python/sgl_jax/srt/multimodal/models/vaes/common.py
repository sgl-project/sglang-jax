from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclasses.dataclass
class DiagonalGaussianDistribution:
    parameters: jax.Array
    deterministic: bool = False
    channel_axis: int | None = None

    def __post_init__(self):
        axis = self.channel_axis
        if axis is None:
            axis = 1 if self.parameters.ndim == 4 else self.parameters.ndim - 1
        self.mean, self.logvar = jnp.split(self.parameters, 2, axis=axis)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = jnp.zeros_like(self.mean)
            self.std = jnp.zeros_like(self.mean)

    def tree_flatten(self):
        children = (self.parameters,)
        aux_data = {
            "deterministic": self.deterministic,
            "channel_axis": self.channel_axis,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.parameters = children[0]
        obj.deterministic = aux_data["deterministic"]
        obj.channel_axis = aux_data["channel_axis"]
        obj.__post_init__()
        return obj

    def sample(self, random: int | jax.Array) -> jax.Array:
        if isinstance(random, int):
            key = jax.random.PRNGKey(random)
        else:
            random = jnp.asarray(random)
            key = jax.random.PRNGKey(int(random)) if random.ndim == 0 else random
        sample = jax.random.normal(key, self.mean.shape, dtype=self.parameters.dtype)
        return self.mean + self.std * sample

    def kl(self, other: DiagonalGaussianDistribution | None = None) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0], dtype=self.parameters.dtype)
        reduce_axes = tuple(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * jnp.sum(
                jnp.square(self.mean) + self.var - 1.0 - self.logvar,
                axis=reduce_axes,
            )
        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=reduce_axes,
        )

    def nll(self, sample: jax.Array, dims: tuple[int, ...] | None = None) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0], dtype=self.parameters.dtype)
        if dims is None:
            dims = tuple(range(1, sample.ndim))
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var,
            axis=dims,
        )

    def mode(self) -> jax.Array:
        return self.mean


class Upsample2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        return self.conv(hidden_states)


class Downsample2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = jnp.pad(hidden_states, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(hidden_states)


class ResnetBlock2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        groups: int = 32,
        use_nin_shortcut: bool | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_nin_shortcut = (
            in_channels != self.out_channels if use_nin_shortcut is None else use_nin_shortcut
        )

        _rngs = rngs or nnx.Rngs(0)
        self.norm1 = nnx.GroupNorm(
            num_features=in_channels,
            num_groups=groups,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_features=self.out_channels,
            num_groups=groups,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.dropout_layer = nnx.Dropout(dropout, rngs=_rngs)
        self.conv2 = nnx.Conv(
            in_features=self.out_channels,
            out_features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )
        if self.use_nin_shortcut:
            self.conv_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                param_dtype=dtype,
                rngs=_rngs,
            )

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = jax.nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = jax.nn.swish(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = self.conv2(hidden_states)

        if hasattr(self, "conv_shortcut"):
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        channels: int,
        num_head_channels: int | None = None,
        num_groups: int = 32,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        _rngs = rngs or nnx.Rngs(0)

        self.group_norm = nnx.GroupNorm(
            num_features=channels,
            num_groups=num_groups,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.query = nnx.Linear(
            in_features=channels,
            out_features=channels,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.key = nnx.Linear(
            in_features=channels,
            out_features=channels,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.value = nnx.Linear(
            in_features=channels,
            out_features=channels,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.proj_attn = nnx.Linear(
            in_features=channels,
            out_features=channels,
            param_dtype=dtype,
            rngs=_rngs,
        )

    def transpose_for_scores(self, projection: jax.Array) -> jax.Array:
        projection = projection.reshape(projection.shape[:-1] + (self.num_heads, -1))
        return jnp.transpose(projection, (0, 2, 1, 3))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.reshape((batch, height * width, channels))

        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = hidden_states.reshape((batch, height * width, channels))
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        return hidden_states + residual


class DownEncoderBlock2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.resnets = nnx.List(
            [
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    dtype=dtype,
                    rngs=_rngs,
                )
                for i in range(num_layers)
            ]
        )
        if add_downsample:
            self.downsamplers = nnx.List([Downsample2D(out_channels, dtype=dtype, rngs=_rngs)])

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)
        if hasattr(self, "downsamplers"):
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class UpDecoderBlock2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        self.resnets = nnx.List(
            [
                ResnetBlock2D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    dtype=dtype,
                    rngs=_rngs,
                )
                for i in range(num_layers)
            ]
        )
        if add_upsample:
            self.upsamplers = nnx.List([Upsample2D(out_channels, dtype=dtype, rngs=_rngs)])

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)
        if hasattr(self, "upsamplers"):
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UNetMidBlock2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_groups: int = 32,
        num_attention_heads: int | None = 1,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = rngs or nnx.Rngs(0)
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.resnets = nnx.List(
            [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    dtype=dtype,
                    rngs=_rngs,
                )
            ]
        )
        self.attentions = nnx.List([])
        for _ in range(num_layers):
            self.attentions.append(
                AttentionBlock(
                    channels=in_channels,
                    num_head_channels=num_attention_heads,
                    num_groups=resnet_groups,
                    dtype=dtype,
                    rngs=_rngs,
                )
            )
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    dtype=dtype,
                    rngs=_rngs,
                )
            )

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attention(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)
        return hidden_states


class Encoder(nnx.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        del down_block_types, act_fn
        _rngs = rngs or nnx.Rngs(0)
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )
        output_channel = block_out_channels[0]
        self.down_blocks = nnx.List([])
        for i, out_ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    dtype=dtype,
                    rngs=_rngs,
                )
            )
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            num_attention_heads=None,
            dtype=dtype,
            rngs=_rngs,
        )
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_norm_out = nnx.GroupNorm(
            num_features=block_out_channels[-1],
            num_groups=norm_num_groups,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.conv_out = nnx.Conv(
            in_features=block_out_channels[-1],
            out_features=conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, sample: jax.Array, deterministic: bool = True) -> jax.Array:
        sample = self.conv_in(sample)
        for block in self.down_blocks:
            sample = block(sample, deterministic=deterministic)
        sample = self.mid_block(sample, deterministic=deterministic)
        sample = self.conv_norm_out(sample)
        sample = jax.nn.swish(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder(nnx.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        del up_block_types, act_fn
        _rngs = rngs or nnx.Rngs(0)
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            num_attention_heads=None,
            dtype=dtype,
            rngs=_rngs,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        self.up_blocks = nnx.List([])
        for i, out_ch in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(block_out_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    resnet_groups=norm_num_groups,
                    add_upsample=not is_final_block,
                    dtype=dtype,
                    rngs=_rngs,
                )
            )
        self.conv_norm_out = nnx.GroupNorm(
            num_features=block_out_channels[0],
            num_groups=norm_num_groups,
            epsilon=1e-6,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.conv_out = nnx.Conv(
            in_features=block_out_channels[0],
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            param_dtype=dtype,
            rngs=_rngs,
        )

    def __call__(self, sample: jax.Array, deterministic: bool = True) -> jax.Array:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, deterministic=deterministic)
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)
        sample = self.conv_norm_out(sample)
        sample = jax.nn.swish(sample)
        sample = self.conv_out(sample)
        return sample
