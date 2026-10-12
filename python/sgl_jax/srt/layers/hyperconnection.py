"""Hyper Connections, and Qwen4Exp's gated-residual variant.

Hyper Connections (https://arxiv.org/abs/2409.19606) widen the inter-layer
residual path from one stream to ``hc_count`` parallel streams, so hidden
states between blocks are ``[..., HC*HS]`` (HC outer, HS inner -- the
checkpoint-native layout) rather than ``[..., HS]``.

Every block is wrapped by a pair::

    mixed, residuals = hc.mix(hyper_input)   # [..., HC*HS] -> [..., HS]
    block_out = block(mixed)
    hyper_input = hc.combine(block_out, residuals)

The block is an ordinary ``HS``-wide module. ``combine`` needs the same
normalized hyper input that ``mix`` computed, so ``mix`` returns it in
``residuals`` rather than ``combine`` recomputing it.

:class:`HyperConnectionBase` is the ungated scheme: read by averaging the
streams, write by adding the block output to each.

:class:`GatedResidual` is Qwen4Exp's variant: a low-rank sigmoid gate on the
read, one learned scalar per stream on the write. That per-stream scalar is the
general form's cross-stream mixing matrix fixed to the identity.

Structure follows upstream SGLang's ``python/sglang/srt/layers/hyperconnection.py``:
``use_mix`` / ``use_combine`` select which projections exist, and the registry
at the bottom maps a config string to a variant.

``sgl_jax.srt.kernels.mhc`` is DeepSeek-V4's variant of the same paper: it keeps
the cross-stream mixing matrix, Sinkhorn-normalized, and gates at full rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.utils.profiling_utils import named_scope


@dataclass(frozen=True)
class HyperConnectionConfig:
    """Mirrors upstream's struct, plus ``mesh``: ``LinearBase`` needs one, and
    the base class builds no weights, so it is optional.
    """

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: jnp.dtype = jnp.bfloat16
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False
    mesh: jax.sharding.Mesh | None = None

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


class GroupedGemmaRMSNorm(nnx.Module):
    """RMSNorm over groups of ``group_size``, with Gemma's ``x * (1 + w)`` offset.

    Qwen4Exp's ``hc_norm`` uses the grouped form: the weight spans all HC
    streams (``[HC*HS]``) while each stream is normalized over its own HS
    elements.
    """

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        group_size: int | None = None,
        kernel_axes: tuple[str | None, ...] | None = None,
        params_dtype: jnp.dtype = jnp.float32,
    ):
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.group_size = group_size
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.zeros, kernel_axes)(
                jax.random.PRNGKey(0), (hidden_size,), params_dtype
            )
        )

    @named_scope
    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x32 = x.astype(jnp.float32)
        if self.group_size is None:
            variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
            normed = x32 * jax.lax.rsqrt(variance + self.epsilon)
        else:
            grouped = x32.reshape(
                *x32.shape[:-1], x32.shape[-1] // self.group_size, self.group_size
            )
            variance = jnp.mean(jnp.square(grouped), axis=-1, keepdims=True)
            normed = (grouped * jax.lax.rsqrt(variance + self.epsilon)).reshape(x32.shape)
        return (normed * (1.0 + jnp.asarray(self.weight, jnp.float32))).astype(orig_dtype)


class HyperConnectionBase(nnx.Module):
    """The ungated scheme: average to read, add to write. Carries no weights.

    Subclasses override :meth:`mix` and :meth:`combine`; ``use_mix`` and
    ``use_combine`` select which projections they build, so a read-only
    instance (the model's final mixer) carries no injection weight.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        scope_name: str = "hyper_connection",
    ):
        if config.hc_count < 1:
            raise ValueError(f"hc_count must be positive, got {config.hc_count}")
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hyper_hidden_size = config.hyper_hidden_size
        self.params_dtype = config.params_dtype
        self.use_mix = use_mix
        self.use_combine = use_combine
        self.name = scope_name

    def _unflatten(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-1], self.hc_count, self.hidden_size)

    def _check_hyper(self, hyper_input: jax.Array) -> None:
        if hyper_input.shape[-1] != self.hyper_hidden_size:
            raise ValueError(
                f"expected last dim {self.hyper_hidden_size}, got {hyper_input.shape[-1]}"
            )

    @named_scope
    def mix(self, hyper_input: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Collapse the streams to one ``HS``-wide input for the block."""
        self._check_hyper(hyper_input)
        mixed = jnp.mean(self._unflatten(hyper_input), axis=-2)
        return mixed, hyper_input

    @named_scope
    def combine(self, block_output: jax.Array, hyper_input: jax.Array) -> jax.Array:
        """Add the block output into every stream."""
        self._check_hyper(hyper_input)
        if block_output.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected block output last dim {self.hidden_size}, "
                f"got {block_output.shape[-1]}"
            )
        out = self._unflatten(hyper_input) + block_output[..., None, :]
        return out.reshape(hyper_input.shape)


class GatedResidual(HyperConnectionBase):
    """Qwen4Exp's variant: learned gates on both the read and the write.

    Weights, as the checkpoint ships them (``[out, in]``, so the mappings need
    ``transpose=True``)::

        hc_norm.weight                [HC*HS]        per-branch, one affine per stream
        input_mix_weight_down.weight  [lowrank, HC*HS]
        input_mix_weight_up.weight    [HC*HS, lowrank]
        block_inject_weight.weight    [HC, HC*HS]    absent when use_combine=False
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        scope_name: str = "hyper_connection",
    ):
        super().__init__(config, use_mix, use_combine, scope_name)
        if config.hc_lowrank < 1:
            raise ValueError(f"hc_lowrank must be positive, got {config.hc_lowrank}")
        if config.mesh is None:
            raise ValueError("GatedResidual builds LinearBase projections and needs a mesh")

        # Qwen3.8-Flash-Next ships the per-branch form: hc_norm.weight is
        # [HC*HS]. The shared form is one [HS] weight broadcast to all streams.
        norm_dim = self.hyper_hidden_size if config.hc_per_branch_norm else self.hidden_size
        group_size = self.hidden_size if config.hc_per_branch_norm else None
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_dim,
            epsilon=config.rms_norm_eps,
            group_size=group_size,
            params_dtype=config.params_dtype,
        )

        # Replicated, not tensor-parallel: `up` emits [..., HC*HS], so a
        # row-parallel split would add an all-reduce at each of the 2*num_layers
        # call sites to save a few hundred MB of weights.
        if use_mix:
            self.input_mix_weight_down = LinearBase(
                input_size=self.hyper_hidden_size,
                output_size=config.hc_lowrank,
                mesh=config.mesh,
                use_bias=False,
                params_dtype=config.params_dtype,
                kernel_axes=(None, None),
                scope_name="input_mix_weight_down",
            )
            self.input_mix_weight_up = LinearBase(
                input_size=config.hc_lowrank,
                output_size=self.hyper_hidden_size,
                mesh=config.mesh,
                use_bias=False,
                params_dtype=config.params_dtype,
                kernel_axes=(None, None),
                scope_name="input_mix_weight_up",
            )
        if use_combine:
            self.block_inject_weight = LinearBase(
                input_size=self.hyper_hidden_size,
                output_size=self.hc_count,
                mesh=config.mesh,
                use_bias=False,
                params_dtype=config.params_dtype,
                kernel_axes=(None, None),
                scope_name="block_inject_weight",
            )

    def _normalize(self, hyper_input: jax.Array) -> jax.Array:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        # Unflatten first so the [HS] weight broadcasts across streams and each
        # stream is normalized over its own HS elements.
        return self.hc_norm(self._unflatten(hyper_input)).reshape(hyper_input.shape)

    @named_scope
    def mix(self, hyper_input: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        self._check_hyper(hyper_input)
        if not self.use_mix:
            raise RuntimeError("mix() called on an instance built with use_mix=False")
        normed = self._normalize(hyper_input)

        # The /hc_count scaling and the silu-then-sigmoid order match what the
        # checkpoint was trained with; changing either is silently wrong.
        gate, _ = self.input_mix_weight_down(normed)
        gate = jax.nn.silu(gate / self.hc_count)
        gate, _ = self.input_mix_weight_up(gate)
        gate = self._unflatten(jax.nn.sigmoid(gate))

        # The mean is over the stream axis, hence the reshape.
        mixed = jnp.mean(gate * self._unflatten(normed), axis=-2)
        return mixed, (hyper_input, normed)

    @named_scope
    def combine(self, block_output: jax.Array, residuals: tuple[jax.Array, jax.Array]) -> jax.Array:
        """Add the block output into every stream, scaled per stream.

        The per-stream scales are computed from the pre-block state, not from
        ``block_output``.
        """
        if not self.use_combine:
            raise RuntimeError("combine() called on an instance built with use_combine=False")
        hyper_input, normed = residuals
        self._check_hyper(hyper_input)
        if block_output.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected block output last dim {self.hidden_size}, "
                f"got {block_output.shape[-1]}"
            )

        # 2*sigmoid, not sigmoid: the per-stream scales span (0, 2), not (0, 1).
        # The second /hc_count is here.
        inject, _ = self.block_inject_weight(normed)
        inject = 2.0 * jax.nn.sigmoid(inject / self.hc_count)

        out = self._unflatten(hyper_input) + block_output[..., None, :] * inject[..., :, None]
        return out.reshape(hyper_input.shape)


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
