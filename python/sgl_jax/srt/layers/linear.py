import logging
import os
from collections.abc import Sequence
from functools import partial

import jax
import numpy as np
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul.blockwise_utils import expand_block_scale
from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

logger = logging.getLogger(__name__)


def _shard_map_output_partition_dim(
    sharding: jax.sharding.Sharding, axis_name: str | None
) -> int | None:
    if axis_name is None:
        return None
    for dim, axis in enumerate(sharding.spec):
        if axis == axis_name:
            return dim
        if isinstance(axis, tuple) and axis_name in axis:
            return dim
    return None


class LinearBase(nnx.Module):
    """Base linear layer.

    Args:
        input_size: Input dimension of the linear layer.
        output_size: Output dimension of the linear layer.
        mesh: Device mesh for sharding.
        use_bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters and accumulation preference.
        kernel_axes: Partition spec for the weight tensor.
        scope_name: Name used for profiling scope.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        mesh: jax.sharding.Mesh,
        use_bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        kernel_axes: Sequence[str | None] | None = None,
        scope_name: str = "linear_base",
    ):
        """Initialize parameters and quantization method."""
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes
        self.mesh = mesh
        self.name = scope_name

        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (input_size, output_size),
                dtype=params_dtype,
                out_sharding=P(*kernel_axes),
            ),
        )
        if use_bias:
            self.bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (output_size,),
                    dtype=params_dtype,
                    out_sharding=P(kernel_axes[-1]),
                ),
            )
        else:
            self.bias = None

    @named_scope
    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass. If ``out_sharding`` is None, falls back to the
        standard TP layout derived from ``kernel_axes`` (col-parallel →
        ``P("data", "tensor")``, row-parallel → ``P("data", None)``).
        SP-aware callers must pass an explicit ``out_sharding``.
        """
        target = out_sharding or NamedSharding(
            self.mesh,
            P("data", *([None] * (x.ndim - 2)), self.kernel_axes[-1]),
        )
        out = lax.dot_general(
            x,
            self.weight.value,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=self.params_dtype,
            out_sharding=target,
        )
        if self.skip_bias_add:
            return out, self.bias
        if self.bias is not None:
            out = out + self.bias.value
        return out, None


class MergedColumnParallelLinear(LinearBase):
    """Column-parallel linear with multiple logical outputs merged into one weight.

    Equivalent to ``N`` independent column-parallel ``LinearBase``s with the
    same ``input_size`` but different ``output_size``, fused into one larger
    GEMM. A single large matmul on TPU's MXU is consistently faster than ``N``
    smaller ones — fewer kernel launches, better pipelining of weight reads,
    and a single MXU pass amortizes the input-side broadcast.

    Sharding contract (mirrors sglang / vLLM's ``MergedColumnParallelLinear``):
    each device's local weight columns hold
    ``[comp_0_my_heads | comp_1_my_heads | ...]`` block-concat. Splitting the
    merged output into per-component pieces must therefore happen on
    per-device data (typically inside :func:`jax.shard_map`) using **per-shard**
    sizes — the global merged tensor is stripe-interleaved across devices,
    not a true ``[comp_0 | comp_1 | ...]`` block-concat.

    Each entry of ``output_sizes`` must be divisible by the mesh's ``"tensor"``
    axis size so the per-shard block-concat boundary aligns with the TP cut.
    Without this, GQA-style projections (where components have different
    head counts) would put a shard boundary mid-component — exactly the
    failure mode this layer exists to avoid.

    Weight loading is the caller's responsibility — there's no built-in
    loader yet because the simple host-side scatter (collect HF tensors,
    stripe them per-rank on host, single ``device_put``) costs N host
    buffers and a full-tensor staging copy. A device-side scatter
    (writing each HF tensor into a sharded merged param via
    ``jax.lax.dynamic_update_slice`` under the right sharding context)
    is the right shape for production but needs more design — left as a
    follow-up.

    Args:
        input_size: Input dimension.
        output_sizes: Per-component output dimensions. Must each be
            divisible by the mesh's ``"tensor"`` axis size.
        mesh: Device mesh (must expose a ``"tensor"`` axis for sharding;
            falls back to TP=1 if absent or ``mesh is None``).
        use_bias / skip_bias_add / params_dtype: forwarded to ``LinearBase``.
        scope_name: profiling scope.
    """

    @staticmethod
    def _mesh_tp_size(mesh: jax.sharding.Mesh | None) -> int:
        """TP size = mesh size on the ``"tensor"`` axis (1 if absent)."""
        if mesh is None:
            return 1
        shape = getattr(mesh, "shape", None)
        if shape is None or "tensor" not in shape:
            return 1
        return int(shape["tensor"])

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        mesh: jax.sharding.Mesh,
        use_bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        scope_name: str = "merged_column_parallel_linear",
    ):
        self.output_sizes = list(output_sizes)
        tp_size = self._mesh_tp_size(mesh)
        for i, sz in enumerate(self.output_sizes):
            if sz % tp_size != 0:
                raise ValueError(
                    f"MergedColumnParallelLinear: output_sizes[{i}]={sz} must be "
                    f"divisible by TP={tp_size} for clean per-shard block-concat layout."
                )
        super().__init__(
            input_size=input_size,
            output_size=sum(self.output_sizes),
            mesh=mesh,
            use_bias=use_bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            kernel_axes=(None, "tensor"),
            scope_name=scope_name,
        )


# The merged layout helpers take the component sizes rather than the layer:
# quantization replaces a MergedColumnParallelLinear with a QuantizedLinear.


def split_merged_output(
    out: jax.Array, output_sizes: Sequence[int], mesh: jax.sharding.Mesh
) -> list[jax.Array]:
    """Split the output of a :class:`MergedColumnParallelLinear` into its
    components, each sharded over ``"tensor"`` like the output of a separate
    column-parallel layer.

    Each device splits its own block, so no data moves between devices.
    """
    tp = MergedColumnParallelLinear._mesh_tp_size(mesh)
    lead = out.shape[:-1]
    spec = ("data", *([None] * (out.ndim - 2)))
    out = out.reshape(*lead, tp, -1, out_sharding=NamedSharding(mesh, P(*spec, "tensor", None)))
    parts = jnp.split(out, np.cumsum([size // tp for size in output_sizes])[:-1], axis=-1)
    sharding = NamedSharding(mesh, P(*spec, "tensor"))
    return [part.reshape(*lead, -1, out_sharding=sharding) for part in parts]


def stripe_merged_weight(weight: np.ndarray, output_sizes: Sequence[int], tp: int) -> np.ndarray:
    """Rearrange a host ``[input_size, sum(output_sizes)]`` weight from
    ``[comp_0 | comp_1 | ...]`` into the per-device block-concat layout of
    :class:`MergedColumnParallelLinear`."""
    splits = np.cumsum(output_sizes)[:-1]
    parts = [np.split(part, tp, axis=1) for part in np.split(weight, splits, axis=1)]
    return np.concatenate([part[rank] for rank in range(tp) for part in parts], axis=1)


class QuantizedLinear(nnx.Module):
    """Quantized linear layer using native quantized matmul.

    This layer stores pre-quantized weights and scales, and uses the native
    quantized matmul kernel for the forward pass. Weights are quantized once
    at initialization/conversion time, and activations are quantized at runtime.

    Block-wise scales are pre-expanded from ``[out_blocks, in_blocks]`` to the
    kernel-ready ``[in_blocks, 1, n_out]`` layout at init time so that no
    ``jnp.repeat`` runs on the inference hot path.

    Args:
        weight_q: Quantized weight tensor with shape ``[output_size, input_size]``.
        weight_scale: Weight quantization scale. Per-channel uses
            ``[output_size]``; block-wise uses ``[in_blocks, 1, output_size]``
            (pre-expanded kernel-ready layout).
        bias: Optional bias tensor with shape ``[output_size]``.
        activation_dtype: Dtype for activation quantization. ``None`` disables
            activation quantization.
        mesh: Device mesh for sharding.
        kernel_axes: Partition spec axes for the weight tensor.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Original parameter dtype used for output casting.
        compute_dtype: Optional dtype override for the quantized kernel.
        weight_block_size: Optional block size ``(block_n, block_k)`` for
            block-wise weight quantization.
        scope_name: Name used for profiling scope.
    """

    def __init__(
        self,
        weight_q: jax.Array,
        weight_scale: jax.Array,
        bias: jax.Array | None,
        activation_dtype: jnp.dtype | None,
        mesh: jax.sharding.Mesh,
        kernel_axes: Sequence[str | None] | None = None,
        skip_bias_add: bool = False,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        compute_dtype: jnp.dtype | None = None,
        weight_block_size: tuple[int, int] | None = None,
        allow_narrow_n_blockwise: bool = False,
        scope_name: str = "quantized_linear",
    ):
        """Initialize the quantized linear layer with pre-quantized weights."""
        # Auto-expand 2D block-quant scale to 3D kernel-ready layout.
        # This handles direct construction (not via from_linear) where the
        # caller passes a compact 2D scale [out_blocks, in_blocks] or
        # [n_out, in_blocks] along with weight_block_size.
        if (
            weight_block_size is not None
            and not isinstance(weight_scale, jax.ShapeDtypeStruct)
            and weight_scale.ndim == 2
        ):
            n_out = weight_q.shape[0] if not isinstance(weight_q, jax.ShapeDtypeStruct) else None
            if n_out is not None:
                weight_scale = expand_block_scale(weight_scale, n_out, int(weight_block_size[0]))

        if (
            mesh is not None
            and kernel_axes is not None
            and not isinstance(weight_scale, jax.ShapeDtypeStruct)
        ):
            scale_spec = (
                P(kernel_axes[0], None, kernel_axes[1])
                if weight_scale.ndim == 3
                else P(kernel_axes[1])
            )
            weight_scale = jax.device_put(weight_scale, NamedSharding(mesh, scale_spec))

        self.weight_q = nnx.Param(weight_q)
        self.weight_scale = nnx.Param(weight_scale)
        self.bias = nnx.Param(bias) if bias is not None else None
        self.activation_dtype = activation_dtype
        self.mesh = mesh
        self.kernel_axes = kernel_axes
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.compute_dtype = compute_dtype
        self.weight_block_size = weight_block_size
        self.allow_narrow_n_blockwise = allow_narrow_n_blockwise
        self.name = scope_name
        # Logical output width when the weight rows were pre-padded to a block
        # multiple at load time (pad_out_rows); None = weight is not padded.
        self.n_out_valid: int | None = None

    @classmethod
    def from_linear(
        cls,
        linear: LinearBase,
        weight_dtype: jnp.dtype,
        activation_dtype: jnp.dtype | None = None,
        is_static_input: bool = False,
        weight_block_size: Sequence[int] | None = None,
        allow_narrow_n_blockwise: bool = False,
    ) -> "QuantizedLinear":
        """Convert a LinearBase layer to a QuantizedLinear layer.

        Uses per-channel weight quantization by default, and block-wise
        weight quantization when ``weight_block_size`` is provided.

        Args:
            linear: The LinearBase layer to convert.
            weight_dtype: Target dtype for weight quantization.
            activation_dtype: Target dtype for activation quantization
                (``None`` = no activation quantization).
            is_static_input: If true, expect a static checkpoint with
                pre-quantized weights.
            weight_block_size: Optional ``(block_n, block_k)`` for
                block-wise weight quantization.

        Returns:
            A new QuantizedLinear layer with quantized weights.
        """
        effective_weight_block_size = (
            tuple(weight_block_size) if weight_block_size is not None else None
        )

        kernel_axes = linear.kernel_axes or (None, None)
        if is_static_input:
            # Static checkpoint already stores pre-quantized weights and scales.
            weight = linear.weight.value

            if isinstance(weight, jax.ShapeDtypeStruct):
                in_features, out_features = map(int, weight.shape)

                if (
                    effective_weight_block_size is not None
                    and len(effective_weight_block_size) == 2
                ):
                    block_n, block_k = int(effective_weight_block_size[0]), int(
                        effective_weight_block_size[1]
                    )
                    in_blocks = (in_features + block_k - 1) // block_k
                    # Row-parallel block-wise layers shard the reduce axis, so
                    # the in_blocks dimension of the [in_blocks, 1, n_out]
                    # scale must also tile across TP. When it doesn't (e.g.
                    # GLM-5.1 dense/shared down_proj: 96/16 blocks vs TP=64),
                    # fall back to a replicated reduce axis for this layer
                    # only. The original LinearBase keeps row-parallel.
                    input_axis = kernel_axes[0]
                    tp = linear.mesh.shape.get(input_axis, 1) if input_axis else 1
                    if input_axis is not None and in_blocks % tp != 0:
                        logger.warning(
                            "QuantizedLinear %s [in=%d, out=%d]: in_blocks=%d not "
                            "divisible by TP=%d on axis %r; replicating reduce axis",
                            linear.name,
                            in_features,
                            out_features,
                            in_blocks,
                            tp,
                            input_axis,
                        )
                        kernel_axes = (None, kernel_axes[1])
                    # Pre-expanded kernel-ready layout: [in_blocks, 1, n_out].
                    scale_sharding = NamedSharding(
                        linear.mesh, P(kernel_axes[0], None, kernel_axes[1])
                    )
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(in_blocks, 1, out_features),
                        dtype=jnp.float32,
                        sharding=scale_sharding,
                    )
                else:
                    scale_sharding = NamedSharding(linear.mesh, P(kernel_axes[1]))
                    weight_scale = jax.ShapeDtypeStruct(
                        shape=(out_features,), dtype=jnp.float32, sharding=scale_sharding
                    )
                wq_sharding = NamedSharding(linear.mesh, P(kernel_axes[1], kernel_axes[0]))
                weight_q = jax.ShapeDtypeStruct(
                    shape=(out_features, in_features),
                    dtype=weight_dtype,
                    sharding=wq_sharding,
                )
                bias = linear.bias.value if linear.bias is not None else None
            else:
                if weight.dtype != weight_dtype:
                    raise ValueError(
                        "QuantizedLinear.from_linear(..., is_static_input=True) requires "
                        "pre-quantized concrete weights or abstract shapes. "
                        f"Got weight.dtype={weight.dtype}, expected {weight_dtype}."
                    )
                weight_q = weight.T
                if (
                    effective_weight_block_size is not None
                    and len(effective_weight_block_size) == 2
                ):
                    block_n, block_k = int(effective_weight_block_size[0]), int(
                        effective_weight_block_size[1]
                    )
                    out_blocks = (weight_q.shape[0] + block_n - 1) // block_n
                    in_blocks = (weight_q.shape[1] + block_k - 1) // block_k
                    scale_2d = jnp.ones((out_blocks, in_blocks), dtype=jnp.float32)
                    weight_scale = expand_block_scale(scale_2d, weight_q.shape[0], block_n)
                else:
                    weight_scale = jnp.ones((weight_q.shape[0],), dtype=jnp.float32)
                bias = linear.bias.value if linear.bias is not None else None
        else:
            # LinearBase weight shape: [input_size, output_size]
            # xla_quantized_matmul expects w_q: [output_size, input_size]
            # So we need to transpose the weight before quantizing.
            weight = linear.weight.value
            weight_t = weight.T  # [output_size, input_size]

            if effective_weight_block_size is not None and len(effective_weight_block_size) == 2:
                # Block-wise quantization over [output_size, input_size].
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype,
                    tensor=weight_t,
                    axis=(0, 1),
                    block_size=tuple(effective_weight_block_size),
                    pad_tensor=True,
                )
                # Expand scale from [out_blocks, in_blocks] to kernel-ready
                # [in_blocks, 1, n_out] at init time.
                weight_scale = expand_block_scale(
                    weight_scale,
                    weight_q.shape[0],
                    int(effective_weight_block_size[0]),
                )
            else:
                # Per-channel quantization along output dimension.
                # After transpose, output_size is axis 0 and input_size is axis 1.
                # We want per-output-channel, so reduce along axis 1.
                weight_q, weight_scale = quantize_tensor(
                    dtype=weight_dtype, tensor=weight_t, axis=1
                )

            # Get bias if it exists.
            bias = linear.bias.value if linear.bias is not None else None

        return cls(
            weight_q=weight_q,
            weight_scale=weight_scale,
            bias=bias,
            activation_dtype=activation_dtype,
            mesh=linear.mesh,
            kernel_axes=kernel_axes,
            skip_bias_add=linear.skip_bias_add,
            params_dtype=linear.params_dtype,
            weight_block_size=effective_weight_block_size,
            allow_narrow_n_blockwise=allow_narrow_n_blockwise,
            scope_name=f"quantized_{linear.name}",
        )

    @named_scope
    def pad_out_rows(self, multiple: int = 256) -> bool:
        """Pad the quantized weight / block scale along the output dim once.

        The block-wise matmul wrapper pads ``w_q`` / ``w_scale`` to a multiple
        of the tuned out block on *every* call when the (per-shard) ``n_out`` is
        not aligned (e.g. GLM-5.2 kv_a_proj 576 -> 768 replicated, or the
        shared-expert gate/up 2048 / tp16 = 128 -> 256 per shard: two weight
        copies per layer per step). Doing it here at load time removes that
        per-step traffic; the call path keeps using the logical (per-shard)
        width for the tuned-block lookup and slices the output back.

        Replicated N is padded directly; a tensor-sharded N is padded inside a
        shard_map so every shard grows by the same rows. Returns True when the
        weight was padded.
        """
        if self.n_out_valid is not None:
            return False
        scale = self.weight_scale.value
        if scale is None or getattr(scale, "ndim", 0) != 3:
            return False
        weight = self.weight_q.value
        out_axis = self.kernel_axes[1]
        shards = int(self.mesh.shape[out_axis]) if out_axis is not None else 1
        n_out = int(weight.shape[0])
        if n_out % shards != 0:
            return False
        n_local = n_out // shards
        padded_local = ((n_local + multiple - 1) // multiple) * multiple
        if padded_local == n_local:
            return False
        pad = padded_local - n_local
        if out_axis is None:
            self.weight_q.value = jnp.pad(weight, ((0, pad), (0, 0)))
            self.weight_scale.value = jnp.pad(scale, ((0, 0), (0, 0), (0, pad)))
        else:
            w_spec = P(out_axis, None)
            s_spec = P(None, None, out_axis)
            weight = jax.sharding.reshard(weight, NamedSharding(self.mesh, w_spec))
            scale = jax.sharding.reshard(scale, NamedSharding(self.mesh, s_spec))
            self.weight_q.value = shard_map(
                lambda w: jnp.pad(w, ((0, pad), (0, 0))),
                mesh=self.mesh,
                in_specs=w_spec,
                out_specs=w_spec,
                check_vma=False,
            )(weight)
            self.weight_scale.value = shard_map(
                lambda sc: jnp.pad(sc, ((0, 0), (0, 0), (0, pad))),
                mesh=self.mesh,
                in_specs=s_spec,
                out_specs=s_spec,
                check_vma=False,
            )(scale)
        self.n_out_valid = n_local
        return True

    def __call__(
        self,
        x: jax.Array,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """Forward pass using quantized matmul. If ``out_sharding`` is None,
        falls back to standard TP layout derived from ``kernel_axes``.
        SP-aware callers must pass an explicit ``out_sharding``.

        Args:
            x: Input tensor [..., input_size]
            out_sharding: Optional output sharding override.

        Returns:
            Tuple of (output, bias) where output is [..., output_size]
            and bias is returned if skip_bias_add is True
        """
        # Determine if we should quantize activations.
        quantize_activation = self.activation_dtype is not None

        # Handle batched inputs by reshaping to 2D.
        x_2d = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x

        scale_val = self.weight_scale.value
        if (
            scale_val.ndim == 2
            and scale_val.shape[1] == 1
            and scale_val.shape[0] == self.weight_q.value.shape[0]
        ):
            scale_val = jnp.squeeze(scale_val, axis=1)

        # Shard specs for shard_map.
        # kernel_axes = (input_axis, output_axis):
        #   row-parallel  (e.g., o_proj): ("tensor", None)
        #   col-parallel  (e.g., q_proj): (None, "tensor")
        input_axis, output_axis = self.kernel_axes[0], self.kernel_axes[1]
        if scale_val.ndim == 3:  # noqa: SIM108
            # Pre-expanded block scale: [in_blocks, 1, n_out]
            w_scale_spec = P(input_axis, None, output_axis)
        else:
            # Per-channel scale: [n_out]
            w_scale_spec = P(output_axis)
        # jax[tpu] 0.10 runs the mesh in Explicit-axis mode ("sharding in types"):
        # shard_map requires each input's committed sharding to match in_specs
        # exactly, and with_sharding_constraint is only an assert. FP8 block-quant
        # scales can arrive REPLICATED from the weight loader (the kernel-ready 3D
        # expansion drops the sharding), which trips col-parallel projections such
        # as q_b_proj. Explicitly reshard the scale to its expected spec — a no-op
        # when it is already correctly sharded.
        scale_val = jax.sharding.reshard(scale_val, NamedSharding(self.mesh, w_scale_spec))
        in_specs = (P("data", input_axis), P(output_axis, input_axis), w_scale_spec)
        weight_val = self.weight_q.value
        if input_axis is None:
            # Reduce axis replicated for this layer (block scale does not tile
            # across TP, see from_linear) while the loader may still have placed
            # the weight row-parallel: bring it to the replicated in_spec. Only
            # narrow-K block-quant layers (e.g. a 2048-wide shared expert at
            # tp32) take this path; tp16 layers are unaffected.
            w_sh = jax.typeof(weight_val).sharding
            if (
                isinstance(w_sh, NamedSharding)
                and w_sh.spec
                and len(w_sh.spec) > 1
                and w_sh.spec[1] is not None
            ):
                weight_val = jax.sharding.reshard(
                    weight_val, NamedSharding(self.mesh, P(output_axis, None))
                )

        target = out_sharding or NamedSharding(self.mesh, P("data", output_axis))
        output_partition_dim = _shard_map_output_partition_dim(target, input_axis)

        output = shard_map(
            partial(
                xla_quantized_matmul_local,
                quantize_activation=quantize_activation,
                reduce_axis=input_axis,
                compute_dtype=self.compute_dtype,
                weight_block_size=self.weight_block_size,
                activation_quant_dtype=self.activation_dtype,
                allow_narrow_n_blockwise=self.allow_narrow_n_blockwise,
                output_scatter_dimension=output_partition_dim,
                n_out_valid=self.n_out_valid,
            ),
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=target.spec,
            check_vma=False,
        )(x_2d, weight_val, scale_val)

        # Reshape back to original batch dimensions.
        if x.ndim > 2:
            output = output.reshape(x.shape[:-1] + (output.shape[-1],))

        # Handle bias.
        if self.skip_bias_add:
            return output, self.bias
        if self.bias is not None:
            output = output + self.bias.value
        return output, None


def fuse_quantized_linears(
    mods: Sequence["QuantizedLinear"],
    *,
    scope_name: str,
    pad_n_to: int = 512,
    member_paths: Sequence[str] | None = None,
) -> tuple["QuantizedLinear", list[int]] | None:
    """Concatenate replicated ``QuantizedLinear`` modules that consume the same
    input into ONE ``QuantizedLinear`` along the output dim.

    GLM-5.2 decode issues q_a_proj / kv_a_proj_with_mqa / indexer.wk as three
    separate fp8 blockwise matmuls over the same ``hidden_states`` every layer;
    at M<=64 each is a ~12-20 us fixed-cost launch far above its HBM time, so
    3 -> 1 launches per layer is the win (weights are byte-identical, blockwise
    scales ``[in_blocks, 1, n_out]`` simply concatenate along ``n_out``).

    The concatenated output width is zero-padded up to a multiple of
    ``pad_n_to`` at build time: the blockwise kernel wrapper otherwise pads the
    weight to a multiple of its out block on EVERY call (09-21 trace: a 25 MB
    ``pad(f8[2752,6144]) -> [4096,6144]`` per layer per step, +2.5 ms/step).

    Returns ``(fused, splits)`` where ``splits`` are the per-module output
    widths (slice the fused output with ``jnp.split``; the padding tail is the
    last, unused segment), or ``None`` when the
    modules are not fusable (different input width / dtype / block size /
    non-replicated layout / 1-D scales / bias) so callers keep the unfused path.

    ``member_paths`` (module paths of ``mods`` relative to the fused module's parent, e.g.
    ``"q_a_proj"`` / ``"indexer/wk"``) are kept on the fused module as
    ``fused_member_paths`` so later per-module passes keyed on the module path (the
    per-channel requantization skip regex) still see the original layers.
    """
    if not mods or not all(isinstance(m, QuantizedLinear) for m in mods):
        return None
    m0 = mods[0]
    w_qs = [m.weight_q.value for m in mods]
    w_scales = [m.weight_scale.value for m in mods]
    same = all(
        w.shape[1] == w_qs[0].shape[1]
        and w.dtype == w_qs[0].dtype
        and ws.ndim == 3
        and ws.shape[:2] == w_scales[0].shape[:2]
        and m.bias is None
        and tuple(m.kernel_axes or (None, None)) == (None, None)
        and m.activation_dtype == m0.activation_dtype
        and m.weight_block_size == m0.weight_block_size
        and m.compute_dtype == m0.compute_dtype
        for m, w, ws in zip(mods, w_qs, w_scales)
    )
    if not same:
        return None
    w_q = jnp.concatenate(w_qs, axis=0)
    w_scale = jnp.concatenate(w_scales, axis=-1)
    n_total = int(w_q.shape[0])
    n_pad = (-n_total) % pad_n_to if pad_n_to else 0
    if n_pad:
        w_q = jnp.concatenate([w_q, jnp.zeros((n_pad, w_q.shape[1]), w_q.dtype)], axis=0)
        w_scale = jnp.concatenate(
            [w_scale, jnp.ones(w_scale.shape[:2] + (n_pad,), w_scale.dtype)], axis=-1
        )
    fused = QuantizedLinear(
        weight_q=w_q,
        weight_scale=w_scale,
        bias=None,
        activation_dtype=m0.activation_dtype,
        mesh=m0.mesh,
        kernel_axes=(None, None),
        skip_bias_add=m0.skip_bias_add,
        params_dtype=m0.params_dtype,
        compute_dtype=m0.compute_dtype,
        weight_block_size=m0.weight_block_size,
        allow_narrow_n_blockwise=m0.allow_narrow_n_blockwise,
        scope_name=scope_name,
    )
    fused.fused_member_paths = tuple(member_paths or ())
    return fused, [int(w.shape[0]) for w in w_qs]


def prepad_replicated_quantized_linears(module: nnx.Module, multiple: int = 256) -> int:
    """Pad every block-quant QuantizedLinear under ``module`` once (see
    QuantizedLinear.pad_out_rows). Disabled with SGLANG_JAX_QMM_PREPAD_N=0.
    Returns the number of layers padded."""
    if os.environ.get("SGLANG_JAX_QMM_PREPAD_N", "1") == "0":
        return 0
    count = 0
    padded: dict[str, int] = {}
    skipped: dict[str, int] = {}
    seen: dict[str, int] = {}
    for path, sub in module.iter_modules():
        if isinstance(sub, (QuantizedLinear, LinearBase)):
            k = f"{str(path[-1]) if path else '?'}:{type(sub).__name__}"
            seen[k] = seen.get(k, 0) + 1
        if not isinstance(sub, QuantizedLinear):
            continue
        leaf = str(path[-1]) if path else type(sub).__name__
        out_axis = sub.kernel_axes[1]
        shards = int(sub.mesh.shape[out_axis]) if out_axis is not None else 1
        n_local = int(sub.weight_q.value.shape[0]) // max(shards, 1)
        if sub.pad_out_rows(multiple):
            count += 1
            padded[leaf] = padded.get(leaf, 0) + 1
        elif n_local % multiple != 0 and sub.n_out_valid is None:
            scale = sub.weight_scale.value
            key = f"{leaf}:scale-ndim-{getattr(scale, 'ndim', -1)}"
            skipped[key] = skipped.get(key, 0) + 1
    if count or skipped:
        logger.info(
            "Pre-padded %d replicated-N block-quant linears to a multiple of %d "
            "output rows (removes per-step weight padding): padded=%s skipped=%s seen=%s",
            count,
            multiple,
            dict(sorted(padded.items())),
            dict(sorted(skipped.items())),
            dict(sorted(seen.items())),
        )
    return count
