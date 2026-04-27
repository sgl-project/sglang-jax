"""GMM-based Expert-Parallel MoE layer and weight mapping utilities."""

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

# Re-export for backward compatibility: external code imports from this module.
from sgl_jax.srt.layers.fused_moe import FusedEPMoE  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping


class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        self.physical_to_logical_map = physical_to_logical_map
        self.pre_gather_quant_dtype = pre_gather_quant_dtype

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype  # original dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.original_mesh = mesh
        self.mesh = mesh
        self.activation = activation
        self.hidden_size = hidden_size

        # Get quantization settings from config
        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        self.weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        world_size = self.mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = self.num_experts // self.ep_size

        devices = self.mesh.devices.flatten()
        self.moe_mesh = jax.sharding.Mesh(
            devices.reshape(self.ep_size, self.tp_size),
            axis_names=("expert", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )

        abstract_mesh = self.mesh.abstract_mesh
        self.updated_mesh = abstract_mesh.update(
            axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
        )

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            # MOE weights' shape is (num_experts, k, n)
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (self.num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            # Scales are None by default - only set by quantize_weights() if quantization is enabled
            # gmm kernel handles None scales properly (no scaling applied)
            self.wi_0_scale = None
            self.wi_1_scale = None
            self.wo_scale = None

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            return can_use_ragged, primary_device
        except Exception as _:
            return False, "cpu"

    def _normalize_scale_for_gmm(
        self,
        scale: jax.Array | None,
        weight: jax.Array,
        *,
        scale_name: str,
    ) -> jax.Array | None:
        """Normalize offline/runtime scale tensors to GMM's 4D layout.

        Accepted inputs intentionally cover the layouts we see in practice:

        - per-channel: ``[E, out_dim]``
        - already-kernel-ready: ``[E, k_blocks, 1, out_dim]``
        - sub-channel / block-channel: ``[E, out_dim, k_blocks]`` or
          ``[E, k_blocks, out_dim]``
        - offline 2D block quant: ``[E, out_blocks, k_blocks]``

        The returned tensor always matches the GMM contract
        ``[E, k_blocks, 1, out_dim]``.
        """
        if scale is None:
            return None

        # Weight layout is [E, k, n] where k=contraction dim, n=output dim.
        num_experts, in_dim, out_dim = weight.shape

        if scale.ndim == 4:
            if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
                raise ValueError(
                    f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                    "Expected 4D GMM scale layout [E, k_blocks, 1, out_dim]."
                )
            if self.weight_block_size is None:
                if scale.shape[1] != 1:
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        "Per-channel 4D GMM scales must have k_blocks=1."
                    )
            else:
                block_size_k = int(self.weight_block_size[1])
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k
                if scale.shape[1] not in (1, expected_k_blocks):
                    raise ValueError(
                        f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
                        f"Expected k_blocks dimension to be 1 or {expected_k_blocks}."
                    )
            return scale

        if scale.ndim == 2 and scale.shape == (num_experts, out_dim):
            return scale[:, None, None, :]

        if scale.ndim == 3:
            if scale.shape == (num_experts, 1, out_dim):
                return scale[:, :, None, :]

            # Support offline 2D block quant checkpoints whose scales are stored as
            # [num_experts, out_blocks, in_blocks]. GMM expects [E, k_blocks, 1, out_dim].
            if (
                self.weight_block_size is not None
                and isinstance(self.weight_block_size, (list, tuple))
                and len(self.weight_block_size) == 2
            ):
                block_size_out = int(self.weight_block_size[0])
                block_size_k = int(self.weight_block_size[1])
                expected_out_blocks = (out_dim + block_size_out - 1) // block_size_out
                expected_k_blocks = (in_dim + block_size_k - 1) // block_size_k

                if scale.shape == (num_experts, out_dim, expected_k_blocks):
                    final_scale_sharding = (
                        P("expert", None, None, None)
                        if scale_name == "wo_scale"
                        else P("expert", None, None, "tensor")
                    )
                    scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
                    return jax.sharding.reshard(scale_gmm, final_scale_sharding)

                if scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
                    scale_per_out_sharding = (
                        P("expert", None, None)
                        if scale_name == "wo_scale"
                        else P("expert", "tensor", None)
                    )
                    final_scale_sharding = (
                        P("expert", None, None, None)
                        if scale_name == "wo_scale"
                        else P("expert", None, None, "tensor")
                    )
                    out_block_ids = jnp.arange(out_dim, dtype=jnp.int32) // block_size_out
                    scale_per_out = scale.at[:, out_block_ids, :].get(
                        out_sharding=scale_per_out_sharding
                    )
                    scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
                    return jax.sharding.reshard(scale_gmm, final_scale_sharding)

                if scale.shape == (num_experts, expected_k_blocks, out_dim):
                    return scale[:, :, None, :]

        raise ValueError(
            f"Unsupported {scale_name} shape {scale.shape} for weight shape {weight.shape}. "
            "Expected one of: [E, out_dim], [E, 1, out_dim], [E, k_blocks, 1, out_dim], "
            "or offline block format [E, out_blocks, k_blocks]."
        )

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place or initialize params for static loading."""
        if self.quantized_dtype is None:
            return

        def _get_block_size_k(
            *,
            hidden_size: int,
            intermediate_dim: int,
            weight_block_size: list[int] | tuple[int, int] | None,
        ) -> int | None:
            """Extract the contracting-dimension block size for MoE weights.

            EPMoE only block-quantizes along the GEMM ``K`` dimension, so for a
            configured ``(block_n, block_k)`` we consume only ``block_k`` here.
            The divisibility checks keep the later GMM scale layout well-defined.
            """
            if weight_block_size is None:
                return None
            if not (isinstance(weight_block_size, (list, tuple)) and len(weight_block_size) == 2):
                raise ValueError(
                    f"EPMoE weight_block_size must be a 2-element list [block_n, block_k], "
                    f"got {weight_block_size}"
                )

            block_size_k = int(weight_block_size[1])
            if block_size_k <= 0:
                raise ValueError(f"EPMoE weight_block_size[1] must be > 0, got {block_size_k}")
            if hidden_size % block_size_k != 0:
                raise ValueError(
                    f"EPMoE hidden_size={hidden_size} not divisible by block_size_k={block_size_k}"
                )
            if intermediate_dim % block_size_k != 0:
                raise ValueError(
                    f"EPMoE intermediate_dim={intermediate_dim} not divisible by block_size_k={block_size_k}"
                )
            return block_size_k

        with jax.set_mesh(self.moe_mesh):
            if is_static:
                # Static checkpoints will load real scale tensors later, but the
                # placeholders must already satisfy expert sharding shape rules.
                num_experts = self.wi_0.value.shape[0]
                # [E, k, n] layout: wi_0=[E, hidden_size, intermediate_dim],
                #                    wo=[E, intermediate_dim, hidden_size]
                hidden_size = self.wi_0.value.shape[1]
                intermediate_dim = self.wo.value.shape[1]

                # Compute k_blocks for block quant placeholders.
                # weight_block_size = [hf_out_block, hf_in_block] (HF convention).
                # EPMoE quantizes along axis=1 (k/contraction dim).
                block_size_k = _get_block_size_k(
                    hidden_size=hidden_size,
                    intermediate_dim=intermediate_dim,
                    weight_block_size=self.weight_block_size,
                )
                k_blocks_wi = (hidden_size // block_size_k) if block_size_k else 1
                k_blocks_wo = (intermediate_dim // block_size_k) if block_size_k else 1
                wi_scale_sharding = P("expert", None, None, "tensor")
                wo_scale_sharding = P("expert", None, None, None)

                if hasattr(self, "wi_0_scale"):
                    del self.wi_0_scale
                self.wi_0_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wi, 1, intermediate_dim),
                        dtype=jnp.float32,
                        out_sharding=wi_scale_sharding,
                    ),
                    out_sharding=wi_scale_sharding,
                )

                if hasattr(self, "wi_1_scale"):
                    del self.wi_1_scale
                self.wi_1_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wi, 1, intermediate_dim),
                        dtype=jnp.float32,
                        out_sharding=wi_scale_sharding,
                    ),
                    out_sharding=wi_scale_sharding,
                )

                if hasattr(self, "wo_scale"):
                    del self.wo_scale
                self.wo_scale = nnx.Param(
                    jnp.zeros(
                        (num_experts, k_blocks_wo, 1, hidden_size),
                        dtype=jnp.float32,
                        out_sharding=wo_scale_sharding,
                    ),
                    out_sharding=wo_scale_sharding,
                )
                return

            # Quantize weights along k-dim (axis=1 in [g, k, n] layout)
            # wi_0=[E, hidden_size, intermediate_dim], wo=[E, intermediate_dim, hidden_size]
            hidden_size = self.wi_0.value.shape[1]
            intermediate_dim = self.wo.value.shape[1]
            block_size_k = _get_block_size_k(
                hidden_size=hidden_size,
                intermediate_dim=intermediate_dim,
                weight_block_size=self.weight_block_size,
            )
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=1,
                block_size=block_size_k,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=1,
                block_size=block_size_k,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=1,
                block_size=block_size_k,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", "tensor", None))

            if block_size_k is not None:
                # axis=1 quantization on [g, k, n] gives scale [g, k_blocks, n]
                # → expand to [g, k_blocks, 1, n]
                w0_scale = w0_scale[:, :, None, :]
                w1_scale = w1_scale[:, :, None, :]
                wo_scale = wo_scale[:, :, None, :]
            else:
                w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
                w1_scale = w1_scale.reshape(w1_scale.shape[0], 1, 1, w1_scale.shape[1])
                wo_scale = wo_scale.reshape(wo_scale.shape[0], 1, 1, wo_scale.shape[1])

            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale,
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale,
                out_sharding=P("expert", None, None, None),
            )

    @named_scope
    def __call__(self, hidden_states, topk_weights, topk_ids) -> jax.Array:
        # Activation quantization is now handled per-GEMM inside _gmm_compute
        # (aligned with sglang-gpu scheme: quantize before each GEMM, dequantize after)

        # Run MoE computation on the expert-parallel mesh
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))

            # Normalize scales to GMM's 4D layout [E, k_blocks, 1, out_dim]
            w0_scale = self._normalize_scale_for_gmm(
                self.wi_0_scale.value if self.wi_0_scale is not None else None,
                self.wi_0.value,
                scale_name="wi_0_scale",
            )
            w1_scale = self._normalize_scale_for_gmm(
                self.wi_1_scale.value if self.wi_1_scale is not None else None,
                self.wi_1.value,
                scale_name="wi_1_scale",
            )
            wo_scale = self._normalize_scale_for_gmm(
                self.wo_scale.value if self.wo_scale is not None else None,
                self.wo.value,
                scale_name="wo_scale",
            )

            result = shard_map(
                self._forward,
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    # weights [g, k, n]
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    # scales [g, 1, 1, n]
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, None),
                    # biases [g, 1, n] (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", None, None),
                ),
                out_specs=P(None),
                check_vma=False,
            )(
                hidden_states_reshard,
                topk_weights_reshard,
                topk_ids_reshard,
                self.wi_0.value,
                self.wi_1.value,
                self.wo.value,
                w0_scale,
                w1_scale,
                wo_scale,
                None,
                None,
                None,
            )

        # Reshard result back to original mesh
        replicated_pspec = P("data", *([None] * (result.ndim - 1)))
        return jax.sharding.reshard(result, jax.sharding.NamedSharding(self.mesh, replicated_pspec))

    def _forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w0_weights,
        w1_weights,
        wo_weights,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
    ):
        expert_shard_id = jax.lax.axis_index("expert")

        if hidden_states.ndim == 2:
            total_tokens = hidden_states.shape[0]
            batch_size, seq_len = 1, total_tokens
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            total_tokens = batch_size * seq_len

        inputs_2d, token_indices, sorted_selected_experts, weights, group_sizes = self._permute(
            hidden_states, topk_ids, topk_weights
        )

        group_sizes = group_sizes.astype(jnp.int32)

        group_offset = self._dispatch(group_sizes, expert_shard_id)

        intermediate_output = self._gmm_compute(
            inputs_2d,
            token_indices,
            group_sizes,
            w0_weights,
            w1_weights,
            wo_weights,
            group_offset,
            w0_kernel_scale,
            w1_kernel_scale,
            wo_kernel_scale,
            w0_kernel_bias,
            w1_kernel_bias,
            wo_kernel_bias,
        )

        output = self._unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size,
            seq_len,
        )

        # All-reduce after unpermute: communication volume is (T, hidden_size)
        # instead of (T * top_k, hidden_size), reducing by a factor of top_k.
        if self.tp_size > 1:
            output = jax.lax.psum(output, "tensor")
        if self.ep_size > 1:
            output = self._combine(output)

        return output

    def _gmm_compute(
        self,
        inputs_2d,
        token_indices,
        group_sizes,
        w0_kernel,
        w1_kernel,
        wo_kernel,
        group_offset,
        w0_kernel_scale=None,
        w1_kernel_scale=None,
        wo_kernel_scale=None,
        w0_kernel_bias=None,
        w1_kernel_bias=None,
        wo_kernel_bias=None,
    ):
        if token_indices.shape[0] == 0:
            return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)

        # indexed_gmm: gather sorted_inputs here instead of in _permute,
        # so XLA can fuse the gather with the matmul and avoid materializing
        # the full [M*top_k, D] sorted_inputs tensor at peak memory.
        pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
        if pre_gather_q is not None:
            x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
            x = x_q[token_indices]
            x_scale = x_scale[token_indices]
            x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
        else:
            x = inputs_2d[token_indices].astype(self.dtype)

        # TODO(Qinghan): DeepSeek-V2-Lite has num_experts_per_tok=6, so with
        # the default power-of-2 bs bucketing `padded_bs * 6` can land on a
        # value > 16 that isn't a multiple of 16 (e.g. bs=4 -> size_m=24),
        # which is why this padding is necessary. Models with power-of-2
        # top_k (Grok=2, DeepSeek-V3=8, Qwen3-MoE=8) wouldn't need it.
        from jax.experimental.pallas import tpu as pltpu

        sublane_align = pltpu.get_tpu_info().get_sublane_tiling(x.dtype)
        pad_size = (-x.shape[0]) % sublane_align
        if pad_size > 0:
            x = jnp.pad(x, ((0, pad_size), (0, 0)))
            group_sizes = group_sizes.at[-1].add(pad_size)

        group_sizes = group_sizes.astype(jnp.int32)
        act_q_dtype = self.activation_quantized_dtype

        gmm_kwargs = dict(
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            group_offset=group_offset,
            maybe_quantize_lhs=act_q_dtype is not None,
            acc_dtype=jnp.float32,
        )

        # === GEMM1: x @ w0 and x @ w1 ===
        layer_w0 = gmm(
            lhs=x,
            rhs=w0_kernel,
            rhs_scale=w0_kernel_scale,
            rhs_bias=w0_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )
        layer_w1 = gmm(
            lhs=x,
            rhs=w1_kernel,
            rhs_scale=w1_kernel_scale,
            rhs_bias=w1_kernel_bias,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

        # === Activation ===
        if self.activation == "silu":
            layer_act = jax.nn.silu(layer_w0)
        elif self.activation == "gelu":
            layer_act = jax.nn.gelu(layer_w0)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")
        intermediate_layer = jnp.multiply(layer_act, layer_w1)

        # === GEMM2: intermediate @ wo ===
        return gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            rhs_scale=wo_kernel_scale,
            rhs_bias=wo_kernel_bias,
            zero_initialize=True,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

    def _dispatch(self, group_sizes, expert_shard_id):
        if self.ep_size <= 1:
            return jnp.array(0, dtype=jnp.int32)
        group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
        return group_offset

    def _get_all_to_all_params(
        self,
        tokens_group: jax.Array,
        shard_id: jax.Array,
        start_idx: jax.Array,
        *,
        ep_size: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        input_offsets = jnp.full(ep_size, start_idx, dtype=tokens_group.dtype)
        send_sizes = jnp.repeat(tokens_group[shard_id], ep_size)
        output_offset = jnp.concatenate(
            (jnp.array([0], dtype=tokens_group.dtype), jnp.cumsum(tokens_group[:-1]))
        )[shard_id]
        output_offsets = jnp.repeat(output_offset, ep_size)
        recv_sizes = tokens_group

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _combine(self, data):
        return jax.lax.psum(data, "expert")

    def _permute(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape

        if len(inputs_shape) == 2:
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))

        del bsz_times_seq_len

        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        # token_indices: maps each sorted position to the original token index.
        # Pass to _gmm_compute so the gather happens there (indexed_gmm pattern),
        # avoiding a full [M*top_k, D] materialization in _permute.
        token_indices = sorted_selected_experts // self.num_experts_per_tok

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        return (
            inputs_2d,
            token_indices,
            sorted_selected_experts,
            top_k_weights,
            group_sizes,
        )

    def _unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, seq_len):
        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros((padding_size, intermediate.shape[1]), dtype=intermediate.dtype)
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        argsort_indices = jnp.argsort(sorted_selected_experts, stable=True)
        unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)

        total_tokens = weights.shape[0] * weights.shape[1] // self.num_experts_per_tok

        reshaped_weights = jnp.reshape(weights, (total_tokens, self.num_experts_per_tok))
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (total_tokens, self.num_experts_per_tok, -1),
        )

        intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
        weights_fp32 = reshaped_weights.astype(jnp.float32)

        output = jnp.einsum(
            "BKE,BK -> BE",
            intermediate_fp32,
            weights_fp32,
        )

        if len(weights.shape) == 2:
            final_output = output.astype(self.dtype)
        else:
            final_output = output.reshape(batch_size, seq_len, -1).astype(self.dtype)

        return final_output


# create_moe_weights_mapping is utility function to generate weight mapping for MOE layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,  # num logical experts
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),  # expert source names [gate, up, down]
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is None)
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,  # np.ndarray shape (num_physical,) or None
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend == "fused":
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    if expert_concat_axis_map is None:
        expert_concat_axis_map = {}

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters (matching EPMoE internal variables)
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths for logical experts only
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Weights are transposed from HF [n, k] to [k, n], stacked to [g, k, n].
            # wi_0/wi_1: [g, hidden_size, intermediate_dim] -> P("expert", None, "tensor")
            # wo:        [g, intermediate_dim, hidden_size] -> P("expert", "tensor", None)
            sharding = (
                ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            )
            transpose = True
        elif moe_backend == "fused":
            # Fused MoE kernel shards experts across the full EP mesh, i.e. the
            # product of ("data", "tensor"). Shard expert dim (axis=0) across
            # both mesh axes so each device owns a disjoint expert slice.
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use __MOE_EXPERTS__ prefix to indicate aggregated MoE weight loading
        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )

    return mappings
