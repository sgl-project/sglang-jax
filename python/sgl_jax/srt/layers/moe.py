import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping


class GateLogit(nnx.Module):
    def __init__(
        self,
        input_size: int,
        num_experts: int = 0,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        enable_expert_bias: bool | None = False,
        score_func: str | None = "softmax",
    ):
        self.weight_dtype = weight_dtype
        self.enable_expert_bias = enable_expert_bias
        self.score_func = score_func

        self.kernel = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (input_size, num_experts),
                dtype=self.weight_dtype,
                out_sharding=P(None, None),
            ),
        )
        if enable_expert_bias:
            self.bias = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts,),
                    dtype=self.weight_dtype,
                    out_sharding=P(None),
                ),
            )
        else:
            self.bias = None

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        logits = jnp.dot(hidden_states, self.kernel.value)

        if self.score_func:
            if self.score_func == "softmax":
                logits = jax.nn.softmax(logits, axis=-1)
            elif self.score_func == "sigmoid":
                logits = jax.nn.sigmoid(logits)
            elif self.score_func == "tanh":
                logits = jax.nn.tanh(logits)
            else:
                raise ValueError("unknown score func")

        return logits


class TopK(nnx.Module):
    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        routed_scaling_factor: float | None = None,
    ):
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

    @named_scope
    def __call__(self, router_logits: jax.Array, correction_bias: jax.Array = None):
        router_logits = router_logits.astype(jnp.float32)

        if self.num_expert_group > 0 or self.topk_group > 0:
            if correction_bias is not None:
                topk_weights, topk_ids = self._biased_grouped_topk(router_logits, correction_bias)
            else:
                topk_weights, topk_ids = self._grouped_topk(router_logits)
        else:
            if correction_bias is not None:
                topk_weights, topk_ids = self._biased_topk(router_logits, correction_bias)
            else:
                topk_weights, topk_ids = self._topk(router_logits)

        if self.renormalize:
            topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True))
            if self.routed_scaling_factor is not None:
                topk_weights *= self.routed_scaling_factor

        topk_weights = topk_weights.astype(jnp.float32)

        return topk_weights, topk_ids

    def _topk(self, router_logits):
        return jax.lax.top_k(router_logits, self.topk)

    def _biased_topk(self, router_logits, correction_bias):
        n_routed_experts = router_logits.shape[-1]
        scores_for_choice = router_logits.reshape(-1, n_routed_experts) + jnp.expand_dims(
            correction_bias, axis=0
        )
        topk_ids = jax.lax.top_k(scores_for_choice, self.topk)[1]
        topk_weights = jnp.take_along_axis(router_logits, topk_ids, axis=1)
        return topk_weights, topk_ids

    def _grouped_topk(
        self,
        router_logits: jax.Array,
    ):
        num_token = router_logits.shape[0]

        # Group scores calculation
        group_shape = (num_token, self.num_expert_group, -1)
        scores_grouped = router_logits.reshape(group_shape)
        group_scores = jnp.max(scores_grouped, axis=-1)  # [n, n_group]

        # Get top group indices # [n, top_k_group]
        group_idx = jax.lax.top_k(group_scores, k=self.topk_group)[1]

        # Create group mask using scatter
        group_mask = jnp.zeros_like(group_scores)  # [n, n_group]
        token_indices = jnp.arange(num_token)[:, None]
        group_mask = group_mask.at[token_indices, group_idx].set(1)  # [n, n_group]

        # Create score mask
        experts_per_group = router_logits.shape[-1] // self.num_expert_group
        score_mask = jnp.expand_dims(group_mask, axis=-1)  # [n, n_group, 1]
        score_mask = jnp.broadcast_to(
            score_mask, (num_token, self.num_expert_group, experts_per_group)
        )
        score_mask = score_mask.reshape(num_token, -1)  # [n, e]

        # Apply mask and get topk
        tmp_scores = jnp.where(score_mask, router_logits, 0.0)  # [n, e]
        topk_weights, topk_ids = jax.lax.top_k(tmp_scores, k=self.topk)

        return topk_weights, topk_ids

    def _biased_grouped_topk(
        self,
        router_logits: jax.Array,
        correction_bias: jax.Array = None,
    ):
        num_token = router_logits.shape[0]
        scores_for_choice = router_logits.reshape(num_token, -1) + jnp.expand_dims(
            correction_bias, axis=0
        )

        # Group scores calculation
        scores_grouped = scores_for_choice.reshape(num_token, self.num_expert_group, -1)
        group_scores = jnp.sum(jax.lax.top_k(scores_grouped, k=2)[0], axis=-1)  # [n, n_group]

        # Get top group indices [n, top_k_group]
        group_idx = jax.lax.top_k(group_scores, k=self.topk_group)[1]

        # Create group mask using scatter
        group_mask = jnp.zeros_like(group_scores)  # [n, n_group]
        token_indices = jnp.arange(num_token)[:, None]
        group_mask = group_mask.at[token_indices, group_idx].set(1)  # [n, n_group]

        # Create score mask
        experts_per_group = router_logits.shape[-1] // self.num_expert_group
        score_mask = jnp.expand_dims(group_mask, axis=-1)  # [n, n_group, 1]
        score_mask = jnp.broadcast_to(
            score_mask, (num_token, self.num_expert_group, experts_per_group)
        )
        score_mask = score_mask.reshape(num_token, -1)  # [n, e]

        # Apply mask and get topk
        tmp_scores = jnp.where(score_mask, scores_for_choice, float("-inf"))  # [n, e]

        topk_ids = jax.lax.top_k(tmp_scores, k=self.topk)[1]
        topk_weights = jnp.take_along_axis(router_logits, topk_ids, axis=1)

        return topk_weights, topk_ids


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
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
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

        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        world_size = self.mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = num_experts // self.ep_size

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
            # MOE weights' shape is (num_experts, n, k)
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, intermediate_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
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

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place or initialize params for static loading."""
        if self.quantized_dtype is None:
            return

        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            if is_static:
                scale_sharding = P("expert", None, None, None)

                if hasattr(self, "wi_0_scale"):
                    del self.wi_0_scale
                self.wi_0_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=scale_sharding
                )

                if hasattr(self, "wi_1_scale"):
                    del self.wi_1_scale
                self.wi_1_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=scale_sharding
                )

                if hasattr(self, "wo_scale"):
                    del self.wo_scale
                self.wo_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=scale_sharding
                )
                return

            # Quantize weights
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=2,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=2,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=2,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", "tensor", None))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", "tensor", None))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", None, "tensor"))

            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1]),
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale.reshape(w1_scale.shape[0], 1, 1, w1_scale.shape[1]),
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale.reshape(wo_scale.shape[0], 1, 1, wo_scale.shape[1]),
                out_sharding=P("expert", None, None, "tensor"),
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

            w0_scale = self.wi_0_scale.value if self.wi_0_scale is not None else None
            w1_scale = self.wi_1_scale.value if self.wi_1_scale is not None else None
            wo_scale = self.wo_scale.value if self.wo_scale is not None else None

            result = shard_map(
                self._forward,
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
                    # weights
                    P("expert", "tensor", None),
                    P("expert", "tensor", None),
                    P("expert", None, "tensor"),
                    # scales
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, None),
                    # biases (unused)
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
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
        replicated_pspec = P(*([None] * result.ndim))
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

        x, sorted_selected_experts, weights, group_sizes = self._permute(
            hidden_states, topk_ids, topk_weights
        )

        group_sizes = group_sizes.astype(jnp.int32)

        group_offset = self._dispatch(group_sizes, expert_shard_id)

        intermediate_output = self._gmm_compute(
            x,
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

        if self.ep_size > 1:
            intermediate_output = self._combine(intermediate_output)

        output = self._unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size,
            seq_len,
        )
        return output

    def _gmm_compute(
        self,
        x,
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
        if x.shape[0] == 0:
            empty_output = jnp.zeros((0, wo_kernel.shape[-1]), dtype=x.dtype)
            return empty_output

        m, k = x.shape[0], x.shape[1]
        n_gate = w0_kernel.shape[1]
        n_down = wo_kernel.shape[1]

        default_tile_size = (512, 1024, 1024)
        tiling_gate = (
            min(default_tile_size[0], m),
            min(default_tile_size[1], k),
            min(default_tile_size[2], n_gate),
        )
        tiling_down = (
            min(default_tile_size[0], m),
            min(default_tile_size[1], n_gate),
            min(default_tile_size[2], n_down),
        )

        group_sizes = group_sizes.astype(jnp.int32)

        # === GEMM1: x @ w0 and x @ w1 ===
        # Quantize input activation for GEMM1 if activation quantization enabled
        if self.activation_quantized_dtype is not None:
            x_q, x_scale = quantize_tensor_simple(x, self.activation_quantized_dtype, dim=-1)
            gemm1_lhs = x_q
        else:
            gemm1_lhs = x
            x_scale = None

        layer_w0 = gmm(
            lhs=gemm1_lhs,
            rhs=w0_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            rhs_scale=w0_kernel_scale,
            rhs_bias=w0_kernel_bias,
            tiling=tiling_gate,
            group_offset=group_offset,
            interpret=not is_tpu_runtime(),
        )

        layer_w1 = gmm(
            lhs=gemm1_lhs,
            rhs=w1_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            rhs_scale=w1_kernel_scale,
            rhs_bias=w1_kernel_bias,
            tiling=tiling_gate,
            group_offset=group_offset,
            interpret=not is_tpu_runtime(),
        )

        # Dequantize GEMM1 output (apply LHS scale if quantized)
        if x_scale is not None:
            # x_scale shape: (m, 1) with keepdims=True, broadcasts to (m, n_gate)
            layer_w0 = layer_w0 * x_scale
            layer_w1 = layer_w1 * x_scale

        # === Activation in BF16 (not quantized) ===
        if self.activation == "silu":
            layer_act = jax.nn.silu(layer_w0)
        elif self.activation == "gelu":
            layer_act = jax.nn.gelu(layer_w0)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")
        intermediate_layer = jnp.multiply(layer_act, layer_w1)

        # === GEMM2: intermediate @ wo ===
        # Quantize intermediate activation for GEMM2 if activation quantization enabled
        if self.activation_quantized_dtype is not None:
            intermediate_q, intermediate_scale = quantize_tensor_simple(
                intermediate_layer, self.activation_quantized_dtype, dim=-1
            )
            gemm2_lhs = intermediate_q
        else:
            gemm2_lhs = intermediate_layer
            intermediate_scale = None

        intermediate_output = gmm(
            lhs=gemm2_lhs,
            rhs=wo_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            rhs_scale=wo_kernel_scale,
            rhs_bias=wo_kernel_bias,
            tiling=tiling_down,
            group_offset=group_offset,
            interpret=not is_tpu_runtime(),
        )

        # Dequantize GEMM2 output (apply LHS scale if quantized)
        if intermediate_scale is not None:
            # intermediate_scale shape: (m, 1) with keepdims=True, broadcasts to (m, n_down)
            intermediate_output = intermediate_output * intermediate_scale

        if self.tp_size > 1:
            intermediate_output = jax.lax.psum(intermediate_output, "tensor")

        return intermediate_output

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
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok

        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        return (
            sorted_inputs,
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


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1/w3 are (num_experts, hidden_size, intermediate_size) for gate/up proj
      and w2 is (num_experts, intermediate_size, hidden_size) for down proj
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        hidden_size: Hidden size of the model
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

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
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )
        self.mesh = mesh

        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w1_scale = None
        self.w3_scale = None
        self.w2_scale = None

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

        self.w1_shared_scale = None
        self.w3_shared_scale = None
        self.w2_shared_scale = None

        self.subc_quant_wsz = None  # Use default sub channel quantization block size
        self.enable_comm_quant = True

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return

        if hasattr(self, "subc_quant_wsz"):
            del self.subc_quant_wsz
            self.subc_quant_wsz = 256

        with jax.set_mesh(self.mesh):
            if is_static:
                ep_scale_sharding = P(("data", "tensor"), None, None, None)

                if hasattr(self, "w1_scale"):
                    del self.w1_scale
                self.w1_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=ep_scale_sharding
                )

                if hasattr(self, "w3_scale"):
                    del self.w3_scale
                self.w3_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=ep_scale_sharding
                )

                if hasattr(self, "w2_scale"):
                    del self.w2_scale
                self.w2_scale = nnx.Param(
                    jnp.zeros((1,), dtype=jnp.float32), out_sharding=ep_scale_sharding
                )

                if self.num_shared_experts > 0:
                    shared_scale_sharding = P(None, None, None)

                    if hasattr(self, "w1_shared_scale"):
                        del self.w1_shared_scale
                    self.w1_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                    if hasattr(self, "w3_shared_scale"):
                        del self.w3_shared_scale
                    self.w3_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                    if hasattr(self, "w2_shared_scale"):
                        del self.w2_shared_scale
                    self.w2_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                return

            # Replace original weights with quantized versions
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.w1.value,
                axis=1,
                block_size=self.subc_quant_wsz,
            )
            w3_value, w3_scale = quantize_tensor(
                self.quantized_dtype,
                self.w3.value,
                axis=1,
                block_size=self.subc_quant_wsz,
            )
            w2_value, w2_scale = quantize_tensor(
                self.quantized_dtype,
                self.w2.value,
                axis=1,
                block_size=self.subc_quant_wsz,
            )

            # NOTE: Fused MoE shards the expert dimension across EP=(data*tensor).
            ep_sharding = P(("data", "tensor"), None, None)
            ep_scale_sharding = P(("data", "tensor"), None, None, None)

            self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
            self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
            self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)

            # Update scales (reshape to 4D for GMM kernel)
            if hasattr(self, "w1_scale"):
                del self.w1_scale
            self.w1_scale = nnx.Param(
                w1_scale.reshape(w1_scale.shape[0], w1_scale.shape[1], 1, w1_scale.shape[2]),
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w3_scale"):
                del self.w3_scale
            self.w3_scale = nnx.Param(
                w3_scale.reshape(w3_scale.shape[0], w3_scale.shape[1], 1, w3_scale.shape[2]),
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w2_scale"):
                del self.w2_scale
            self.w2_scale = nnx.Param(
                w2_scale.reshape(w2_scale.shape[0], w2_scale.shape[1], 1, w2_scale.shape[2]),
                out_sharding=ep_scale_sharding,
            )

            if self.w1_shared is not None:
                w1_shared_value, w1_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1_shared.value,
                    axis=0,
                )
                w3_shared_value, w3_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3_shared.value,
                    axis=0,
                )
                w2_shared_value, w2_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2_shared.value,
                    axis=0,
                )

                self.w1_shared = nnx.Param(w1_shared_value, out_sharding=P(None, None))
                self.w3_shared = nnx.Param(w3_shared_value, out_sharding=P(None, None))
                self.w2_shared = nnx.Param(w2_shared_value, out_sharding=P(None, None))

                if hasattr(self, "w1_shared_scale"):
                    del self.w1_shared_scale
                self.w1_shared_scale = nnx.Param(
                    w1_shared_scale.reshape(
                        1,
                        1,
                        w1_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w3_shared_scale"):
                    del self.w3_shared_scale
                self.w3_shared_scale = nnx.Param(
                    w3_shared_scale.reshape(
                        1,
                        1,
                        w3_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w2_shared_scale"):
                    del self.w2_shared_scale
                self.w2_shared_scale = nnx.Param(
                    w2_shared_scale.reshape(
                        1,
                        1,
                        w2_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

    def __call__(
        self,
        hidden_states: jax.Array,
        router_logits: jax.Array,
        router_bias: jax.Array | None = None,
        token_valid_mask: jax.Array | None = None,
        *,
        block_config: FusedMoEBlockConfig | None = None,
    ) -> jax.Array:
        """
        Forward pass through the fused MoE layer.

        Args:
            hidden_states: Input tokens, shape (num_tokens, hidden_size) or
                          (batch_size, seq_len, hidden_size)
            router_logits: Router output logits, shape (num_tokens, num_experts)
                          Note: Should be raw logits, not after softmax or top-k

        Returns:
            MoE layer output, same shape as hidden_states
        """
        assert hidden_states.ndim == 2

        if router_bias is not None:
            router_bias = jax.sharding.reshard(router_bias, P())

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None
        # w1_shared_val = None
        # w3_shared_val = None
        # w2_shared_val = None

        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        w1_shared_scale = self.w1_shared_scale.value if self.w1_shared_scale is not None else None
        w3_shared_scale = self.w3_shared_scale.value if self.w3_shared_scale is not None else None
        w2_shared_scale = self.w2_shared_scale.value if self.w2_shared_scale is not None else None
        # w1_shared_scale = None
        # w3_shared_scale = None
        # w2_shared_scale = None
        subc_quant_wsz = self.subc_quant_wsz if self.subc_quant_wsz is not None else None

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            gating_output=router_logits,
            bias=router_bias,
            top_k=self.num_experts_per_tok,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            act_fn=self.activation,
            block_config=block_config,
            token_valid_mask=token_valid_mask,
            # Optional parameters (not used in basic case)
            subc_quant_wsz=subc_quant_wsz,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            b1=None,
            b2=None,
            b3=None,
            dp_axis_name="data",
            tp_axis_name="tensor",
            enable_comm_quant=self.enable_comm_quant,
        )

        output = jax.sharding.reshard(output, NamedSharding(self.mesh, P("data", None)))
        return output


# create_moe_weights_mapping is utility function to generate weight mapping for MOE layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
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

        # Source weight paths for all experts to be loaded and concatenated
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Sharding logic based on EPMoE PartitionSpec:
            # wi_0/wi_1 (Input projections) use P("expert", "tensor", None)
            # wo (Output projection) uses P("expert", None, "tensor")
            sharding = (
                ("expert", None, "tensor") if target_name == "wo" else ("expert", "tensor", None)
            )
            transpose = False
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
        )

    return mappings
