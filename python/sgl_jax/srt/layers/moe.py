import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
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
            nnx.with_partitioning(nnx.initializers.normal(), (None, None))(
                jax.random.PRNGKey(0), (input_size, num_experts), self.weight_dtype
            )
        )
        if enable_expert_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.zeros_init(), (None,))(
                    jax.random.PRNGKey(0), (num_experts,), self.weight_dtype
                )
            )
        else:
            self.bias = None

    def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, jax.Array | None]:
        logits = hidden_states.astype(jnp.float32) @ self.kernel

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

    def __call__(self, router_logits: jax.Array, correction_bias: jax.Array = None):
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

        topk_weights = topk_weights.astype(router_logits.dtype)

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
        config,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
    ):
        self.config = config
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

        # Get quantization settings from config
        quant_config = getattr(config, "quantization_config", None)
        self.quantized_dtype = quant_config.get_moe_weight_dtype() if quant_config else None
        self.activation_quantized_dtype = (
            quant_config.get_moe_activation_dtype() if quant_config else None
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
                    (num_experts, intermediate_dim, config.hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, intermediate_dim, config.hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, config.hidden_size, intermediate_dim),
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

    def quantize_weights(self):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return

        # Replace original weights with quantized versions
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            # Quantize weights
            w0_value, w0_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_0.value,
                axis=2,
                pad_tensor=True,
            )
            w1_value, w1_scale = quantize_tensor(
                self.quantized_dtype,
                self.wi_1.value,
                axis=2,
                pad_tensor=True,
            )
            wo_value, wo_scale = quantize_tensor(
                self.quantized_dtype,
                self.wo.value,
                axis=2,
                pad_tensor=True,
            )

            self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", "tensor", None))
            self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", "tensor", None))
            self.wo = nnx.Param(wo_value, out_sharding=P("expert", None, "tensor"))

            # Update scales (reshape to 4D for GMM kernel)
            # Wrap with nnx.data() to override static attribute status
            if hasattr(self, "wi_0_scale"):
                del self.wi_0_scale
            self.wi_0_scale = nnx.Param(
                w0_scale.reshape(
                    w0_scale.shape[0],
                    1,
                    1,
                    w0_scale.shape[1],
                    out_sharding=P("expert", None, None, "tensor"),
                ),
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wi_1_scale"):
                del self.wi_1_scale
            self.wi_1_scale = nnx.Param(
                w1_scale.reshape(
                    w1_scale.shape[0],
                    1,
                    1,
                    w1_scale.shape[1],
                    out_sharding=P("expert", None, None, "tensor"),
                ),
                out_sharding=P("expert", None, None, "tensor"),
            )

            if hasattr(self, "wo_scale"):
                del self.wo_scale
            self.wo_scale = nnx.Param(
                wo_scale.reshape(
                    wo_scale.shape[0],
                    1,
                    1,
                    wo_scale.shape[1],
                    out_sharding=P("expert", None, None, "tensor"),
                ),
                out_sharding=P("expert", None, None, None),
            )

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


# create_moe_weights_mapping is utility function to generate weight mapping for EPMoe layers
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    expert_type_map: dict[
        str, str
    ] = None,  # Default mapping: HuggingFace weight name -> EPMoE internal variable name
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is 0)
    moe_path: str = "mlp",  # Path to the MoE module within a layer (e.g., "mlp" or "block_sparse_moe")
    source_expert_pattern: str = "experts.{i}",  # Pattern for expert indexing in the source weight file
) -> dict:
    """
    Generate a unified mapping dictionary for MoE layer expert weights.
    The sharding strategy is strictly aligned with the PartitionSpec defined in EPMoE.
    """
    if expert_type_map is None:
        expert_type_map = {
            "gate_proj": "wi_0",
            "up_proj": "wi_1",
            "down_proj": "wo",
        }
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

        # Sharding logic based on EPMoE PartitionSpec:
        # wi_0/wi_1 (Input projections) use P("expert", "tensor", None)
        # wo (Output projection) uses P("expert", None, "tensor")
        sharding = ("expert", None, "tensor") if target_name == "wo" else ("expert", "tensor", None)

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use __MOE_EXPERTS__ prefix to indicate aggregated MoE weight loading
        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=False,
            concat_axis=concat_axis,
        )

    return mappings
