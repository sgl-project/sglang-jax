from typing import Iterable, Optional, Sequence, Tuple, Union

import jax
from flax import nnx
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.debug_tracer import trace_function
from sgl_jax.srt.layers import linear


class GateLogit(nnx.Module):
    """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing.

    Attributes:
        input_size: input dimension of the layer.
        features: tuple with numbers of output features.
        model_name: which model to run.
        axis: tuple with axes to apply the transformation on.
        weight_dtype: the dtype of the weights (default: float32).
        dtype: the dtype of the computation (default: float32).
        kernel_axes: tuple with axes to apply kernel function.
        use_bias: whether to add learnable bias in gate logit scores.
          When enabled, this bias aids expert load balancing (like in DeepSeek V3),
          and is not part of the loss calculation.
        score_func: scoring function for output normalization before applying bias.
        matmul_precision: precision for JAX functions.
    """

    def __init__(
        self,
        input_size: int,
        features: Union[Iterable[int], int],
        model_name: str,
        axis: Union[Iterable[int], int] = -1,
        weight_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        kernel_axes: Optional[Sequence[str]] = None,
        use_bias: bool = False,
        score_func: str = "",
        matmul_precision: str = "default",
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
    ):

        self.features = linear._canonicalize_tuple(features)
        self.axis = linear._canonicalize_tuple(axis)
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.use_bias = use_bias
        self.score_func = score_func
        self.matmul_precision = matmul_precision
        self.layer_id = layer_id

        self.kernel_axes = kernel_axes or ()

        kernel_shape = (input_size,) + self.features

        self.kernel = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.kernel_axes)(
                rngs.params(), kernel_shape, self.weight_dtype
            )
        )

        if self.use_bias:
            bias_shape = self.features
            bias_axes = (
                self.kernel_axes[-len(self.features) :] if self.kernel_axes else ()
            )
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.zeros_init(), bias_axes)(
                    rngs.params(), bias_shape, self.weight_dtype
                )
            )
        else:
            self.bias = None

    @trace_function(stage="MOE_GATE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        inputs = jnp.asarray(inputs, self.dtype)


        kernel = jnp.asarray(self.kernel.value, self.dtype)
        output = jnp.dot(inputs, kernel)

        if self.score_func:
            if self.score_func == "softmax":
                output = jax.nn.softmax(output)
            elif self.score_func == "sigmoid":
                output = jax.nn.sigmoid(output)
            elif self.score_func == "tanh":
                output = jax.nn.tanh(output)


        if self.use_bias and self.bias is not None:
            bias = jnp.asarray(self.bias.value, self.dtype)
            output += bias

        return output


class EPMoE(nnx.Module):
    def __init__(
        self,
        config,
        num_experts: int,
        num_experts_per_tok: int,
        expert_parallel_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        layer_id: int = 0,
        rngs: nnx.Rngs = None,
    ):

        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.expert_parallel_size = expert_parallel_size
        self.mesh = mesh
        if num_experts % self.expert_parallel_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by expert_parallel_size ({self.expert_parallel_size})"
            )

        self.experts_per_device = num_experts // self.expert_parallel_size
        expert_kernel_axes = (("data", "tensor"), None, None)

        self.wi_0 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wi_1 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wo = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, intermediate_dim, config.hidden_size),
                weight_dtype,
            )
        )

        state = nnx.state(self)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(self, sharded_state)

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            return can_use_ragged, primary_device
        except Exception as e:
            return False, "cpu"

    @trace_function(stage="MOE_SPARSE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for EPMoE")

        inputs = inputs.astype(self.dtype)
        total_tokens, hidden_dim = inputs.shape

        if router_logits.shape[0] != total_tokens:
            raise ValueError(
                f"router_logits shape {router_logits.shape} doesn't match inputs shape {inputs.shape}"
            )

        if self.expert_parallel_size == 1:
            output = self._single_device_forward(inputs, router_logits)
        else:
            output = self._expert_parallel_forward_with_shard_map(inputs, router_logits)

        return output

    def _expert_parallel_forward_with_shard_map(self, inputs, router_logits):
        def _internal_moe_computation(
            hidden_states, router_logits, w0_weights, w1_weights, wo_weights
        ):
            data_index = jax.lax.axis_index("data")
            tensor_index = jax.lax.axis_index("tensor")
            tensor_size = jax.lax.axis_size("tensor")
            expert_shard_id = data_index * tensor_size + tensor_index

            # topk
            top_k_logits, top_k_indices = jax.lax.top_k(
                router_logits, self.num_experts_per_tok
            )
            top_k_weights = jax.nn.softmax(
                top_k_logits.astype(jnp.bfloat16), axis=-1
            ).astype(self.dtype)

            # ep moe norm_topk_prob=true
            top_k_weights = top_k_weights / jnp.sum(
                top_k_weights, axis=-1, keepdims=True
            )

            if hidden_states.ndim == 2:
                total_tokens = hidden_states.shape[0]
                batch_size, seq_len = 1, total_tokens
            else:
                batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
                total_tokens = batch_size * seq_len
            # Permute
            x, sorted_selected_experts, weights, group_sizes, selected_experts = (
                self._permute(hidden_states, top_k_indices, top_k_weights)
            )

            # EP Dispatch
            if self.expert_parallel_size > 1:
                x, local_group_sizes, selected_experts = (
                    self._expert_all_to_all_dispatch(
                        x, selected_experts, expert_shard_id
                    )
                )
            else:
                local_group_sizes = group_sizes

            # GMM
            intermediate_output = self._gmm_compute_with_sharded_weights(
                x,
                local_group_sizes,
                selected_experts,
                w0_weights,
                w1_weights,
                wo_weights,
            )

            # EP Combine
            if self.expert_parallel_size > 1:
                original_size = total_tokens * self.num_experts_per_tok
                intermediate_output = self._expert_all_to_all_collect(
                    intermediate_output, group_sizes, expert_shard_id, original_size
                )

            # Unpermute
            output = self._unpermute(
                intermediate_output,
                sorted_selected_experts,
                weights,
                batch_size,
                seq_len,
            )
            return output

        return shard_map(
            _internal_moe_computation,
            mesh=self.mesh,
            in_specs=(
                P(None),  # hidden_states
                P(None),  # router_logits
                P(("data", "tensor"), None, None),  # w0_weights
                P(("data", "tensor"), None, None),  # w1_weights
                P(("data", "tensor"), None, None),  # wo_weights
            ),
            out_specs=P(None),
            check_rep=False,
        )(inputs, router_logits, self.wi_0.value, self.wi_1.value, self.wo.value)

    def _gmm_compute_with_sharded_weights(
        self, x, local_group_sizes, selected_experts, w0_kernel, w1_kernel, wo_kernel
    ):
        if x.shape[0] == 0:
            empty_output = jnp.zeros(
                (0, wo_kernel.shape[-1]), dtype=x.dtype
            )  # (0, hidden_dim)
            return empty_output

        # gate
        layer_w0 = jax.lax.ragged_dot(
            lhs=x,
            rhs=w0_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
        )
        # up
        layer_w1 = jax.lax.ragged_dot(
            lhs=x,
            rhs=w1_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
        )

        # activation
        layer_act = jax.nn.silu(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)

        # down
        intermediate_output = jax.lax.ragged_dot(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
        )

        return intermediate_output

    def _single_device_forward(self, inputs, router_logits):
        top_k_logits, top_k_indices = jax.lax.top_k(
            router_logits, self.num_experts_per_tok
        )
        top_k_weights = jax.nn.softmax(
            top_k_logits.astype(jnp.float32), axis=-1
        ).astype(self.dtype)

        top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)

        return self._single_device_forward(inputs, top_k_indices, top_k_weights)

    def _single_device_forward(self, inputs, top_k_indices, top_k_weights):
        num_tokens = inputs.shape[0] * (inputs.shape[1] if inputs.ndim > 1 else 1)
        inputs_flat = inputs.reshape(num_tokens, -1)

        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        token_indices = jnp.arange(num_tokens)[:, None]

        top_k_indices_flat = top_k_indices.reshape(num_tokens, -1)
        top_k_weights_flat = top_k_weights.reshape(num_tokens, -1)

        expert_weights = expert_weights.at[token_indices, top_k_indices_flat].set(
            top_k_weights_flat
        )

        all_wi_0 = self.wi_0.value
        all_wi_1 = self.wi_1.value
        all_wo = self.wo.value

        layer_w0 = jnp.einsum("th,ehd->ted", inputs_flat, all_wi_0)
        layer_w1 = jnp.einsum("th,ehd->ted", inputs_flat, all_wi_1)

        activated = jax.nn.silu(layer_w0) * layer_w1
        expert_outputs = jnp.einsum("ted,edh->teh", activated, all_wo)
        final_output = jnp.einsum("te,teh->th", expert_weights, expert_outputs)

        return final_output.reshape(inputs.shape).astype(self.dtype)

    def _expert_all_to_all_dispatch(
        self, data, sorted_experts, expert_shard_id
    ):
        local_expert_size = self.experts_per_device

        # compute each token's expert shard
        divided_assignments = jnp.floor_divide(sorted_experts, local_expert_size)

        # mask
        belongs_to_this_shard = divided_assignments == expert_shard_id

        local_experts = jnp.where(
            belongs_to_this_shard,
            jnp.mod(sorted_experts, local_expert_size),
            local_expert_size,
        )

        valid_indices = jnp.nonzero(belongs_to_this_shard, size=data.shape[0])[0]
        num_valid_tokens = jnp.sum(belongs_to_this_shard)

        local_data = data[valid_indices]
        local_experts_extracted = local_experts[valid_indices]

        valid_expert_mask = jnp.arange(data.shape[0]) < num_valid_tokens
        valid_experts_for_bincount = jnp.where(
            valid_expert_mask, local_experts_extracted, local_expert_size
        )
        local_group_sizes = jnp.bincount(
            valid_experts_for_bincount, length=local_expert_size
        )

        return local_data, local_group_sizes, local_experts_extracted

    def _get_all_to_all_params(self, group_sizes, shard_id):
        input_offsets = jnp.zeros(self.expert_parallel_size, dtype=group_sizes.dtype)
        send_sizes = jnp.repeat(group_sizes[shard_id], self.expert_parallel_size)
        output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(group_sizes[:-1])))[shard_id]
        output_offsets = jnp.repeat(output_offset, self.expert_parallel_size)
        recv_sizes = group_sizes
        
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _expert_all_to_all_collect(
        self, data, global_group_sizes, expert_shard_id, target_size
    ):
        # Calculate the number of tokens to be handled by each device.
        reshaped_group_sizes = global_group_sizes.reshape(
            self.expert_parallel_size, self.experts_per_device
        )
        tokens_per_device = jnp.sum(reshaped_group_sizes, axis=1)

        # Get parameters for ragged_all_to_all
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            tokens_per_device, expert_shard_id
        )
        
        # Create output shape buffer
        output_shape = jnp.zeros((target_size, data.shape[1]), dtype=data.dtype)

        # Use ragged_all_to_all to gather data from all devices
        result = jax.lax.ragged_all_to_all(
            data,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=("data", "tensor")
        )

        return result

    def _permute(self, inputs, top_k_indices, top_k_weights):
        inputs_shape = inputs.shape

        if len(inputs_shape) == 2:
            inputs_2d = inputs
            bsz_times_seq_len = inputs_shape[0]
        else:
            bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
            inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))

        flatten_selected_experts = jnp.ravel(top_k_indices)
        sorted_selected_experts = jnp.argsort(flatten_selected_experts)
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok

        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(
            self.dtype
        )

        group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)

        expert_indices = jnp.arange(self.num_experts)
        sorted_experts = jnp.repeat(
            expert_indices,
            repeats=group_sizes,
            total_repeat_length=flatten_selected_experts.shape[0],
        )

        return (
            sorted_inputs,
            sorted_selected_experts,
            top_k_weights,
            group_sizes,
            sorted_experts,
        )

    def _unpermute(
        self, intermediate, sorted_selected_experts, weights, batch_size, seq_len
    ):
        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros(
                    (padding_size, intermediate.shape[1]), dtype=intermediate.dtype
                )
                intermediate = jnp.concatenate([intermediate, padding], axis=0)

        argsort_indices = jnp.argsort(sorted_selected_experts)
        unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)

        total_tokens = weights.shape[0] * weights.shape[1] // self.num_experts_per_tok

        reshaped_weights = jnp.reshape(
            weights, (total_tokens, self.num_experts_per_tok)
        )
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
