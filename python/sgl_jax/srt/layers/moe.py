from typing import Iterable, Optional, Sequence, Tuple, Union

import jax
from flax import nnx
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.debug_tracer import global_tracer, trace_function
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

        global_tracer.print(inputs, f"gate_input", f"moe_gate_layer_id_{self.layer_id}")

        kernel = jnp.asarray(self.kernel.value, self.dtype)
        output = jnp.dot(inputs, kernel)

        global_tracer.print(
            output, f"gate_raw_output", f"moe_gate_layer_id_{self.layer_id}"
        )

        if self.score_func:
            if self.score_func == "softmax":
                output = jax.nn.softmax(output)
            elif self.score_func == "sigmoid":
                output = jax.nn.sigmoid(output)
            elif self.score_func == "tanh":
                output = jax.nn.tanh(output)

            global_tracer.print(
                output, f"gate_after_score_func", f"moe_gate_layer_id_{self.layer_id}"
            )

        if self.use_bias and self.bias is not None:
            bias = jnp.asarray(self.bias.value, self.dtype)
            output += bias
            global_tracer.print(
                output, f"gate_after_bias", f"moe_gate_layer_id_{self.layer_id}"
            )

        global_tracer.print(
            output, f"gate_final_output", f"moe_gate_layer_id_{self.layer_id}"
        )

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

        # 在纯EP设置中，hidden维度不被切分，专家维度被切分
        # MoE的输入输出都应该是完整的hidden_size
        input_hidden_size = config.hidden_size
        output_hidden_size = config.hidden_size
        
        self.wi_0 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, input_hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wi_1 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, input_hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wo = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), expert_kernel_axes)(
                rngs.params(),
                (self.experts_per_device, intermediate_dim, output_hidden_size),
                weight_dtype,
            )
        )

        state = nnx.state(self)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(self, sharded_state)

        # Detect device capabilities for choosing communication strategy
        self.can_use_ragged, self.primary_device = self._detect_device_capabilities()
        print(f"[MOE] Layer {layer_id}: can_use_ragged={self.can_use_ragged}, device={self.primary_device}, ep_size={expert_parallel_size}")

    def _detect_device_capabilities(self):
        try:
            devices = jax.devices()
            is_cpu_only = all(device.platform == "cpu" for device in devices)
            can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

            device_types = [device.platform for device in devices]
            primary_device = device_types[0] if device_types else "unknown"

            global_tracer.print(
                jnp.array([is_cpu_only, can_use_ragged]),
                f"device_capabilities_cpu_ragged",
                f"moe_device_layer_id_{self.layer_id}",
            )
            print(f"can_use_ragged: {can_use_ragged}, primary_device: {primary_device}")
            return can_use_ragged, primary_device
        except Exception as e:
            return False, "cpu"

    @trace_function(stage="MOE_SPARSE_FORWARD", include_args=False, include_output=True)
    def __call__(self, inputs, router_logits=None):
        if router_logits is None:
            raise ValueError("router_logits is required for EPMoE")

        inputs = inputs.astype(self.dtype)
        total_tokens, hidden_dim = inputs.shape

        print(f"[MOE] Forward layer {self.layer_id}: tokens={total_tokens}, ep_size={self.expert_parallel_size}, use_ragged={self.can_use_ragged}")

        global_tracer.print(
            inputs, f"moe_input", f"moe_sparse_layer_id_{self.layer_id}"
        )
        global_tracer.print(
            router_logits, f"router_logits", f"moe_sparse_layer_id_{self.layer_id}"
        )

        if router_logits.shape[0] != total_tokens:
            raise ValueError(
                f"router_logits shape {router_logits.shape} doesn't match inputs shape {inputs.shape}"
            )

        if self.expert_parallel_size == 1:
            print(f"[MOE] Layer {self.layer_id}: Using single device forward")
            output = self._single_device_forward(inputs, router_logits)
        else:
            print(f"[MOE] Layer {self.layer_id}: Using expert parallel forward with shard_map")
            output = self._expert_parallel_forward_with_shard_map(inputs, router_logits)

        print(f"[MOE] Layer {self.layer_id}: Forward completed, output.shape={output.shape}")
        
        # 检查输出数据的健康状态
        output_sum = jnp.sum(output)
        output_has_nan = jnp.any(jnp.isnan(output))
        output_has_inf = jnp.any(jnp.isinf(output))
        output_min = jnp.min(output)
        output_max = jnp.max(output)
        
        jax.debug.print(
            "[MOE] Layer {layer_id}: Output health check - sum={sum}, has_nan={has_nan}, has_inf={has_inf}, min={min}, max={max}",
            layer_id=self.layer_id,
            sum=output_sum,
            has_nan=output_has_nan,
            has_inf=output_has_inf,
            min=output_min,
            max=output_max
        )
        
        global_tracer.print(
            output, f"moe_final_output", f"moe_sparse_layer_id_{self.layer_id}"
        )
        print(f"[MOE] Layer {self.layer_id}: About to return output")
        return output

    def _expert_parallel_forward_with_shard_map(self, inputs, router_logits):
        print(f"[SHARD_MAP] Layer {self.layer_id}: Starting shard_map forward")
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
                        x, group_sizes, selected_experts, expert_shard_id
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
            output = self.unpermute(
                intermediate_output,
                sorted_selected_experts,
                weights,
                batch_size,
                seq_len,
            )
            return output

        print(f"[SHARD_MAP] Layer {self.layer_id}: About to call shard_map")
        result = shard_map(
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
        print(f"[SHARD_MAP] Layer {self.layer_id}: shard_map completed, result.shape={result.shape}")
        return result

    def _gmm_compute_with_sharded_weights(
        self, x, local_group_sizes, selected_experts, w0_kernel, w1_kernel, wo_kernel
    ):
        global_tracer.print(
            x, f"gmm_sharded_input_x", f"moe_compute_layer_id_{self.layer_id}"
        )
        global_tracer.print(
            w0_kernel,
            f"gmm_sharded_w0_kernel_shape",
            f"moe_compute_layer_id_{self.layer_id}",
        )
        
        # 调试维度信息
        jax.debug.print(
            "[GMM_DEBUG] Layer {layer_id}: x.shape={x_shape}, w0.shape={w0_shape}",
            layer_id=self.layer_id,
            x_shape=x.shape,
            w0_shape=w0_kernel.shape
        )
        global_tracer.print(
            local_group_sizes,
            f"gmm_sharded_local_group_sizes",
            f"moe_compute_layer_id_{self.layer_id}",
        )

        if x.shape[0] == 0:
            empty_output = jnp.zeros(
                (0, wo_kernel.shape[-1]), dtype=x.dtype
            )  # (0, hidden_dim)
            global_tracer.print(
                empty_output,
                f"gmm_sharded_empty_output",
                f"moe_compute_layer_id_{self.layer_id}",
            )
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

        global_tracer.print(
            intermediate_output,
            f"gmm_sharded_final_output",
            f"moe_compute_layer_id_{self.layer_id}",
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

        return self._single_device_forward_impl(inputs, top_k_indices, top_k_weights)

    def _single_device_forward_impl(self, inputs, top_k_indices, top_k_weights):
        global_tracer.print(
            inputs, f"moe_local_input", f"moe_compute_layer_id_{self.layer_id}"
        )

        num_tokens = inputs.shape[0] * (inputs.shape[1] if inputs.ndim > 1 else 1)
        inputs_flat = inputs.reshape(num_tokens, -1)

        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        token_indices = jnp.arange(num_tokens)[:, None]

        top_k_indices_flat = top_k_indices.reshape(num_tokens, -1)
        top_k_weights_flat = top_k_weights.reshape(num_tokens, -1)

        expert_weights = expert_weights.at[token_indices, top_k_indices_flat].set(
            top_k_weights_flat
        )

        global_tracer.print(
            expert_weights,
            f"expert_weights_matrix",
            f"moe_compute_layer_id_{self.layer_id}",
        )

        all_wi_0 = self.wi_0.value
        all_wi_1 = self.wi_1.value
        all_wo = self.wo.value

        layer_w0 = jnp.einsum("th,ehd->ted", inputs_flat, all_wi_0)
        layer_w1 = jnp.einsum("th,ehd->ted", inputs_flat, all_wi_1)

        global_tracer.print(
            layer_w0, f"layer_w0_output", f"moe_compute_layer_id_{self.layer_id}"
        )
        global_tracer.print(
            layer_w1, f"layer_w1_output", f"moe_compute_layer_id_{self.layer_id}"
        )

        activated = jax.nn.silu(layer_w0) * layer_w1
        expert_outputs = jnp.einsum("ted,edh->teh", activated, all_wo)
        final_output = jnp.einsum("te,teh->th", expert_weights, expert_outputs)

        global_tracer.print(
            final_output,
            f"moe_local_final_output",
            f"moe_compute_layer_id_{self.layer_id}",
        )
        return final_output.reshape(inputs.shape).astype(self.dtype)

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
            total_repeat_length=bsz_times_seq_len * self.num_experts_per_tok,
        )

        return (
            sorted_inputs,
            sorted_selected_experts,
            top_k_weights,
            group_sizes,
            sorted_experts,
        )


    def _expert_all_to_all_dispatch(
        self, data, global_group_sizes, sorted_experts, expert_shard_id
    ):
        return self._simple_dispatch(
            data, global_group_sizes, sorted_experts, expert_shard_id
        )

    def _simple_dispatch(
        self, data, global_group_sizes, sorted_experts, expert_shard_id
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

        global_tracer.print(
            local_data, f"cpu_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}"
        )
        global_tracer.print(
            local_group_sizes,
            f"cpu_dispatch_group_sizes",
            f"moe_dispatch_layer_id_{self.layer_id}",
        )

        return local_data, local_group_sizes, local_experts_extracted

    def _expert_all_to_all_collect(
        self, data, global_group_sizes, expert_shard_id, target_size
    ):
        print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: START - data.shape={data.shape}, target_size={target_size}")
        
        # 按照 MaxText lines 710-721 的纯 EP collect 逻辑
        local_expert_size = self.experts_per_device
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(-1, local_expert_size), axis=1)
        
        # 使用 MaxText 的 get_all_to_all_params
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            reshaped_group_sizes, expert_shard_id, self.expert_parallel_size, is_batch_sharded=False
        )
        
        # 创建输出形状（MaxText lines 687-690）
        output_shape = jnp.zeros((target_size, data.shape[1]), dtype=data.dtype)
        
        # 执行 ragged_all_to_all（MaxText lines 713-721）
        intermediate_output = jax.lax.ragged_all_to_all(
            data,
            output_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=("data", "tensor"),
        )
        
        print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: END - result.shape={intermediate_output.shape}")
        return intermediate_output

    def _cpu_simple_collect(
        self, data, global_group_sizes, expert_shard_id, target_size
    ):
        """
        Gathers variable-sized data from all expert devices into a single, correctly
        ordered tensor using a JIT-compatible scatter-and-sum pattern.
        """
        # Calculate the number of tokens to be handled by each device.
        reshaped_group_sizes = global_group_sizes.reshape(
            self.expert_parallel_size, self.experts_per_device
        )
        tokens_per_device = jnp.sum(reshaped_group_sizes, axis=1)

        # Calculate the start and end indices for this device's data in the global buffer.
        cumsum = jnp.cumsum(tokens_per_device)
        start_indices = jnp.concatenate(
            [jnp.array([0], dtype=tokens_per_device.dtype), cumsum]
        )
        my_start_index = start_indices[expert_shard_id]
        my_end_index = start_indices[expert_shard_id + 1]

        # JIT-safe scatter operation.
        # This block constructs a buffer of the full `target_size` for the local device,
        # with this device's data placed in the correct slice. This avoids creating
        # intermediate tensors with dynamic shapes, which is required for JIT compilation.
        output_indices = jnp.arange(target_size)
        source_indices = output_indices - my_start_index

        # Create a mask for the slice this device is responsible for.
        mask = (output_indices >= my_start_index) & (output_indices < my_end_index)

        # Gather from source `data`, using a safe index (0) for out-of-bounds access.
        # The mask ensures these gathered-but-invalid values are discarded.
        safe_source_indices = jnp.where(mask, source_indices, 0)
        gathered_data = data[safe_source_indices]

        # Place the gathered data into the buffer using the mask.
        local_result_buffer = jnp.where(mask[:, None], gathered_data, 0.0)

        # Sum the buffers from all devices. Since each buffer is zero outside its
        # assigned slice, this sum is equivalent to a concatenation.
        result = jax.lax.psum(local_result_buffer, axis_name=("data", "tensor"))

        return result

    def _ragged_all_to_all_collect(
        self, data, global_group_sizes, expert_shard_id, target_size, local_sorted_indices=None
    ):
        """
        完全按照 MaxText 的 collect 实现 (lines 692-705)
        """
        local_expert_size = self.experts_per_device
        
        # 步骤1：局部反排列（MaxText line 693） 
        if local_sorted_indices is not None:
            # locally unpermute back to the original order
            local_output = jnp.take(data, indices=jnp.argsort(local_sorted_indices), axis=0)
        else:
            local_output = data
        
        # 步骤2：重新构建 all_shards_group_sizes（与 dispatch 阶段一致）
        reshaped_group_sizes = jnp.sum(global_group_sizes.reshape(-1, local_expert_size), axis=1)
        
        # 与 dispatch 阶段一样，重新计算 all_shards_group_sizes
        all_shards_group_sizes = jax.lax.all_gather(
            reshaped_group_sizes, axis_name=("data", "tensor"), axis=0
        )
        
        # 步骤3：纯 EP 使用 reshaped_group_sizes（与 dispatch 保持一致）
        input_offsets, send_sizes, output_offsets, recv_sizes = self._get_all_to_all_params(
            reshaped_group_sizes, expert_shard_id, self.expert_parallel_size, is_batch_sharded=False
        )
        
        # 步骤4：创建输出缓冲区（MaxText lines 687-690）
        # 验证目标大小，确保与预期一致（MaxText line 685-686）
        original_inputs_first_dim = target_size  # 这应该等于 batch_size * sequence_length * self.num_experts_per_tok
        
        output_buffer = jnp.zeros(
            (target_size, data.shape[1]),  # 保持原始hidden维度
            dtype=data.dtype,
        )
        
        # 步骤5：执行 ragged_all_to_all（MaxText lines 697-705）
        intermediate_output = jax.lax.ragged_all_to_all(
            local_output,
            output_buffer,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=("data", "tensor"),
        )
        
        return intermediate_output
        
    def _get_tensor_parallelism_size(self):
        """
        获取tensor parallelism size
        在纯EP设置中，没有独立的tensor parallelism，所以返回1
        因为EP使用的是 ("data", "tensor") 组合，不是独立的TP
        """
        # 在纯EP设置中，tensor轴是EP的一部分，不是独立的TP
        return 1

    def _get_all_to_all_params(self, all_shards_group_sizes, shard_id, num_expert_parallelism, is_batch_sharded=True):
        """
        完全按照 MaxText 的 get_all_to_all_params 实现
        """
        def transform_array(input_array, shard_id, strategy, is_batch_sharded):
            """This function transforms the input array based on the specified strategy,
            preparing it for the usage with `ragged_all_to_all` API. The transformation
            determines how data is sent and received between shards.
            """
            if is_batch_sharded:
                if strategy == "INPUT_OFFSET":
                    # Index of input array for the send
                    local_array = input_array[shard_id]
                    return jnp.concatenate((jnp.array([0]), jnp.cumsum(local_array)[:-1]))
                elif strategy == "SEND_SIZE":
                    # Size of input array for the send
                    return input_array[shard_id]
                elif strategy == "OUTPUT_OFFSET":
                    # Received index in the target output
                    zero_row = jnp.zeros((1,) + input_array.shape[1:], dtype=input_array.dtype)
                    array_with_zeros = jnp.concatenate((zero_row, input_array), axis=0)
                    cumulated_array = jnp.cumsum(array_with_zeros, axis=0, dtype=input_array.dtype)
                    return cumulated_array[shard_id]
                elif strategy == "RECV_SIZE":
                    # Received size in the target output
                    return input_array[:, shard_id]
                else:
                    raise ValueError(f"Unknown transform array strategy: {strategy}")
            
            # If the batch is unsharded then we send the same data slice to all other shards.
            # We also assume each shard will have the local processed inputs sorted to start from index 0.
            # Finally, len(input_array.shape) == 1 since there is only one batch shard.
            else:
                if strategy == "INPUT_OFFSET":
                    # The data on each shard always starts at 0.
                    return jnp.zeros(num_expert_parallelism, dtype=input_array.dtype)
                elif strategy == "SEND_SIZE":
                    # The send amount is always the amount of data the current expert shard needs to process.
                    return jnp.repeat(input_array[shard_id], num_expert_parallelism)
                elif strategy == "OUTPUT_OFFSET":
                    # The offset in each shard will just be the start of the group which that shard is
                    # responsible for.
                    output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(input_array[:-1])))[shard_id]
                    return jnp.repeat(output_offset, num_expert_parallelism)
                # The amount that each shard receives from all other shards is equivalent to the group sizes
                # (aka input_array).
                elif strategy == "RECV_SIZE":
                    # Received size in the target output
                    return input_array
                else:
                    raise ValueError(f"Unknown transform array strategy: {strategy}")

        input_offsets = transform_array(all_shards_group_sizes, shard_id, "INPUT_OFFSET", is_batch_sharded)
        send_sizes = transform_array(all_shards_group_sizes, shard_id, "SEND_SIZE", is_batch_sharded)
        output_offsets = transform_array(all_shards_group_sizes, shard_id, "OUTPUT_OFFSET", is_batch_sharded)
        recv_sizes = transform_array(all_shards_group_sizes, shard_id, "RECV_SIZE", is_batch_sharded)
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _get_ragged_all_to_all_params(self, reshaped_group_sizes, expert_shard_id):
        """
        之前能工作版本的参数计算方法
        """
        # 简单的参数计算逻辑，基于之前能工作的版本
        input_offsets = jnp.zeros(self.expert_parallel_size, dtype=jnp.int32)
        send_sizes = jnp.repeat(reshaped_group_sizes[expert_shard_id], self.expert_parallel_size)
        
        cumsum_sizes = jnp.cumsum(reshaped_group_sizes)
        output_offsets = jnp.concatenate([jnp.array([0]), cumsum_sizes[:-1]])
        recv_sizes = reshaped_group_sizes
        
        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _local_permute_with_offset(
        self, inputs, global_group_sizes, local_expert_size, shard_index, global_sorted_experts
    ):
        """
        基于 MaxText local_permute 的 is_offset=True 逻辑，使用 JIT 兼容语法
        """
        # 步骤1：使用 dynamic_slice_in_dim 提取本地专家的 group sizes（JIT 兼容）
        # 注意：global_group_sizes 需要是 2D 形状 [num_batch_shards, num_experts]
        # 但我们的是 1D，所以先添加维度
        global_group_sizes_2d = global_group_sizes[None, :]  # [1, num_experts]
        
        # Slice the count of local expert IDs in each batch shard.
        # all_shard_local_sizes.shape: [expert_shard, local_expert_size] 
        all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes_2d, shard_index * local_expert_size, local_expert_size, axis=1
        )        
        # Total count of the local expert IDs is the sum of the counts across all batch shards,
        # since all batch shards will send their contributions to the current expert shard.
        local_group_size = jnp.sum(all_shard_local_sizes, axis=0)
        
        # 步骤2：is_offset=True 逻辑 - 筛选属于当前 shard 的数据
        # 判断每个 token 属于哪个 expert shard
        divided_assignments = jnp.floor_divide(global_sorted_experts, local_expert_size)
        
        # 筛选属于当前 shard 的数据，其他数据标记为 local_expert_size（无效）
        expert_indices = jnp.where(
            divided_assignments == shard_index, 
            jnp.mod(global_sorted_experts, local_expert_size), 
            local_expert_size  # 标记为无效
        )
        
        # 步骤3：重新排序（无效数据会排到最后）
        sorted_indices = jnp.argsort(expert_indices)
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = expert_indices[sorted_indices]
        
        return sorted_inputs, local_group_size, sorted_experts_ids, sorted_indices

    def unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, sequence_length):
        """Unpermute tokens to original order and combine weights."""

        unsort_intermediate = jnp.take(intermediate, indices=jnp.argsort(sorted_selected_experts), axis=0)
        reshaped_weights = jnp.reshape(weights, (-1, self.num_experts_per_tok))
        reshaped_intermediate = jnp.reshape(
            unsort_intermediate,
            (reshaped_weights.shape[0], self.num_experts_per_tok, -1),
        )
        with jax.named_scope("weight_sum"):
            output = jnp.einsum(
                "BKE,BK -> BE",
                reshaped_intermediate.astype(jnp.float32),
                reshaped_weights.astype(jnp.float32),
            )
        return output.reshape(batch_size, sequence_length, -1).astype(self.dtype)

