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
        global_tracer.print(
            output, f"moe_final_output", f"moe_sparse_layer_id_{self.layer_id}"
        )
        print(f"[MOE] Layer {self.layer_id}: About to return output")
        return output

    @nnx.jit
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
            output = self._unpermute(
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
        print(f"[DISPATCH] Layer {self.layer_id} shard {expert_shard_id}: START - data.shape={data.shape}, use_ragged={self.can_use_ragged}")
        
        if self.can_use_ragged:
            print(f"[DISPATCH] Layer {self.layer_id} shard {expert_shard_id}: Using ragged_all_to_all")
            result = self._ragged_all_to_all_dispatch(
                data, global_group_sizes, sorted_experts, expert_shard_id
            )
        else:
            print(f"[DISPATCH] Layer {self.layer_id} shard {expert_shard_id}: Using simple dispatch")
            result = self._simple_dispatch(
                data, global_group_sizes, sorted_experts, expert_shard_id
            )
            
        print(f"[DISPATCH] Layer {self.layer_id} shard {expert_shard_id}: END - result.shape={result[0].shape}")
        return result

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

    def _ragged_all_to_all_dispatch(
        self, data, global_group_sizes, sorted_experts, expert_shard_id
    ):
        """
        正确的 ragged_all_to_all dispatch 流程：
        1. 根据 sorted_experts 确定每个 token 应该发送到哪个 shard
        2. 计算每个 shard 应该发送/接收多少数据
        3. 使用 ragged_all_to_all 进行通信
        4. 对接收到的数据进行本地排序
        """
        local_expert_size = self.experts_per_device
        
        # 步骤1：根据 sorted_experts 计算每个 token 的目标 shard
        # sorted_experts 包含每个 token 的 expert ID (0 到 num_experts-1)
        target_shards = jnp.floor_divide(sorted_experts, local_expert_size).astype(jnp.int32)
        
        # 步骤2：计算发送到每个 shard 的 token 数量
        send_counts = jnp.bincount(
            target_shards, 
            length=self.expert_parallel_size
        )
        
        # 步骤3：计算接收数量 - 这需要通过 all_gather 获取所有 shard 的 send_counts
        all_send_counts = jax.lax.all_gather(
            send_counts, 
            axis_name=("data", "tensor"),
            axis=0
        )  # shape: (expert_parallel_size, expert_parallel_size)
        
        # 每个 shard 接收的数量是所有其他 shard 发送给它的总和
        recv_counts = all_send_counts[:, expert_shard_id]
        
        # 步骤4：计算 ragged_all_to_all 的参数
        send_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(send_counts)[:-1]])
        recv_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(recv_counts)[:-1]])
        
        # 步骤5：准备发送数据 - 按目标 shard 排序
        sort_indices = jnp.argsort(target_shards)
        sorted_data = jnp.take(data, indices=sort_indices, axis=0)
        sorted_expert_ids = jnp.take(sorted_experts, indices=sort_indices)
        
        # 步骤6：计算输出缓冲区大小
        total_recv_size = jnp.sum(recv_counts)
        max_possible_recv = data.shape[0] * self.expert_parallel_size
        
        # 使用静态上界避免动态 shape 问题
        output_buffer = jnp.zeros((max_possible_recv, data.shape[1]), dtype=data.dtype)
        
        print(f"[RAGGED_DISPATCH] send_counts={send_counts}, recv_counts={recv_counts}")
        print(f"[RAGGED_DISPATCH] total_recv_size={total_recv_size}, max_buffer={max_possible_recv}")
        
        # 步骤7：执行 ragged_all_to_all 通信
        communicated_data = jax.lax.ragged_all_to_all(
            sorted_data,
            output_buffer,
            send_offsets,
            send_counts,
            recv_offsets,
            recv_counts,
            axis_name=("data", "tensor"),
        )
        
        # 步骤8：使用 dynamic_slice 代替动态索引来截取有效数据
        valid_received_data = jax.lax.dynamic_slice(
            communicated_data,
            start_indices=[0, 0],
            slice_sizes=[total_recv_size, communicated_data.shape[1]]
        )
        
        # 步骤9：计算本地 expert 分组
        local_group_sizes = jax.lax.dynamic_slice(
            global_group_sizes,
            start_indices=[expert_shard_id * local_expert_size],
            slice_sizes=[local_expert_size]
        )
        
        # 步骤10：使用 _local_permute_for_ragged 来处理 expert 分配
        # 这个函数已经正确处理了 mask 和动态大小问题
        _, final_local_group_sizes, local_expert_assignments = self._local_permute_for_ragged(
            valid_received_data, 
            global_group_sizes, 
            local_expert_size, 
            expert_shard_id
        )
        
        global_tracer.print(
            valid_received_data, f"ragged_dispatch_output", f"moe_dispatch_layer_id_{self.layer_id}"
        )
        
        return valid_received_data, final_local_group_sizes, local_expert_assignments

    def _expert_all_to_all_collect(
        self, data, global_group_sizes, expert_shard_id, target_size
    ):
        print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: START - data.shape={data.shape}, target_size={target_size}, use_ragged={self.can_use_ragged}")
        
        if self.can_use_ragged:
            print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: Using ragged_all_to_all")
            result = self._ragged_all_to_all_collect(
                data, global_group_sizes, expert_shard_id, target_size
            )
        else:
            print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: Using simple collect")
            result = self._cpu_simple_collect(
                data, global_group_sizes, expert_shard_id, target_size
            )
            
        print(f"[COLLECT] Layer {self.layer_id} shard {expert_shard_id}: END - result.shape={result.shape}")
        return result

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
        self, data, global_group_sizes, expert_shard_id, target_size
    ):
        """TPU/GPU: Use ragged_all_to_all for collection - inverse of dispatch"""
        # For now, let's use the CPU simple collect to avoid the complex ragged_all_to_all issues
        # We can optimize this later once the basic functionality works
        return self._cpu_simple_collect(
            data, global_group_sizes, expert_shard_id, target_size
        )

    def _get_ragged_all_to_all_params(self, group_sizes, shard_id):
        input_offsets = jnp.zeros(self.expert_parallel_size, dtype=jnp.int32)
        
        # Use dynamic indexing for JIT compatibility
        shard_group_size = jax.lax.dynamic_slice(
            group_sizes, start_indices=[shard_id], slice_sizes=[1]
        )[0]
        send_sizes = jnp.repeat(shard_group_size, self.expert_parallel_size)

        # Calculate cumulative offsets using dynamic indexing
        cumsum_sizes = jnp.cumsum(jnp.concatenate([jnp.array([0]), group_sizes]))
        output_offset = jax.lax.dynamic_slice(
            cumsum_sizes, start_indices=[shard_id], slice_sizes=[1]
        )[0]
        output_offsets = jnp.repeat(output_offset, self.expert_parallel_size)

        recv_sizes = group_sizes

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _local_permute_for_ragged(
        self, inputs, global_group_sizes, local_expert_size, shard_index
    ):
        # Use dynamic_slice instead of dynamic indexing for JIT compatibility
        start_index = shard_index * local_expert_size
        local_group_sizes = jax.lax.dynamic_slice(
            global_group_sizes, 
            start_indices=[start_index], 
            slice_sizes=[local_expert_size]
        )

        # Simplified approach: use all available input data and let mask handle validation
        # This avoids dynamic slicing issues while maintaining correctness
        max_possible_tokens = inputs.shape[0]
        
        # Create expert indices with fixed size
        expert_indices = jnp.repeat(
            jnp.arange(local_expert_size),
            local_group_sizes,
            total_repeat_length=max_possible_tokens,
        )
        
        # Calculate actual number of valid tokens
        total_local_tokens = jnp.sum(local_group_sizes)
        
        # Create a mask for valid tokens instead of slicing
        valid_mask = jnp.arange(max_possible_tokens) < total_local_tokens
        
        # Apply mask to get valid expert indices (fill invalid with -1)
        valid_expert_indices = jnp.where(valid_mask, expert_indices, -1)
        
        # Sort all indices (invalid ones will go to the end due to -1)
        sorted_indices = jnp.argsort(valid_expert_indices)
        
        # Take from the sorted indices, the valid ones come first
        sorted_inputs = jnp.take(inputs, indices=sorted_indices, axis=0)
        sorted_experts_ids = jnp.take(expert_indices, indices=sorted_indices)
        
        # Only return the valid portion by using the known total_local_tokens
        # This is safe because we know total_local_tokens <= max_possible_tokens
        return sorted_inputs, local_group_sizes, sorted_experts_ids

    def _unpermute(
        self, intermediate, sorted_selected_experts, weights, batch_size, seq_len
    ):
        global_tracer.print(
            intermediate, f"unpermute_input", f"moe_combine_layer_id_{self.layer_id}"
        )
        global_tracer.print(
            sorted_selected_experts,
            f"unpermute_sorted_experts",
            f"moe_combine_layer_id_{self.layer_id}",
        )
        global_tracer.print(
            weights, f"unpermute_weights", f"moe_combine_layer_id_{self.layer_id}"
        )

        expected_tokens = sorted_selected_experts.shape[0]
        actual_tokens = intermediate.shape[0]

        global_tracer.print(
            jnp.array([actual_tokens, expected_tokens]),
            f"unpermute_token_count_check",
            f"moe_combine_layer_id_{self.layer_id}",
        )

        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                intermediate = intermediate[:expected_tokens]
                global_tracer.print(
                    jnp.array([1, actual_tokens, expected_tokens]),
                    f"unpermute_truncated",
                    f"moe_combine_layer_id_{self.layer_id}",
                )
            else:
                padding_size = expected_tokens - actual_tokens
                padding = jnp.zeros(
                    (padding_size, intermediate.shape[1]), dtype=intermediate.dtype
                )
                intermediate = jnp.concatenate([intermediate, padding], axis=0)
                global_tracer.print(
                    jnp.array([2, actual_tokens, expected_tokens, padding_size]),
                    f"unpermute_padded",
                    f"moe_combine_layer_id_{self.layer_id}",
                )

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

        global_tracer.print(
            reshaped_weights,
            f"unpermute_reshaped_weights",
            f"moe_combine_layer_id_{self.layer_id}",
        )
        global_tracer.print(
            reshaped_intermediate,
            f"unpermute_reshaped_intermediate",
            f"moe_combine_layer_id_{self.layer_id}",
        )

        intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
        weights_fp32 = reshaped_weights.astype(jnp.float32)

        output = jnp.einsum(
            "BKE,BK -> BE",
            intermediate_fp32,
            weights_fp32,
        )

        global_tracer.print(
            output, f"unpermute_einsum_output", f"moe_combine_layer_id_{self.layer_id}"
        )

        if len(weights.shape) == 2:
            final_output = output.astype(self.dtype)
        else:
            final_output = output.reshape(batch_size, seq_len, -1).astype(self.dtype)

        global_tracer.print(
            final_output,
            f"unpermute_final_output",
            f"moe_combine_layer_id_{self.layer_id}",
        )
        return final_output
