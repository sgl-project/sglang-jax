import jax
from flax import nnx
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm


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
        logits = hidden_states.astype(self.weight_dtype) @ self.kernel

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
        layer_id: int = 0,
    ):
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.mesh = mesh
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )
        world_size = (
            self.mesh.shape.get("data", 1)
            * mesh.shape.get("tensor", 1)
            * mesh.shape.get("expert", 1)
        )
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = num_experts // self.ep_size
        print("tp_size: %d, ep_size: %d", self.tp_size, self.ep_size)

        # if self.tp_size > 1:
        wi_kernel_axes = ("tensor", None, "data")
        wo_kernel_axes = ("tensor", "data", None)
        # else:
        #     wi_kernel_axes = (("data", "tensor"), None, None)
        #     wo_kernel_axes = (("data", "tensor"), None, None)

        self.wi_0 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), wi_kernel_axes)(
                jax.random.PRNGKey(0),
                (self.experts_per_device, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wi_1 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), wi_kernel_axes)(
                jax.random.PRNGKey(0),
                (self.experts_per_device, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wo = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), wo_kernel_axes)(
                jax.random.PRNGKey(0),
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
        except Exception as _:
            return False, "cpu"

    def __call__(self, hidden_states, topk_weights, topk_ids):
        return shard_map(
            self._forward,
            mesh=self.mesh,
            in_specs=(
                P(None),  # hidden_states
                P(None),  # topk_weights
                P(None),  # topk_ids
                P("tensor", None, "data"),  # w0_weights
                P("tensor", None, "data"),  # w1_weights
                P("tensor", "data", None),  # wo_weights
            ),
            out_specs=P(None),
            check_rep=False,
        )(
            hidden_states,
            topk_weights,
            topk_ids,
            self.wi_0.value,
            self.wi_1.value,
            self.wo.value,
        )

    def _forward(self, hidden_states, topk_weights, topk_ids, w0_weights, w1_weights, wo_weights):
        tensor_index = jax.lax.axis_index("tensor")
        expert_shard_id = tensor_index

        if hidden_states.ndim == 2:
            total_tokens = hidden_states.shape[0]
            batch_size, seq_len = 1, total_tokens
        else:
            batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            total_tokens = batch_size * seq_len
        # Permute
        x, sorted_selected_experts, weights, group_sizes, selected_experts = self._permute(
            hidden_states, topk_ids, topk_weights
        )

        # EP Dispatch
        if self.ep_size > 1:
            x, local_group_sizes, selected_experts = self._expert_all_to_all_dispatch(
                x, selected_experts, expert_shard_id
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
        if self.ep_size > 1:
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

    def _gmm_compute_with_sharded_weights(
        self, x, local_group_sizes, selected_experts, w0_kernel, w1_kernel, wo_kernel
    ):
        if x.shape[0] == 0:
            empty_output = jnp.zeros((0, wo_kernel.shape[-1]), dtype=x.dtype)  # (0, hidden_dim)
            return empty_output

        m, k = x.shape[0], x.shape[1]
        n_gate = w0_kernel.shape[2]
        n_down = wo_kernel.shape[2]

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
        # gate
        layer_w0 = gmm(
            lhs=x,
            rhs=w0_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_gate,
        )
        # up
        layer_w1 = gmm(
            lhs=x,
            rhs=w1_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_gate,
        )

        # activation
        layer_act = jax.nn.silu(layer_w0)
        intermediate_layer = jnp.multiply(layer_act, layer_w1)

        # down
        intermediate_output = gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            group_sizes=local_group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_down,
        )

        return intermediate_output

    def _expert_all_to_all_dispatch(self, data, sorted_experts, expert_shard_id):
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
        local_group_sizes = jnp.bincount(valid_experts_for_bincount, length=local_expert_size)

        return local_data, local_group_sizes, local_experts_extracted

    def _get_all_to_all_params(self, group_sizes, shard_id):
        input_offsets = jnp.zeros(self.ep_size, dtype=group_sizes.dtype)
        send_sizes = jnp.repeat(group_sizes[shard_id], self.ep_size)
        output_offset = jnp.concatenate((jnp.array([0]), jnp.cumsum(group_sizes[:-1])))[shard_id]
        output_offsets = jnp.repeat(output_offset, self.ep_size)
        recv_sizes = group_sizes

        return input_offsets, send_sizes, output_offsets, recv_sizes

    def _expert_all_to_all_collect(self, data, global_group_sizes, expert_shard_id, target_size):
        # Calculate the number of tokens to be handled by each device.
        reshaped_group_sizes = global_group_sizes.reshape(self.ep_size, self.experts_per_device)
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
            axis_name=("tensor"),
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
        sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
        sorted_indices = sorted_selected_experts // self.num_experts_per_tok

        sorted_inputs = jnp.take(inputs_2d, indices=sorted_indices, axis=0).astype(self.dtype)

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


class FusedMoE(nnx.Module):
    def __init__(
        self,
        config,
        num_experts: int,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        layer_id: int = 0,
        *,
        mesh: Mesh,
    ):
        self.config = config
        self.num_experts = num_experts
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id

        self.tp_size = mesh.shape.get("data") * mesh.shape.get("tensor")

        self.mesh = mesh

        self.experts_per_device = num_experts

        self.wi_sharding = (None, None, ("data", "tensor"))
        self.wo_sharding = (None, ("data", "tensor"), None)

        self.wi_0 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_sharding)(
                jax.random.PRNGKey(0),
                (num_experts, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wi_1 = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wi_sharding)(
                jax.random.PRNGKey(0),
                (num_experts, config.hidden_size, intermediate_dim),
                weight_dtype,
            )
        )

        self.wo = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), self.wo_sharding)(
                jax.random.PRNGKey(0),
                (num_experts, intermediate_dim, config.hidden_size),
                weight_dtype,
            )
        )

    def __call__(self, hidden_states, topk_weights, topk_ids):
        inputs = hidden_states.astype(self.dtype)

        if inputs.ndim == 2:
            num_tokens = inputs.shape[0]
            inputs_flat = inputs
        else:
            num_tokens = inputs.shape[0] * inputs.shape[1]
            inputs_flat = inputs.reshape(num_tokens, -1)

        expert_weights = jnp.zeros((num_tokens, self.num_experts), dtype=self.dtype)
        token_indices = jnp.arange(num_tokens)[:, None]

        top_k_indices_flat = topk_ids.reshape(num_tokens, -1)
        top_k_weights_flat = topk_weights.reshape(num_tokens, -1)

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
