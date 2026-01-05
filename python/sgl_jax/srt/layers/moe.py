import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.utils.profiling_utils import named_scope


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

    @named_scope
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

    @named_scope
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
    ):
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.original_mesh = mesh
        self.mesh = mesh
        self.activation = activation
        self.hidden_size = hidden_size
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
            self.wi_0 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, self.hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wi_1 = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, self.hidden_size, intermediate_dim),
                    dtype=weight_dtype,
                    out_sharding=P("expert", None, "tensor"),
                )
            )

            self.wo = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    (num_experts, intermediate_dim, self.hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P("expert", "tensor", None),
                )
            )

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

    @named_scope
    def __call__(self, hidden_states, topk_weights, topk_ids) -> jax.Array:
        with jax.sharding.use_abstract_mesh(self.updated_mesh):
            hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_reshard = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_reshard = jax.sharding.reshard(topk_ids, P(None))

            result = shard_map(
                self._forward,
                mesh=self.moe_mesh,
                in_specs=(
                    P(None),
                    P(None),
                    P(None),
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
            )

        output_pspec = P(*([None] * (result.ndim)))
        return jax.sharding.reshard(
            result, jax.sharding.NamedSharding(self.original_mesh, output_pspec)
        )

    def _forward(self, hidden_states, topk_weights, topk_ids, w0_weights, w1_weights, wo_weights):
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
        )

        output = self._unpermute(
            intermediate_output,
            sorted_selected_experts,
            weights,
            batch_size,
            seq_len,
        )
        # Combine across expert shards after unpermute to reduce EP communication
        # volume from O(T*K*H) to O(T*H).
        if self.ep_size > 1:
            output = self._combine(output)
        return output

    def _gmm_compute(self, x, group_sizes, w0_kernel, w1_kernel, wo_kernel, group_offset):
        if x.shape[0] == 0:
            empty_output = jnp.zeros((0, wo_kernel.shape[-1]), dtype=x.dtype)
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

        group_sizes = group_sizes.astype(jnp.int32)

        layer_w0 = gmm(
            lhs=x,
            rhs=w0_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_gate,
            group_offset=group_offset,
        )

        layer_w1 = gmm(
            lhs=x,
            rhs=w1_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_gate,
            group_offset=group_offset,
        )

        if self.activation == "silu":
            layer_act = jax.nn.silu(layer_w0)
        elif self.activation == "gelu":
            layer_act = jax.nn.gelu(layer_w0)
        else:
            raise ValueError(f"Unsupported activation function {self.activation}")
        intermediate_layer = jnp.multiply(layer_act, layer_w1)

        intermediate_output = gmm(
            lhs=intermediate_layer,
            rhs=wo_kernel,
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            tiling=tiling_down,
            group_offset=group_offset,
        )

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
        renormalize_topk_logits: bool = False,
        a2a_only: bool = True,
        no_comm: bool = False,
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
        self.renormalize_topk_logits = renormalize_topk_logits
        self.a2a_only = a2a_only
        self.no_comm = no_comm
        self.mesh = mesh

        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        router_logits: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
        a2a_only: bool = False,
        no_comm: bool = False,
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

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            gating_output=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize_topk_logits=self.renormalize_topk_logits,
            act_fn=self.activation,
            block_config=block_config,
            a2a_only=self.a2a_only,
            no_comm=self.no_comm,
            # Optional parameters (not used in basic case)
            subc_quant_wsz=None,
            w1_scale=None,
            w2_scale=None,
            w3_scale=None,
            b1=None,
            b2=None,
            b3=None,
            ep_axis_name="tensor",
        )

        output = jax.sharding.reshard(output, NamedSharding(self.mesh, P(None, None)))
        return output
