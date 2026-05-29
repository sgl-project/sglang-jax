"""Gating and Top-K routing for MoE layers."""

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    topk_ids_logical_to_physical,
)
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
            jax.random.normal(
                jax.random.PRNGKey(0),
                (input_size, num_experts),
                dtype=jnp.float32,
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
        logits = jnp.dot(hidden_states, self.kernel.value, precision=jax.lax.Precision.HIGHEST)

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
        layer_id: int = 0,
    ):
        self.topk = topk
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.layer_id = layer_id

    @named_scope
    def __call__(
        self,
        router_logits: jax.Array,
        correction_bias: jax.Array = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
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

        if dispatch_info is not None:
            topk_ids = topk_ids_logical_to_physical(topk_ids, dispatch_info, self.layer_id)

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
        group_mask = jnp.clip(
            jax.nn.one_hot(group_idx, self.num_expert_group).sum(axis=1), 0, 1
        )  # [n, n_group]

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

        # Create group mask using scatter [n, n_group]
        group_mask = jnp.clip(jax.nn.one_hot(group_idx, self.num_expert_group).sum(axis=1), 0, 1)

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
