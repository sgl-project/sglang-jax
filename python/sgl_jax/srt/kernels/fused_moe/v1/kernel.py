# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/fused_moe/v1/kernel.py
# Copyright 2025 The tpu-inference Authors. All rights reserved.
"""TPU-Friendly Fused Mixture of Experts (MoE) kernel."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

cdiv = pl.cdiv


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FusedMoEBlockConfig:
    bt: int
    btc: int
    bf: int
    bfc: int
    bd1: int
    bd1c: int
    bd2: int
    bd2c: int
    bts: int | None = None

    def effective_for(
        self,
        *,
        num_tokens: int,
        ep_size: int,
        dtype: jnp.dtype,
        subc_quant_wsz: int | None = None,
    ) -> FusedMoEBlockConfig:
        """Return the *effective* config after applying kernel override rules.

        Important: validate after overrides, because these overrides affect the
        actual compiled kernel shapes/scratch.
        """
        if ep_size <= 0:
            raise ValueError(f"Expected {ep_size=} to be > 0.")
        if num_tokens % ep_size != 0:
            raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")

        t_packing = get_dtype_packing(dtype)
        local_num_tokens = num_tokens // ep_size

        # `bt` is the outer token tile size used for routing/comm and output tiling.
        # It must not exceed the local token count and must evenly divide it.
        # `bts` is the token tile size used inside expert_ffn for HBM<->VMEM staging and
        # inner GEMM batching. When unset, `bts` defaults to `bt`.
        bt = min(self.bt, local_num_tokens)
        bt = math.gcd(bt, local_num_tokens)
        bts = bt if self.bts is None else min(self.bts, bt)
        btc = min(self.btc, bts)
        if bts % btc != 0:
            raise ValueError(f"Expected {bts=} to be divisible by {btc=}.")

        bd1c = self.bd1c
        bfc = self.bfc
        if subc_quant_wsz is not None:
            bd1c = subc_quant_wsz * t_packing
            bfc = subc_quant_wsz

        return FusedMoEBlockConfig(
            bt=bt,
            bts=bts,
            bf=self.bf,
            bd1=self.bd1,
            bd2=self.bd2,
            btc=btc,
            bfc=bfc,
            bd1c=bd1c,
            bd2c=self.bd2c,
        )

    def tree_flatten(self):
        # Store values as aux data so configs behave like static leaves for JAX.
        bts = 0 if self.bts is None else int(self.bts)
        return (), (
            self.bt,
            self.bf,
            self.bd1,
            self.bd2,
            self.btc,
            self.bfc,
            self.bd1c,
            self.bd2c,
            bts,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        if len(aux_data) == 8:
            bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = aux_data
            bts = None
        else:
            bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c, bts = aux_data
            bts = None if bts == 0 else int(bts)
        return cls(bt=bt, bf=bf, bd1=bd1, bd2=bd2, btc=btc, bfc=bfc, bd1c=bd1c, bd2c=bd2c, bts=bts)

    def as_kwargs(self) -> dict[str, int]:
        out = {
            "bt": self.bt,
            "bts": self.bt if self.bts is None else int(self.bts),
            "bf": self.bf,
            "bd1": self.bd1,
            "bd2": self.bd2,
            "btc": self.btc,
            "bfc": self.bfc,
            "bd1c": self.bd1c,
            "bd2c": self.bd2c,
        }
        return out


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_packing(dtype):
    bits = jnp.dtype(dtype).itemsize * 8
    return 32 // bits


def broadcast_minor(src, shape):
    if src.shape == shape:
        return src
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % 128 == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[
        ..., : shape[-1]
    ]


def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""
    gate = jnp.clip(gate, a_max=limit)
    up = jnp.clip(up, a_min=-limit, a_max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def activation_fn(acc1, acc3, act_fn):
    if act_fn == "silu":
        return jax.nn.silu(acc1) * acc3
    elif act_fn == "gelu":
        return jax.nn.gelu(acc1) * acc3
    elif act_fn == "swigluoai":
        return swigluoai(acc1, acc3)
    else:
        raise RuntimeError(f"Unsupported activation function: {act_fn}")


def validate_fused_moe_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    ep_size: int,
    subc_quant_wsz: int | None,
    block_config: FusedMoEBlockConfig,
) -> None:
    """Validate a (post-override) block config against kernel constraints."""
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(f"Expected {hidden_size=} and {intermediate_size=} to be aligned to 128.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")
    if num_experts % ep_size != 0:
        raise ValueError(f"Expected {num_experts=} to be aligned to {ep_size=}.")
    if not (0 < top_k <= num_experts):
        raise ValueError(f"Expected {top_k=} to be in range (0, {num_experts=}].")

    t_packing = get_dtype_packing(dtype)
    tile_align = t_packing * 128
    local_num_tokens = num_tokens // ep_size

    bt = block_config.bt
    bts = bt if block_config.bts is None else block_config.bts
    bf = block_config.bf
    bd1 = block_config.bd1
    bd2 = block_config.bd2
    btc = block_config.btc
    bfc = block_config.bfc
    bd1c = block_config.bd1c
    bd2c = block_config.bd2c

    if local_num_tokens % t_packing != 0:
        raise ValueError(f"Expected {local_num_tokens=} to be aligned to {t_packing=}.")
    if bt % t_packing != 0:
        raise ValueError(f"Expected {bt=} to be aligned to {t_packing=}.")
    if local_num_tokens % bt != 0:
        raise ValueError(f"Expected {local_num_tokens=} to be divisible by {bt=}.")
    if bts % t_packing != 0:
        raise ValueError(f"Expected {bts=} to be aligned to {t_packing=}.")
    if bts > bt:
        raise ValueError(f"Expected {bts=} to be <= {bt=}.")
    if not (0 < btc <= bts):
        raise ValueError(f"Expected {btc=} to satisfy 0 < btc <= bts (got {bts=}).")
    if bts % btc != 0:
        raise ValueError(f"Expected {bts=} to be divisible by {btc=}.")

    if bf % 128 != 0:
        raise ValueError(f"Expected {bf=} to be aligned to 128.")
    if intermediate_size % bf != 0:
        raise ValueError(f"Expected {intermediate_size=} to be aligned to {bf=}.")
    if bfc % 128 != 0:
        raise ValueError(f"Expected {bfc=} to be aligned to 128.")
    if bf % bfc != 0:
        raise ValueError(f"Expected {bf=} to be aligned to {bfc=}.")

    if bd1 % tile_align != 0 or bd2 % tile_align != 0:
        raise ValueError(f"Expected {bd1=} and {bd2=} to be aligned to {tile_align=}.")
    if bd1c % tile_align != 0:
        raise ValueError(f"Expected {bd1c=} to be aligned to {tile_align=}.")
    if bd2c % tile_align != 0:
        raise ValueError(f"Expected {bd2c=} to be aligned to {tile_align=}.")
    if bd1 % bd1c != 0:
        raise ValueError(f"Expected {bd1=} to be aligned to {bd1c=}.")
    if bd2 % bd2c != 0:
        raise ValueError(f"Expected {bd2=} to be aligned to {bd2c=}.")
    if hidden_size % bd1 != 0 or hidden_size % bd2 != 0:
        raise ValueError(f"Expected {hidden_size=} to be aligned to {bd1=} and {bd2=}.")

    if subc_quant_wsz is not None:
        if subc_quant_wsz <= 0:
            raise ValueError(f"Expected {subc_quant_wsz=} to be non-negative.")
        if subc_quant_wsz % 256 != 0:
            raise ValueError(f"Expected {subc_quant_wsz=} to be aligned to 256.")
        if hidden_size % subc_quant_wsz != 0:
            raise ValueError(f"Expected {hidden_size=} to be aligned to {subc_quant_wsz=}.")
        if intermediate_size % subc_quant_wsz != 0:
            raise ValueError(f"Expected {intermediate_size=} to be aligned to {subc_quant_wsz=}.")
        if bd1c != subc_quant_wsz * t_packing:
            raise ValueError(
                f"Expected {bd1c=} to be {subc_quant_wsz * t_packing=} when quantized."
            )
        if bfc != subc_quant_wsz:
            raise ValueError(f"Expected {bfc=} to be {subc_quant_wsz=} when quantized.")


def ref_moe(
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    w3: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    use_grouped_topk: bool = False,
    num_groups: int = 1,
    top_k_groups: int = 1,
    bias: jax.Array | None = None,
    renormalize_topk_logits: bool = False,
    routed_scaling_factor: float | None = None,
    act_fn: str = "silu",
    subc_quant_wsz: int | None = None,
    w1_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // subc_quant_wsz, 1, intermediate_size)
    w2_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, intermediate_size // subc_quant_wsz, 1, hidden_size)
    w3_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // subc_quant_wsz, 1, intermediate_size)
    b1: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
    b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
    b3: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
):
    n_tokens = tokens.shape[0]  # num_tokens
    num_experts = gating_output.shape[-1]

    # Compute gating scores for all experts
    gating_logits_f32 = gating_output.astype(jnp.float32)

    routing_scores = (
        gating_logits_f32 + jnp.expand_dims(bias.astype(jnp.float32), 0)
        if bias is not None
        else gating_logits_f32
    )

    if use_grouped_topk:
        assert num_experts % num_groups == 0
        experts_per_group = num_experts // num_groups
        reshaped_routing_scores = routing_scores.reshape(n_tokens, num_groups, experts_per_group)

        if bias is not None:
            top2_vals, _ = lax.top_k(reshaped_routing_scores, 2)
            group_scores = jnp.sum(top2_vals, axis=-1)
        else:
            group_scores = jnp.max(reshaped_routing_scores, axis=-1)

        group_mask_accum = jnp.zeros((n_tokens, num_groups), dtype=jnp.bool_)
        temp_group_scores = group_scores
        group_iota = jax.lax.broadcasted_iota(jnp.int32, (n_tokens, num_groups), 1)

        for _ in range(top_k_groups):
            curr_max_group_idx = jnp.argmax(temp_group_scores, axis=1, keepdims=True)
            curr_mask = group_iota == curr_max_group_idx
            group_mask_accum = jnp.logical_or(group_mask_accum, curr_mask)
            temp_group_scores = jnp.where(curr_mask, -jnp.float32(jnp.inf), temp_group_scores)

        expert_mask = jnp.repeat(
            jnp.expand_dims(group_mask_accum, axis=2), experts_per_group, axis=2
        ).reshape(n_tokens, num_experts)

        routing_scores = jnp.where(expert_mask, routing_scores, -jnp.float32(jnp.inf))

    # Select top-k experts per token
    _, top_k_indices = lax.top_k(routing_scores, top_k)
    top_k_logits = jnp.take_along_axis(gating_logits_f32, top_k_indices, axis=-1)

    if renormalize_topk_logits:
        top_k_logits = top_k_logits / jnp.sum(top_k_logits, axis=-1, keepdims=True)

    if routed_scaling_factor is not None:
        top_k_logits *= routed_scaling_factor

    t_outputs = []
    hidden_size, intermediate_size = w1.shape[-2:]

    # Process each token individually
    for i in range(n_tokens):
        curr_token = jnp.expand_dims(tokens[i], axis=0)  # [1, hidden_size]
        assigned_expert_ids = top_k_indices[i]  # [top_k] - indices of selected experts for token i
        tok_expert_act = []

        # Process each selected expert for the current token
        for expert_id in assigned_expert_ids:
            # Get expert weights
            expert_w1 = w1[expert_id].astype(jnp.float32)
            expert_w3 = w3[expert_id].astype(jnp.float32)
            if w1_scale is not None:
                expert_w1 *= jnp.repeat(w1_scale[expert_id, :, 0], subc_quant_wsz, axis=0)[
                    :hidden_size
                ]
            if w3_scale is not None:
                expert_w3 *= jnp.repeat(w3_scale[expert_id, :, 0], subc_quant_wsz, axis=0)[
                    :hidden_size
                ]
            expert_weight_2 = w2[expert_id].astype(jnp.float32)  # [intermediate_size, hidden_size]
            if w2_scale is not None:
                expert_weight_2 *= jnp.repeat(w2_scale[expert_id, :, 0], subc_quant_wsz, axis=0)[
                    :intermediate_size
                ]

            # First linear layer (gate/up projections).
            gmm1_w1_proj = curr_token @ expert_w1  # [1, intermediate_size]
            gmm1_w3_proj = curr_token @ expert_w3  # [1, intermediate_size]
            if b1 is not None:
                gmm1_w1_proj += b1[expert_id : expert_id + 1, 0]
            if b3 is not None:
                gmm1_w3_proj += b3[expert_id : expert_id + 1, 0]

            # Apply gated activation: activation(gate) * up
            act = activation_fn(gmm1_w1_proj, gmm1_w3_proj, act_fn)

            # Second linear layer (down projection)
            gmm_2_out = act @ expert_weight_2  # [1, hidden_size]
            if b2 is not None:
                gmm_2_out += b2[expert_id : expert_id + 1, 0]
            tok_expert_act.append(gmm_2_out)

        # Combine outputs from all selected experts
        experts_act = jnp.concatenate(tok_expert_act, axis=0)  # [top_k, hidden_size]

        # Weighted sum using top-k gating weights
        top_k_weights = top_k_logits[i]  # [top_k]
        top_k_weights = jnp.expand_dims(top_k_weights, axis=1)  # [top_k, 1]
        weighted_output = jnp.sum(
            experts_act * top_k_weights, axis=0, keepdims=True
        )  # [1, hidden_size]

        t_outputs.append(weighted_output.astype(tokens.dtype))

    return jnp.concatenate(t_outputs, axis=0)  # [actual_num_tokens, hidden_size]


def _fused_ep_moe_kernel(
    # Input
    tokens_hbm,  # (local_num_tokens, t_packing, hidden_size // t_packing)
    w1_hbm,  # (local_num_experts, hidden_size, intermediate_size)
    w2_hbm,  # (local_num_experts, intermediate_size, hidden_size)
    w3_hbm,  # (local_num_experts, hidden_size, intermediate_size)
    # TODO(jevinjiang): We choose F32 scale for easier slicing. The extra
    # latency should be hidden in the pipeline overlapping. But is there a better
    # way to do this?
    w1_scale_hbm,  # None | F32(local_num_experts, cdiv(hidden_size, subc_quant_wsz), 1, intermediate_size)
    w2_scale_hbm,  # None | F32(local_num_experts, cdiv(intermediate_size, subc_quant_wsz), 1, hidden_size)
    w3_scale_hbm,  # None | F32(local_num_experts, cdiv(hidden_size, subc_quant_wsz), 1, intermediate_size)
    b1_hbm,  # None | F32(local_num_experts, 1, intermediate_size)
    b2_hbm,  # None | F32(local_num_experts, 1, hidden_size)
    b3_hbm,  # None | F32(local_num_experts, 1, intermediate_size)
    gating_hbm,  # (local_num_tokens, padded_num_experts)
    a2a_s_x2_hbm,  # (2, align_to(bt * num_devices, bts), t_packing, hidden_size // t_packing)
    a2a_s_acc_x2_hbm,  # (2, align_to(bt * num_devices, bts), t_packing, hidden_size // t_packing)
    a2a_g_hbm,  # (num_experts, bt, t_packing, hidden_size // t_packing)
    bias_hbm,  # None | F32(padded_num_experts,)
    # Output
    output_hbm,  # (local_num_tokens, hidden_size)
    # Scratch
    t2e_routing_x2_smem,  # (2, bt, padded_top_k)
    d2e_count_x2_smem,  # (2, num_devices, 1, padded_num_experts)
    expert_offsets_x2_smem,  # (2, 2, padded_num_experts): <bt_sem_id> x (a2a_s/a2a_g)
    expert_starts_x2_smem,  # (2, 1, padded_num_experts)
    expert_sizes_x2_smem,  # (2, 1, padded_num_experts)
    a2a_s_sends_x2_smem,  # <e_sem_id> (2,)
    ### Accumulation for gathered tokens:
    a2a_g_acc_vmem,  # (1, top_k, acc_bt, t_packing, hidden_size // t_packing)
    top_k_logits_vmem,  # F32(bt, top_k)
    ### Expert weight double buffering:
    b_gating_x2_vmem,  # (2, bt, padded_num_experts)
    b_output_x2_vmem,  # (2, bt, hidden_size)
    b_w1_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
    b_w3_x2_vmem,  # <bw_sem_id> (2, t_packing, bd1 // t_packing, bf)
    b_w2_x2_vmem,  # <bw_sem_id> (2, t_packing, bf, bd2 // t_packing)
    b_w1_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bd1 // t_packing // subc_quant_wsz, 1, bf)
    b_w3_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bd1 // t_packing // subc_quant_wsz, 1, bf)
    b_w2_scale_x2_vmem,  # None | <bw_sem_id> (2, t_packing, bf // subc_quant_wsz, 1, bd2 // t_packing)
    b_b1_x2_vmem,  # None | <bw_sem_id> (2, 1, bf)
    b_b3_x2_vmem,  # None | <bw_sem_id> (2, 1, bf)
    b_b2_x2_vmem,  # None | <bw_sem_id> (2, t_packing, 1, bd2 // t_packing)
    b_acc_vmem,  # F32(2, align_to(bt * num_devices, bts), 1, bf)
    t_stage_x2_vmem,  # <token_buf_id> (2, bts, t_packing, bd1 // t_packing)
    a2a_s_acc_stage_x3_vmem,  # <acc_buf_id> (3, bts, t_packing, bd2 // t_packing)
    b_bias_vmem,  # None | F32(padded_num_experts,)
    ### Semaphores:
    token_stage_x2_sems,  # DMA(2,): <token_buf_id>
    acc_stage_x3_sems,  # DMA(3,): <acc_buf_id>
    local_sems,  # (2, 5): weight ping-pong semaphores (plus gating/output on slot 0)
    send_x2_sems,  # <e_sem_id> (2,)
    recv_x2_sems,  # <e_sem_id> (2,)
    a2a_gather_sem,
    a2a_acc_sems,  # DMA(1,)
    barrier_sem,
    *,
    top_k: int,
    use_grouped_topk: bool = False,
    num_groups: int = 1,
    top_k_groups: int = 1,
    renormalize_topk_logits: bool,
    routed_scaling_factor: float | None = None,
    balanced_topk: bool,
    ep_axis_name: str,
    act_fn: str,
    subc_quant_wsz: int | None = None,
    # Kernel tuning params.
    bt: int,  # Outer token tile size (output tiling).
    bf: int,  # Block size of intermediate_size.
    bd1: int,  # Block size of hidden_size in w1.
    bd2: int,  # Block size of hidden_size in w2.
    bts: int,  # Token staging tile size inside expert_ffn.
    btc: int,  # Compute size of block tokens for active expert.
    bfc: int,  # Compute size of block intermediate_size.
    bd1c: int,  # Compute size of block hidden_size.
    bd2c: int,  # Compute size of block hidden_size.
):
    my_id = lax.axis_index(ep_axis_name)
    num_devices = lax.axis_size(ep_axis_name)
    local_num_tokens = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    assert b_output_x2_vmem.shape[1] == bt, (b_output_x2_vmem.shape[1], bt)
    assert local_num_tokens % bt == 0, (local_num_tokens, bt)
    num_bt = local_num_tokens // bt
    a2a_max_tokens = a2a_s_x2_hbm.shape[1]
    right_id = (my_id + 1) % num_devices
    num_experts = a2a_g_hbm.shape[0]
    padded_num_experts = d2e_count_x2_smem.shape[-1]
    padded_top_k = t2e_routing_x2_smem.shape[-1]
    assert padded_num_experts == align_to(num_experts, 128)
    assert padded_top_k == align_to(top_k, 128)

    t_dtype = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    assert a2a_g_hbm.dtype == t_dtype
    assert w1_hbm.dtype == w2_hbm.dtype
    assert w3_hbm.dtype == w2_hbm.dtype

    assert bd1 % bd1c == 0
    assert bd2 % bd2c == 0
    assert bf % bfc == 0
    assert hidden_size % t_packing == 0
    assert bd1 % t_packing == 0
    assert bd2 % t_packing == 0
    assert bd1c % t_packing == 0
    assert bd2c % t_packing == 0

    assert bts % t_packing == 0
    assert bts % btc == 0
    assert bts <= bt

    h_per_t_packing = hidden_size // t_packing
    assert tokens_hbm.shape[-1] == h_per_t_packing
    bd1_per_t_packing = bd1 // t_packing
    bd2_per_t_packing = bd2 // t_packing
    bd1c_per_t_packing = bd1c // t_packing
    bd2c_per_t_packing = bd2c // t_packing

    if subc_quant_wsz is not None:
        assert subc_quant_wsz % 256 == 0
        assert bd1c_per_t_packing == subc_quant_wsz
        assert bfc == subc_quant_wsz
        assert bd1 % subc_quant_wsz == 0
        assert bf % subc_quant_wsz == 0
        assert bd1_per_t_packing % subc_quant_wsz == 0
        assert h_per_t_packing % subc_quant_wsz == 0

    num_bf = cdiv(intermediate_size, bf)
    num_bd1 = cdiv(hidden_size, bd1)
    num_bd2 = cdiv(hidden_size, bd2)

    def get_mesh_device_id(ep_rank):
        dp_rank = jax.lax.axis_index("data")
        return (dp_rank, ep_rank)

    def sync_barrier():
        # Full mesh barrier (matches epic/integrate-fused-moe). The previous
        # "signal right + wait 1" is only a neighbor fence (not a global barrier)
        # and can lead to rare deadlocks when subsequent comm assumes all peers
        # reached the same phase.
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    def start_fetch_b_gating(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        bt_start = bt_id * bt
        bt_size = bt
        # Hint Mosaic about HBM tiling alignment for `tpu.memref_slice`:
        # - f32 is typically tiled as (8, 128)
        # - bf16/f16 is typically tiled as (16, 128)
        gating_tile0 = 256 // (jnp.dtype(gating_hbm.dtype).itemsize * 8)
        if bt_size % gating_tile0 == 0:
            bt_start = pl.multiple_of(bt_start, gating_tile0)
        pltpu.make_async_copy(
            src_ref=gating_hbm.at[pl.ds(bt_start, bt_size)],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id, pl.ds(0, bt_size)],
            sem=b_gating_sem,
        ).start(priority=priority)

    def wait_fetch_b_gating(*, bt_id):
        bt_sem_id = bt_id & jnp.int32(1)
        b_gating_sem = local_sems.at[bt_sem_id, 0]
        pltpu.make_async_copy(
            src_ref=b_gating_x2_vmem.at[bt_sem_id],
            dst_ref=b_gating_x2_vmem.at[bt_sem_id],
            sem=b_gating_sem,
        ).wait()

    def get_top_k(input_logits, top_k, renormalize_topk_logits, *, out_top_k_logits_vmem):
        num_tokens = input_logits.shape[0]

        if b_bias_vmem is not None:
            # b_bias_vmem (padded_num_experts,)
            bias_val = b_bias_vmem[...]
            routing_scores = input_logits + jnp.expand_dims(bias_val[: input_logits.shape[1]], 0)
        else:
            routing_scores = input_logits

        if use_grouped_topk:
            curr_num_experts = input_logits.shape[1]
            experts_per_group = curr_num_experts // num_groups

            group_scores_list = []
            for g in range(num_groups):
                start = g * experts_per_group
                end = start + experts_per_group
                group_slice = routing_scores[:, start:end]

                if b_bias_vmem is not None:
                    # Specific bias logic for grouped topk: sum of top 2
                    val1 = jnp.max(group_slice, axis=1, keepdims=True)
                    idx1 = jnp.argmax(group_slice, axis=1, keepdims=True)
                    iota_slice = jax.lax.broadcasted_iota(jnp.int32, group_slice.shape, 1)
                    mask1 = iota_slice == idx1
                    group_slice_masked = jnp.where(mask1, -jnp.float32(jnp.inf), group_slice)
                    val2 = jnp.max(group_slice_masked, axis=1, keepdims=True)
                    g_score = val1 + val2
                else:
                    g_score = jnp.max(group_slice, axis=1, keepdims=True)
                group_scores_list.append(g_score)

            group_scores = jnp.concatenate(group_scores_list, axis=1)
            group_mask_accum = jnp.zeros((num_tokens, num_groups), dtype=jnp.bool_)
            temp_group_scores = group_scores
            group_iota = jax.lax.broadcasted_iota(jnp.int32, (num_tokens, num_groups), 1)

            for _ in range(top_k_groups):
                curr_max_group_idx = jnp.argmax(temp_group_scores, axis=1, keepdims=True)
                curr_mask = group_iota == curr_max_group_idx
                group_mask_accum = jnp.logical_or(group_mask_accum, curr_mask)
                temp_group_scores = jnp.where(curr_mask, -jnp.float32(jnp.inf), temp_group_scores)

            masked_routing_slices = []
            for g in range(num_groups):
                g_mask = group_mask_accum[:, g : g + 1]
                start = g * experts_per_group
                end = start + experts_per_group
                inp_slice = routing_scores[:, start:end]
                masked_slice = jnp.where(g_mask, inp_slice, -jnp.float32(jnp.inf))
                masked_routing_slices.append(masked_slice)
            curr_scores = jnp.concatenate(masked_routing_slices, axis=1)
        else:
            curr_scores = routing_scores

        padded_k_shape = (curr_scores.shape[0], padded_top_k)
        top_k_logits_lst = []
        t2e = jnp.zeros(curr_scores.shape, dtype=jnp.int32)
        t2e_routing = jnp.zeros(padded_k_shape, dtype=jnp.int32)
        iota = jax.lax.broadcasted_iota(jnp.int32, curr_scores.shape, 1)
        padded_k_iota = jax.lax.broadcasted_iota(jnp.int32, padded_k_shape, 1)
        top_k_logits_sum = jnp.zeros(padded_k_shape, jnp.float32)

        for k_id in range(top_k):
            # Select expert from current scores (masked/biased)
            curr_indices = jnp.argmax(curr_scores[:, :num_experts], axis=1, keepdims=True)
            top_k_indices = jnp.broadcast_to(curr_indices, padded_k_shape)

            selection_mask = iota == broadcast_minor(top_k_indices, curr_scores.shape)

            # Extract value from original input logits
            val = jnp.sum(
                jnp.where(selection_mask, input_logits[:, :num_experts], 0.0), axis=1, keepdims=True
            )

            top_k_logits = jnp.broadcast_to(val, padded_k_shape).astype(input_logits.dtype)
            top_k_logits_lst.append(top_k_logits)

            if renormalize_topk_logits:
                top_k_logits_sum += top_k_logits

            t2e_routing = jnp.where(padded_k_iota == k_id, top_k_indices, t2e_routing)
            mask = selection_mask
            t2e += mask.astype(jnp.int32)
            if k_id != top_k - 1:
                curr_scores = jnp.where(mask, -jnp.float32(jnp.inf), curr_scores)

        if renormalize_topk_logits:
            for k_id in range(top_k):
                top_k_logits_lst[k_id] /= top_k_logits_sum + 1e-6

        if routed_scaling_factor is not None:
            for k_id in range(top_k):
                top_k_logits_lst[k_id] *= routed_scaling_factor

        for k_id in range(top_k):
            out_top_k_logits_vmem.at[pl.ds(0, input_logits.shape[0]), pl.ds(k_id, 1)][...] = (
                top_k_logits_lst[k_id][:, :1].astype(jnp.float32)
            )

        expert_sizes = jnp.sum(t2e, axis=0, keepdims=True)
        expert_starts = jnp.zeros_like(expert_sizes)
        return t2e_routing, expert_sizes, expert_starts

    def all_reduce_metadata(*, bt_sem_id, t2e_routing, starts, sizes):
        send_sem = send_x2_sems.at[0]
        recv_sem = recv_x2_sems.at[0]

        # All-reduce to accumulate starts and sizes and transfer to SMEM.
        def _all_reduce_metadata(
            t2e_routing_vmem,
            d2e_count_vmem,
            offsets_vmem,
            starts_vmem,
            sizes_vmem,
        ):
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            # TODO(jevinjiang): check how slow is VMEM -> SMEM.
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            t2e_routing_vmem[...] = t2e_routing
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            reduced_sizes = sizes
            reduced_starts = starts
            row_id = my_id
            d2e_count_vmem[row_id] = sizes
            for i in range(num_devices - 1):
                sync_barrier()
                # TODO(jevinjiang): we can use double buffering to improve AR if needed.
                pltpu.async_remote_copy(
                    src_ref=d2e_count_vmem.at[row_id],
                    dst_ref=d2e_count_vmem.at[row_id],
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=get_mesh_device_id(right_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).wait()
                row_id = (row_id + num_devices - 1) % num_devices
                new_sizes = d2e_count_vmem[row_id]
                reduced_sizes += new_sizes
                reduced_starts += lax.select(my_id > i, new_sizes, jnp.zeros_like(new_sizes))
            starts_vmem[...] = reduced_starts
            sizes_vmem[...] = reduced_sizes

            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            # TODO(jevinjiang): if d2e_count is too big, we can store in HBM and fetch
            # to SMEM partially.
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=send_sem,
            )

            t2e_routing_copy.wait()
            d2e_count_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

        pl.run_scoped(
            _all_reduce_metadata,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
        )

    def start_a2a_scatter(*, bt_sem_id, e_sem_id, local_e_id, bt_start):
        # Counting the number of remote sends from the current device.
        # Use `lax.fori_loop` to avoid unrolling `bt` (huge MLIR / slow compile).
        def _scatter_one(
            t_id, send_sz, e_sem_id=e_sem_id, local_e_id=local_e_id, bt_start=bt_start
        ):
            src_t_id = bt_start + t_id
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]
                is_active_expert = e_id % local_num_experts == local_e_id
                recv_id = e_id // local_num_experts
                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id]
                sz = lax.select(is_active_expert, jnp.int32(1), jnp.int32(0))
                is_local = recv_id == my_id
                local_sz = lax.select(is_local, sz, jnp.int32(0))
                remote_sz = lax.select(is_local, jnp.int32(0), sz)
                send_sz += remote_sz
                expert_offsets_x2_smem[bt_sem_id, 0, e_id] = offset + local_sz + remote_sz
                start = expert_starts_x2_smem[bt_sem_id, 0, e_id] + offset
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(src_t_id, local_sz)],
                    dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(start, local_sz)],
                    sem=recv_x2_sems.at[e_sem_id],
                ).start()
                pltpu.make_async_remote_copy(
                    src_ref=tokens_hbm.at[pl.ds(src_t_id, remote_sz)],
                    dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(start, remote_sz)],
                    send_sem=send_x2_sems.at[e_sem_id],
                    recv_sem=recv_x2_sems.at[e_sem_id],
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            return send_sz

        send_sz = lax.fori_loop(
            0,
            bt,
            _scatter_one,
            jnp.int32(0),
            unroll=False,
        )
        a2a_s_sends_x2_smem[e_sem_id] = send_sz

    def wait_a2a_scatter_recv(*, bt_sem_id, e_sem_id, local_e_id):
        e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, sz)],
            sem=recv_x2_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_scatter_send(e_sem_id):
        sz = a2a_s_sends_x2_smem[e_sem_id]
        pltpu.make_async_copy(
            src_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, sz)],
            dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, sz)],
            sem=send_x2_sems.at[e_sem_id],
        ).wait()

    def start_a2a_gather(*, bt_sem_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        start = 0
        src_ref = a2a_s_acc_x2_hbm
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, 0)
            remote_sz = lax.select(is_local, 0, sz)
            pltpu.make_async_copy(
                src_ref=src_ref.at[e_sem_id, pl.ds(start, local_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                sem=a2a_gather_sem,
            ).start()
            pltpu.make_async_remote_copy(
                src_ref=src_ref.at[e_sem_id, pl.ds(start, remote_sz)],
                dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                send_sem=send_x2_sems.at[e_sem_id],
                recv_sem=a2a_gather_sem,
                device_id=get_mesh_device_id(recv_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

            start += sz

    def wait_a2a_gather_send(*, bt_sem_id, e_sem_id, local_e_id):
        my_e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(local_e_id >= 0, local_e_id < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)

        # Important: wait via `a2a_g_hbm` itself (matches f5b4) so reads from
        # `a2a_g_hbm` can't be reordered before the gather completes.
        ref = a2a_g_hbm.at[0, pl.ds(0, remote_sz)]
        pltpu.make_async_copy(
            src_ref=ref,
            dst_ref=ref,
            sem=send_x2_sems.at[e_sem_id],
        ).wait()

    def wait_a2a_gather_recv_all(*, bt_size):
        # Align to f5b4: wait using a flat slice into `a2a_g_hbm` sized to the
        # total gathered token vectors for this bt tile (`top_k * bt_size`).
        sz = jnp.int32(bt_size * top_k)
        ref = a2a_g_hbm.at[0, pl.ds(0, sz)]
        pltpu.make_async_copy(
            src_ref=ref,
            dst_ref=ref,
            sem=a2a_gather_sem,
        ).wait()

    def start_fetch_and_wait_bias():
        if bias_hbm is not None:
            bias_copy = pltpu.make_async_copy(
                src_ref=bias_hbm,
                dst_ref=b_bias_vmem,
                sem=local_sems.at[0, 0],
            )
            bias_copy.start()
            bias_copy.wait()

    def start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd1_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    pl.ds(offset, bd1_per_t_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w1_x2_vmem.at[bw1_sem_id, p],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()
            if w1_scale_hbm is not None:
                assert subc_quant_wsz is not None
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        local_e_id,
                        pl.ds(
                            offset // subc_quant_wsz,
                            bd1_per_t_packing // subc_quant_wsz,
                        ),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w1_scale_x2_vmem.at[bw1_sem_id, p],
                    sem=local_sems.at[bw1_sem_id, 1],
                ).start()
        if b1_hbm is not None and bd1_id == 0:
            pltpu.make_async_copy(
                src_ref=b1_hbm.at[local_e_id, pl.ds(0, 1), pl.ds(bf_id * bf, bf)],
                dst_ref=b_b1_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw1_sem_id, 1],
            ).start()

    def start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd2_id * bd2_per_t_packing
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, bf),
                    pl.ds(offset, bd2_per_t_packing),
                ],
                dst_ref=b_w2_x2_vmem.at[bw2_sem_id, p],
                sem=local_sems.at[bw2_sem_id, 2],
            ).start()
            if w2_scale_hbm is not None:
                assert subc_quant_wsz is not None
                pltpu.make_async_copy(
                    src_ref=w2_scale_hbm.at[
                        local_e_id,
                        pl.ds(bf_id * bf // subc_quant_wsz, bf // subc_quant_wsz),
                        pl.ds(0, 1),
                        pl.ds(offset, bd2_per_t_packing),
                    ],
                    dst_ref=b_w2_scale_x2_vmem.at[bw2_sem_id, p],
                    sem=local_sems.at[bw2_sem_id, 2],
                ).start()
            if b2_hbm is not None and bf_id == 0:
                pltpu.make_async_copy(
                    src_ref=b2_hbm.at[local_e_id, pl.ds(0, 1), pl.ds(offset, bd2_per_t_packing)],
                    dst_ref=b_b2_x2_vmem.at[bd2_id % 2, p],
                    sem=local_sems.at[bw2_sem_id, 2],
                ).start()

    def start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        for p in range(t_packing):
            offset = p * h_per_t_packing + bd3_id * bd1_per_t_packing
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[
                    local_e_id,
                    pl.ds(offset, bd1_per_t_packing),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w3_x2_vmem.at[bw3_sem_id, p],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()
            if w3_scale_hbm is not None:
                assert subc_quant_wsz is not None
                pltpu.make_async_copy(
                    src_ref=w3_scale_hbm.at[
                        local_e_id,
                        pl.ds(
                            offset // subc_quant_wsz,
                            bd1_per_t_packing // subc_quant_wsz,
                        ),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w3_scale_x2_vmem.at[bw3_sem_id, p],
                    sem=local_sems.at[bw3_sem_id, 3],
                ).start()
        if b3_hbm is not None and bd3_id == 0:
            pltpu.make_async_copy(
                src_ref=b3_hbm.at[local_e_id, pl.ds(0, 1), pl.ds(bf_id * bf, bf)],
                dst_ref=b_b3_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw3_sem_id, 3],
            ).start()

    def wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[bw1_sem_id],
            dst_ref=b_w1_x2_vmem.at[bw1_sem_id],
            sem=local_sems.at[bw1_sem_id, 1],
        ).wait()
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w1_scale_x2_vmem.at[bw1_sem_id],
                dst_ref=b_w1_scale_x2_vmem.at[bw1_sem_id],
                sem=local_sems.at[bw1_sem_id, 1],
            ).wait()
        if b1_hbm is not None and bd1_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b1_x2_vmem.at[bf_id % 2],
                dst_ref=b_b1_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw1_sem_id, 1],
            ).wait()

    def wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[bw2_sem_id],
            dst_ref=b_w2_x2_vmem.at[bw2_sem_id],
            sem=local_sems.at[bw2_sem_id, 2],
        ).wait()
        if w2_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w2_scale_x2_vmem.at[bw2_sem_id],
                dst_ref=b_w2_scale_x2_vmem.at[bw2_sem_id],
                sem=local_sems.at[bw2_sem_id, 2],
            ).wait()
        if b2_hbm is not None and bf_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b2_x2_vmem.at[bd2_id % 2],
                dst_ref=b_b2_x2_vmem.at[bd2_id % 2],
                sem=local_sems.at[bw2_sem_id, 2],
            ).wait()

    def wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id):
        del local_e_id
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[bw3_sem_id],
            dst_ref=b_w3_x2_vmem.at[bw3_sem_id],
            sem=local_sems.at[bw3_sem_id, 3],
        ).wait()
        if w3_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w3_scale_x2_vmem.at[bw3_sem_id],
                dst_ref=b_w3_scale_x2_vmem.at[bw3_sem_id],
                sem=local_sems.at[bw3_sem_id, 3],
            ).wait()
        if b3_hbm is not None and bd3_id == 0:
            pltpu.make_async_copy(
                src_ref=b_b3_x2_vmem.at[bf_id % 2],
                dst_ref=b_b3_x2_vmem.at[bf_id % 2],
                sem=local_sems.at[bw3_sem_id, 3],
            ).wait()

    def start_fetch_next_bw(local_e_id, bw_sem_id, bf_id, bd1_id, bd2_id):
        next_bd1_id = bd1_id + 1
        next_bd2_id = bd2_id + 1
        next_sem_id = (bw_sem_id + 1) % 2

        if bf_id >= num_bf:
            return
        if next_bd1_id < num_bd1:
            start_fetch_bw1(local_e_id, next_sem_id, bf_id, next_bd1_id)
            start_fetch_bw3(local_e_id, next_sem_id, bf_id, next_bd1_id)
        elif next_bd1_id == num_bd1:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, 0)
        elif next_bd2_id < num_bd2:
            start_fetch_bw2(local_e_id, next_sem_id, bf_id, next_bd2_id)
        elif next_bd2_id == num_bd2:
            start_fetch_next_bw(local_e_id, bw_sem_id, bf_id + 1, -1, -1)
        else:
            raise RuntimeError("Unreachable")

    def dynamic_ffn1(
        t_vmem,
        w1_vmem,
        w1_scale_vmem,
        b1_vmem,
        w3_vmem,
        w3_scale_vmem,
        b3_vmem,
        acc1_vmem,
        acc3_vmem,
        dyn_sz,
        should_init,
    ):
        token_tile = t_vmem.shape[0]
        assert t_vmem.shape == (token_tile, t_packing, bd1 // t_packing)
        assert w1_vmem.shape == w3_vmem.shape == (t_packing, bd1_per_t_packing, bf)
        assert acc1_vmem.shape == acc3_vmem.shape == (token_tile, bf)
        assert bd1 % (t_packing * 128) == 0, (bd1, t_packing)
        assert bd1c % (t_packing * 128) == 0, (bd1c, t_packing)
        assert bd1_per_t_packing % bd1c_per_t_packing == 0
        if w1_scale_vmem is not None:
            assert w1_scale_vmem.shape == (
                t_packing,
                bd1_per_t_packing // subc_quant_wsz,
                1,
                bf,
            )
            assert bd1c_per_t_packing == subc_quant_wsz
        if w3_scale_vmem is not None:
            assert w3_scale_vmem.shape == (
                t_packing,
                bd1_per_t_packing // subc_quant_wsz,
                1,
                bf,
            )
            assert bd1c_per_t_packing == subc_quant_wsz

        dyn_sz_i32 = dyn_sz.astype(jnp.int32)
        num_loops = lax.select(dyn_sz_i32 > 0, (dyn_sz_i32 + (btc - 1)) // btc, 0)

        def body(btc_id, _):
            for bd1c_id in range(cdiv(bd1, bd1c)):
                for p_id in range(t_packing):
                    t = t_vmem[
                        pl.ds(btc_id * btc, btc),
                        p_id,
                        pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                    ]
                    for bfc_id in range(cdiv(bf, bfc)):
                        w_slices = (
                            p_id,
                            pl.ds(bd1c_id * bd1c_per_t_packing, bd1c_per_t_packing),
                            pl.ds(bfc_id * bfc, bfc),
                        )
                        w1 = w1_vmem[*w_slices]
                        acc1 = jnp.dot(t, w1, preferred_element_type=jnp.float32)

                        if w1_scale_vmem is not None:
                            w1_scale_slices = (
                                p_id,
                                bd1c_id,
                                pl.ds(0, 1),
                                pl.ds(bfc_id * bfc, bfc),
                            )
                            # TODO(jevinjiang): can use mosaic to load with stride 0.
                            w1_scale = jnp.broadcast_to(w1_scale_vmem[*w1_scale_slices], acc1.shape)
                            acc1 *= w1_scale

                        w3 = w3_vmem[*w_slices]

                        acc3 = jnp.dot(t, w3, preferred_element_type=jnp.float32)

                        if w3_scale_vmem is not None:
                            w3_scale_slices = (
                                p_id,
                                bd1c_id,
                                pl.ds(0, 1),
                                pl.ds(bfc_id * bfc, bfc),
                            )
                            w3_scale = jnp.broadcast_to(w3_scale_vmem[*w3_scale_slices], acc3.shape)
                            acc3 *= w3_scale

                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        if should_init and p_id == bd1c_id == 0:
                            if b1_vmem is not None:
                                b1_scale_slices = (
                                    pl.ds(0, 1),
                                    pl.ds(bfc_id * bfc, bfc),
                                )
                                b1 = jnp.broadcast_to(b1_vmem[*b1_scale_slices], acc1.shape)
                                acc1 += b1
                            if b3_vmem is not None:
                                b3_scale_slices = (
                                    pl.ds(0, 1),
                                    pl.ds(bfc_id * bfc, bfc),
                                )
                                b3 = jnp.broadcast_to(b3_vmem[*b3_scale_slices], acc1.shape)
                                acc3 += b3

                            acc1_vmem[*acc_slices] = acc1
                            acc3_vmem[*acc_slices] = acc3
                        else:
                            acc1_vmem[*acc_slices] += acc1
                            acc3_vmem[*acc_slices] += acc3

        lax.fori_loop(0, num_loops, body, None)

    def dynamic_ffn2(
        acc1_vmem,
        acc3_vmem,
        w2_vmem,
        w2_scale_vmem,
        b2_vmem,
        res_vmem,
        dyn_sz,
        should_init,
    ):
        token_tile = res_vmem.shape[0]
        assert res_vmem.shape == (token_tile, t_packing, bd2_per_t_packing)
        assert w2_vmem.shape == (t_packing, bf, bd2_per_t_packing)
        assert acc1_vmem.shape == acc3_vmem.shape == (token_tile, bf)
        assert bd2 % (t_packing * 128) == 0, (bd2, t_packing)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)
        assert t_dtype in (jnp.float32, jnp.bfloat16)

        if w2_scale_vmem is not None:
            assert w2_scale_vmem.shape == (
                t_packing,
                bf // subc_quant_wsz,
                1,
                bd2_per_t_packing,
            )
            assert bfc == subc_quant_wsz

        dyn_sz_i32 = dyn_sz.astype(jnp.int32)
        num_loops = lax.select(dyn_sz_i32 > 0, (dyn_sz_i32 + (btc - 1)) // btc, 0)
        assert bd2c % (t_packing * 128) == 0, (bd2c, t_packing)

        def body(btc_id, __):
            for bd2c_id in range(cdiv(bd2, bd2c)):
                for p_id in range(t_packing):
                    res = jnp.zeros((btc, bd2c_per_t_packing), dtype=jnp.float32)

                    if b2_vmem is not None and should_init:
                        b2_scale_slices = (
                            p_id,
                            pl.ds(0, 1),
                            pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                        )
                        b2 = jnp.broadcast_to(b2_vmem[*b2_scale_slices], res.shape)
                        res += b2

                    for bfc_id in range(cdiv(bf, bfc)):
                        acc_slices = (pl.ds(btc_id * btc, btc), pl.ds(bfc_id * bfc, bfc))
                        acc1 = acc1_vmem[*acc_slices]
                        acc3 = acc3_vmem[*acc_slices]
                        act = activation_fn(acc1, acc3, act_fn)
                        w2 = w2_vmem[
                            p_id,
                            pl.ds(bfc_id * bfc, bfc),
                            pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                        ]
                        acc = jnp.dot(act, w2, preferred_element_type=jnp.float32)
                        if w2_scale_vmem is not None:
                            w2_scale_slices = (
                                p_id,
                                bfc_id,
                                pl.ds(0, 1),
                                pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                            )
                            w2_scale = jnp.broadcast_to(w2_scale_vmem[*w2_scale_slices], acc.shape)
                            acc *= w2_scale
                        res += acc
                    res_slice = res_vmem.at[
                        pl.ds(btc_id * btc, btc),
                        p_id,
                        pl.ds(bd2c_id * bd2c_per_t_packing, bd2c_per_t_packing),
                    ]
                    if should_init:
                        res_slice[...] = res.astype(t_dtype)
                    else:
                        res_slice[...] = (res_slice[...].astype(jnp.float32) + res).astype(t_dtype)

        lax.fori_loop(0, num_loops, body, None)

    def expert_ffn(bt_sem_id, e_sem_id, local_e_id):
        bw_sem_id = jnp.int32(0)
        b_acc_vmem_2d = b_acc_vmem.reshape(2, a2a_max_tokens, bf)
        b_acc1_vmem = b_acc_vmem_2d.at[0]
        b_acc3_vmem = b_acc_vmem_2d.at[1]

        e_id = my_id * local_num_experts + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        dyn_sz_i32 = dyn_sz.astype(jnp.int32)

        bd1_per_t_packing = bd1 // t_packing
        bd2_per_t_packing = bd2 // t_packing
        # Stage tokens in bt-sized tiles from HBM -> VMEM (to reduce staging frequency),
        # while keeping `btc` as the inner compute block size.
        token_tile = bts
        num_token_tiles = (dyn_sz_i32 + (token_tile - 1)) // token_tile

        def start_stage_a2a_s_tile_from_hbm(tile_start, bd1_id, buf_id):
            pltpu.make_async_copy(
                src_ref=a2a_s_x2_hbm.at[
                    e_sem_id,
                    pl.ds(tile_start, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(bd1_id * bd1_per_t_packing, bd1_per_t_packing),
                ],
                dst_ref=t_stage_x2_vmem.at[
                    buf_id,
                    pl.ds(0, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(0, bd1_per_t_packing),
                ],
                sem=token_stage_x2_sems.at[buf_id],
            ).start()

        def wait_stage_a2a_s_tile(buf_id):
            pltpu.make_async_copy(
                src_ref=t_stage_x2_vmem.at[buf_id, pl.ds(0, token_tile)],
                dst_ref=t_stage_x2_vmem.at[buf_id, pl.ds(0, token_tile)],
                sem=token_stage_x2_sems.at[buf_id],
            ).wait()

        def start_load_stage_a2a_s_acc_tile_from_hbm(tile_start, bd2_start, buf_id):
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_x2_hbm.at[
                    e_sem_id,
                    pl.ds(tile_start, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(bd2_start, bd2_per_t_packing),
                ],
                dst_ref=a2a_s_acc_stage_x3_vmem.at[
                    buf_id,
                    pl.ds(0, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(0, bd2_per_t_packing),
                ],
                sem=acc_stage_x3_sems.at[buf_id],
            ).start()

        def wait_stage_a2a_s_acc_tile(buf_id):
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_stage_x3_vmem.at[
                    buf_id,
                    pl.ds(0, token_tile),
                ],
                dst_ref=a2a_s_acc_stage_x3_vmem.at[
                    buf_id,
                    pl.ds(0, token_tile),
                ],
                sem=acc_stage_x3_sems.at[buf_id],
            ).wait()

        def start_store_stage_a2a_s_acc_tile_to_hbm(tile_start, bd2_start, buf_id):
            pltpu.make_async_copy(
                src_ref=a2a_s_acc_stage_x3_vmem.at[
                    buf_id,
                    pl.ds(0, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(0, bd2_per_t_packing),
                ],
                dst_ref=a2a_s_acc_x2_hbm.at[
                    e_sem_id,
                    pl.ds(tile_start, token_tile),
                    pl.ds(0, t_packing),
                    pl.ds(bd2_start, bd2_per_t_packing),
                ],
                sem=acc_stage_x3_sems.at[buf_id],
            ).start()

        def with_static_bw(bw_sem_id, body):
            return lax.cond(
                bw_sem_id == 0,
                lambda _: body(0),
                lambda _: body(1),
                operand=None,
            )

        def run_gate_up_slices(*, bf_id: int, bw_sem_id):
            def run_gate_up_bd1(*, bd1_id, bw_sem_id, should_init_ffn1: bool):
                def body(bw_sem_id: int):
                    next_bw_sem_id = 1 - bw_sem_id
                    next_bd1_id = bd1_id + jnp.int32(1)

                    @pl.when(next_bd1_id < num_bd1)
                    def _():
                        start_fetch_bw1(local_e_id, next_bw_sem_id, bf_id, next_bd1_id)
                        start_fetch_bw3(local_e_id, next_bw_sem_id, bf_id, next_bd1_id)

                    @pl.when(next_bd1_id == num_bd1)
                    def _():
                        start_fetch_bw2(local_e_id, next_bw_sem_id, bf_id, jnp.int32(0))

                    w1_scale_vmem = (
                        None if b_w1_scale_x2_vmem is None else b_w1_scale_x2_vmem.at[bw_sem_id]
                    )
                    w3_scale_vmem = (
                        None if b_w3_scale_x2_vmem is None else b_w3_scale_x2_vmem.at[bw_sem_id]
                    )
                    b1_vmem = None if b_b1_x2_vmem is None else b_b1_x2_vmem.at[bf_id % 2]
                    b3_vmem = None if b_b3_x2_vmem is None else b_b3_x2_vmem.at[bf_id % 2]

                    wait_fetch_bw1(local_e_id, bw_sem_id, bf_id, bd1_id)
                    wait_fetch_bw3(local_e_id, bw_sem_id, bf_id, bd1_id)
                    w1_vmem = b_w1_x2_vmem.at[bw_sem_id]
                    w3_vmem = b_w3_x2_vmem.at[bw_sem_id]

                    # Double-buffer token staging from HBM -> VMEM to overlap with FFN1 compute.
                    # Note: a2a_s_x2_hbm is already double-buffered by e_sem_id.
                    @pl.when(num_token_tiles > 0)
                    def _(bd1_id=bd1_id):
                        start_stage_a2a_s_tile_from_hbm(jnp.int32(0), bd1_id, jnp.int32(0))

                    def run_ffn1_tile(
                        token_tile_id,
                        token_buf_id,
                        num_token_tiles=num_token_tiles,
                        token_tile=token_tile,
                        dyn_sz_i32=dyn_sz_i32,
                        bd1_id=bd1_id,
                        w1_vmem=w1_vmem,
                        w1_scale_vmem=w1_scale_vmem,
                        b1_vmem=b1_vmem,
                        w3_vmem=w3_vmem,
                        w3_scale_vmem=w3_scale_vmem,
                        b3_vmem=b3_vmem,
                        should_init_ffn1=should_init_ffn1,
                    ):
                        tile_start = token_tile_id * token_tile

                        next_tile_id = token_tile_id + 1
                        next_buf_id = token_buf_id ^ jnp.int32(1)
                        next_start = next_tile_id * token_tile

                        @pl.when(next_tile_id < num_token_tiles)
                        def _prefetch(
                            next_start=next_start, next_buf_id=next_buf_id, bd1_id=bd1_id
                        ):
                            start_stage_a2a_s_tile_from_hbm(next_start, bd1_id, next_buf_id)

                        wait_stage_a2a_s_tile(token_buf_id)

                        tile_sz = jnp.maximum(jnp.minimum(dyn_sz_i32 - tile_start, token_tile), 0)
                        dynamic_ffn1(
                            t_vmem=t_stage_x2_vmem.at[token_buf_id],
                            w1_vmem=w1_vmem,
                            w1_scale_vmem=w1_scale_vmem,
                            b1_vmem=b1_vmem,
                            w3_vmem=w3_vmem,
                            w3_scale_vmem=w3_scale_vmem,
                            b3_vmem=b3_vmem,
                            acc1_vmem=b_acc1_vmem.at[pl.ds(tile_start, token_tile)],
                            acc3_vmem=b_acc3_vmem.at[pl.ds(tile_start, token_tile)],
                            dyn_sz=tile_sz,
                            should_init=should_init_ffn1,
                        )
                        return next_buf_id

                    lax.fori_loop(
                        0,
                        num_token_tiles,
                        run_ffn1_tile,
                        jnp.int32(0),
                        unroll=False,
                    )
                    return jnp.int32(next_bw_sem_id)

                return with_static_bw(bw_sem_id, body)

            if num_bd1 <= 0:
                return bw_sem_id

            # Peel bd1_id=0 so `should_init_ffn1` stays static.
            bw_sem_id = run_gate_up_bd1(
                bd1_id=jnp.int32(0), bw_sem_id=bw_sem_id, should_init_ffn1=True
            )

            def run_one_bd1_no_init(bd1_id, bw_sem_id):
                return run_gate_up_bd1(bd1_id=bd1_id, bw_sem_id=bw_sem_id, should_init_ffn1=False)

            return lax.fori_loop(1, num_bd1, run_one_bd1_no_init, bw_sem_id, unroll=False)

        def run_down_slices(*, bf_id: int, bw_sem_id):
            should_init_ffn2 = bf_id == 0

            def run_down_bd2(bd2_id, bw_sem_id):
                def body(bw_sem_id: int):
                    next_bw_sem_id = 1 - bw_sem_id
                    next_bd2_id = bd2_id + jnp.int32(1)

                    @pl.when(next_bd2_id < num_bd2)
                    def _():
                        start_fetch_bw2(local_e_id, next_bw_sem_id, bf_id, next_bd2_id)

                    if bf_id + 1 < num_bf:

                        @pl.when(next_bd2_id == num_bd2)
                        def _():
                            start_fetch_bw1(local_e_id, next_bw_sem_id, bf_id + 1, jnp.int32(0))
                            start_fetch_bw3(local_e_id, next_bw_sem_id, bf_id + 1, jnp.int32(0))

                    wait_fetch_bw2(local_e_id, bw_sem_id, bf_id, bd2_id)
                    if should_init_ffn2:

                        @pl.when(bd2_id == 0)
                        def _():
                            wait_a2a_gather_send(
                                bt_sem_id=bt_sem_id,
                                e_sem_id=e_sem_id,
                                local_e_id=local_e_id - 2,
                            )

                    w2_scale_vmem = (
                        None if b_w2_scale_x2_vmem is None else b_w2_scale_x2_vmem.at[bw_sem_id]
                    )
                    b2_vmem = None if b_b2_x2_vmem is None else b_b2_x2_vmem.at[bd2_id & 1]
                    bd2_start = bd2_id * bd2_per_t_packing
                    w2_vmem = b_w2_x2_vmem.at[bw_sem_id]

                    # Triple-buffer a2a_s_acc staging to overlap:
                    # - load(next tile) / compute(curr tile) / store(prev tile)
                    init_buf_compute = jnp.int32(0)
                    init_buf_store = jnp.int32(1)
                    init_buf_load = jnp.int32(2)
                    has_tiles = num_token_tiles > 0

                    if not should_init_ffn2:

                        @pl.when(has_tiles)
                        def _(bd2_start=bd2_start, init_buf_compute=init_buf_compute):
                            start_load_stage_a2a_s_acc_tile_from_hbm(
                                jnp.int32(0), bd2_start, init_buf_compute
                            )

                    def run_ffn2_tile(
                        token_tile_id,
                        state,
                        *,
                        bd2_start=bd2_start,
                        token_tile=token_tile,
                        dyn_sz_i32=dyn_sz_i32,
                        num_token_tiles=num_token_tiles,
                        w2_vmem=w2_vmem,
                        w2_scale_vmem=w2_scale_vmem,
                        b2_vmem=b2_vmem,
                        should_init_ffn2=should_init_ffn2,
                    ):
                        buf_compute, buf_store, buf_load = state
                        tile_start = token_tile_id * token_tile
                        tile_sz = jnp.maximum(jnp.minimum(dyn_sz_i32 - tile_start, token_tile), 0)

                        if not should_init_ffn2:
                            do_prefetch = token_tile_id + 1 < num_token_tiles
                            next_tile_start = (token_tile_id + 1) * token_tile

                            @pl.when(jnp.logical_and(do_prefetch, token_tile_id >= 2))
                            def _(buf_load=buf_load):
                                wait_stage_a2a_s_acc_tile(buf_load)

                            @pl.when(do_prefetch)
                            def _(
                                next_tile_start=next_tile_start,
                                bd2_start=bd2_start,
                                buf_load=buf_load,
                            ):
                                start_load_stage_a2a_s_acc_tile_from_hbm(
                                    next_tile_start, bd2_start, buf_load
                                )

                            wait_stage_a2a_s_acc_tile(buf_compute)
                        else:

                            @pl.when(token_tile_id >= 3)
                            def _(buf_compute=buf_compute):
                                wait_stage_a2a_s_acc_tile(buf_compute)

                        dynamic_ffn2(
                            acc1_vmem=b_acc1_vmem.at[pl.ds(tile_start, token_tile)],
                            acc3_vmem=b_acc3_vmem.at[pl.ds(tile_start, token_tile)],
                            w2_vmem=w2_vmem,
                            w2_scale_vmem=w2_scale_vmem,
                            b2_vmem=b2_vmem,
                            res_vmem=a2a_s_acc_stage_x3_vmem.at[buf_compute],
                            dyn_sz=tile_sz,
                            should_init=should_init_ffn2,
                        )
                        start_store_stage_a2a_s_acc_tile_to_hbm(tile_start, bd2_start, buf_compute)
                        return (buf_load, buf_compute, buf_store)

                    state = (init_buf_compute, init_buf_store, init_buf_load)
                    state = lax.fori_loop(0, num_token_tiles, run_ffn2_tile, state, unroll=False)

                    @pl.when(num_token_tiles >= 1)
                    def _():
                        wait_stage_a2a_s_acc_tile(jnp.int32(0))

                    @pl.when(num_token_tiles >= 2)
                    def _():
                        wait_stage_a2a_s_acc_tile(jnp.int32(2))

                    @pl.when(num_token_tiles >= 3)
                    def _():
                        wait_stage_a2a_s_acc_tile(jnp.int32(1))

                    return jnp.int32(next_bw_sem_id)

                return with_static_bw(bw_sem_id, body)

            return lax.fori_loop(0, num_bd2, run_down_bd2, bw_sem_id, unroll=False)

        for bf_id in range(num_bf):
            bw_sem_id = run_gate_up_slices(bf_id=bf_id, bw_sem_id=bw_sem_id)
            bw_sem_id = run_down_slices(bf_id=bf_id, bw_sem_id=bw_sem_id)

    def acc_and_store_output(*, bt_sem_id, out_buf_id):
        acc_bt = a2a_g_acc_vmem.shape[2]
        assert bt % acc_bt == 0, (bt, acc_bt)

        def load_acc_bt_sync(*, tile_start):
            def _load_one(t_i, _):
                t_id = tile_start + t_i
                for k_id in range(top_k):
                    e_id = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]
                    offset = expert_offsets_x2_smem[bt_sem_id, 1, e_id]
                    expert_offsets_x2_smem[bt_sem_id, 1, e_id] = offset + 1
                    pltpu.make_async_copy(
                        src_ref=a2a_g_hbm.at[e_id, pl.ds(offset, 1)],
                        dst_ref=a2a_g_acc_vmem.at[0, k_id, pl.ds(t_i, 1)],
                        sem=a2a_acc_sems.at[0],
                    ).start()
                return None

            lax.fori_loop(0, acc_bt, _load_one, None, unroll=False)
            pltpu.make_async_copy(
                src_ref=a2a_g_acc_vmem.at[0],
                dst_ref=a2a_g_acc_vmem.at[0],
                sem=a2a_acc_sems.at[0],
            ).wait()

        def bt_acc_acc_bt(*, tile_start, out_offset):
            # Vectorized per-(acc_bt) reduction to avoid dynamic_slice in TPU TC lowering.
            output_tile = jnp.zeros((acc_bt, t_packing, h_per_t_packing), dtype=jnp.float32)
            logits_tile = top_k_logits_vmem.at[
                pl.ds(tile_start, acc_bt),
                pl.ds(0, top_k),
            ][...]
            for k_id in range(top_k):
                acc_tile = a2a_g_acc_vmem[0, k_id, :acc_bt].astype(jnp.float32)
                logits = logits_tile[:, k_id].reshape(acc_bt, 1, 1)
                output_tile += acc_tile * logits

            out_offset = pl.multiple_of(out_offset, 16)
            b_output_x2_vmem.at[out_buf_id, pl.ds(out_offset, acc_bt), pl.ds(0, hidden_size)][
                ...
            ] = output_tile.reshape(acc_bt, hidden_size).astype(output_hbm.dtype)

        num_acc_tiles = bt // acc_bt

        def run_acc_tile(acc_tile_idx, _):
            acc_tile_start = acc_tile_idx * acc_bt
            out_offset = acc_tile_idx * acc_bt

            load_acc_bt_sync(tile_start=acc_tile_start)
            bt_acc_acc_bt(
                tile_start=acc_tile_start,
                out_offset=out_offset,
            )
            return None

        lax.fori_loop(
            0,
            num_acc_tiles,
            run_acc_tile,
            None,
            unroll=False,
        )
        return None

    def start_send_bo(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        bt_start = bt_id * bt
        b_output_sem = local_sems.at[bt_sem_id, 4]
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=output_hbm.at[pl.ds(bt_start, bt)],
            sem=b_output_sem,
        ).start(priority=priority)

    def wait_store_output(*, bt_id):
        is_valid = jnp.logical_and(bt_id >= 0, bt_id < num_bt)
        sz = pl.multiple_of(lax.select(is_valid, bt, 0), bt)
        bt_sem_id = (bt_id + 2) & 1
        pltpu.make_async_copy(
            src_ref=output_hbm.at[pl.ds(0, sz)],
            dst_ref=output_hbm.at[pl.ds(0, sz)],
            sem=local_sems.at[bt_sem_id, 4],
        ).wait()

    ### ------- Kernel start ------- ###
    sync_barrier()
    start_fetch_and_wait_bias()

    def run_per_expert(local_e_id, e_sem_id, *, bt_sem_id, bt_start):
        # Prefetch weights for CURRENT active expert.
        # TODO(jevinjiang): It is hard to prefetch weights in previous iteration
        # because the expert_ffn keeps overwriting the buffers. Triple buffering
        # could resolve this but it takes more VMEM scratch. Need further
        # experiment on this.
        start_fetch_bw1(local_e_id, bw1_sem_id=0, bf_id=0, bd1_id=0)
        start_fetch_bw3(local_e_id, bw3_sem_id=0, bf_id=0, bd3_id=0)

        # Next ids.
        next_e_sem_id = lax.select(e_sem_id == 0, 1, 0)
        next_local_e_id = local_e_id + 1

        # Start a2a scatter for NEXT active expert.
        @pl.when(next_local_e_id < local_num_experts)
        def _(
            next_e_sem_id=next_e_sem_id,
            next_local_e_id=next_local_e_id,
            bt_sem_id=bt_sem_id,
            bt_start=bt_start,
        ):
            start_a2a_scatter(
                bt_sem_id=bt_sem_id,
                e_sem_id=next_e_sem_id,
                local_e_id=next_local_e_id,
                bt_start=bt_start,
            )

        # Wait a2a scatter for CURRENT active expert.
        wait_a2a_scatter_recv(bt_sem_id=bt_sem_id, e_sem_id=e_sem_id, local_e_id=local_e_id)

        # Perform FFN for CURRENT active expert.
        expert_ffn(bt_sem_id, e_sem_id, local_e_id)

        # Start a2a gather to send back tokens for CURRENT active expert.
        start_a2a_gather(bt_sem_id=bt_sem_id, e_sem_id=e_sem_id, local_e_id=local_e_id)

        # A must-wait before sync_barrier (matches epic/integrate-fused-moe).
        wait_a2a_scatter_send(e_sem_id)
        sync_barrier()
        return next_e_sem_id

    if num_bt >= 1:
        start_fetch_b_gating(bt_id=jnp.int32(0))

    def run_bt(bt_id, e_sem_id):
        bt_start = bt_id * bt
        bt_sem_id = bt_id & jnp.int32(1)
        next_bt_id = bt_id + jnp.int32(1)

        @pl.when(next_bt_id < num_bt)
        def _():
            start_fetch_b_gating(bt_id=next_bt_id)

        wait_fetch_b_gating(bt_id=bt_id)

        b_gating = b_gating_x2_vmem.at[bt_sem_id][...]
        t2e_routing, expert_sizes, expert_starts = get_top_k(
            b_gating,
            top_k,
            renormalize_topk_logits,
            out_top_k_logits_vmem=top_k_logits_vmem,
        )

        all_reduce_metadata(
            bt_sem_id=bt_sem_id,
            t2e_routing=t2e_routing,
            starts=expert_starts,
            sizes=expert_sizes,
        )
        sync_barrier()

        # Start a2a scatter for first active expert.
        start_a2a_scatter(bt_sem_id=bt_sem_id, e_sem_id=e_sem_id, local_e_id=0, bt_start=bt_start)

        e_sem_id = lax.fori_loop(
            0,
            local_num_experts,
            lambda local_e_id, e_sem_id: run_per_expert(
                local_e_id,
                e_sem_id,
                bt_sem_id=bt_sem_id,
                bt_start=bt_start,
            ),
            e_sem_id,
            unroll=False,
        )

        # Wait to receive a2a gather for ALL experts before consuming `a2a_g_hbm`.
        wait_a2a_gather_recv_all(bt_size=bt)
        sync_barrier()

        out_buf_id = bt_id & jnp.int32(1)
        wait_store_output(bt_id=bt_id - 2)
        # Accumulate results for current bt into b_output_x2_vmem, then start async send to output_hbm.
        acc_and_store_output(bt_sem_id=bt_sem_id, out_buf_id=out_buf_id)
        start_send_bo(bt_id=bt_id)

        # Drain the last outstanding gather sends (the loop body waits `local_e_id - 2`).
        wait_a2a_gather_send(
            bt_sem_id=bt_sem_id,
            e_sem_id=e_sem_id,
            local_e_id=local_num_experts - 2,
        )
        wait_a2a_gather_send(
            bt_sem_id=bt_sem_id,
            e_sem_id=lax.select(e_sem_id == 0, 1, 0),
            local_e_id=local_num_experts - 1,
        )
        sync_barrier()
        return e_sem_id

    lax.fori_loop(0, num_bt, run_bt, jnp.int32(0), unroll=False)
    # Drain outstanding output stores (matches epic wait_send_bo for last two bts).
    wait_store_output(bt_id=jnp.int32(num_bt - 2))
    wait_store_output(bt_id=jnp.int32(num_bt - 1))

    ### ------- Kernel end ------- ###


def _validate_fused_ep_moe_args(
    *,
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    gating_output: jax.Array,
    top_k: int,
    use_grouped_topk: bool,
    num_groups: int,
    top_k_groups: int,
    bias: jax.Array | None,
    subc_quant_wsz: int | None,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w3_scale: jax.Array | None,
    b1: jax.Array | None,
    b2: jax.Array | None,
    b3: jax.Array | None,
    block_config: FusedMoEBlockConfig,
    ep_axis_name: str,
) -> None:
    if len(mesh.shape) != 2:
        raise NotImplementedError("Only 2D mesh is supported.")

    for axis_name in mesh.axis_names:
        if axis_name == ep_axis_name:
            continue
        if mesh.shape[axis_name] != 1:
            raise NotImplementedError(f"Expected all non-ep axis to have size 1 in {mesh.shape=}")

    ep_size = mesh.shape[ep_axis_name]
    num_tokens, hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape

    if w1.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"Expected {w1.shape=} to be {(num_experts, hidden_size, intermediate_size)}."
        )

    if w2.shape != (num_experts, intermediate_size, hidden_size):
        raise ValueError(
            f"Expected {w2.shape=} to be" f" {(num_experts, intermediate_size, hidden_size)}."
        )

    if w3.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"Expected {w3.shape=} to be {(num_experts, hidden_size, intermediate_size)}."
        )

    if gating_output.shape != (num_tokens, num_experts):
        raise ValueError(f"Expected {gating_output.shape=} to be {(num_tokens, num_experts)}.")

    validate_fused_moe_block_config(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=tokens.dtype,
        ep_size=ep_size,
        subc_quant_wsz=subc_quant_wsz,
        block_config=block_config,
    )

    # Note: block_config.bt is the outer expert-side token tile (routing/comm + output tiling);
    # block_config.bts is the inner token staging tile used inside expert_ffn.

    # Note: we should dump scale as the kernel expected shape in the
    # checkpoint offline or reshape right after weight loading.
    if w1_scale is not None:
        if subc_quant_wsz is None:
            raise ValueError("Expected subc_quant_wsz to be set when w1_scale is provided.")
        expected_w1_scale_shape = (
            num_experts,
            hidden_size // subc_quant_wsz,
            1,
            intermediate_size,
        )
        if w1_scale.shape != expected_w1_scale_shape:
            raise ValueError(f"Expected {w1_scale.shape=} to be {expected_w1_scale_shape}.")
        if w1_scale.dtype != jnp.float32:
            w1_scale = w1_scale.astype(jnp.float32)

    if w2_scale is not None:
        if subc_quant_wsz is None:
            raise ValueError("Expected subc_quant_wsz to be set when w2_scale is provided.")
        expected_w2_scale_shape = (
            num_experts,
            intermediate_size // subc_quant_wsz,
            1,
            hidden_size,
        )
        if w2_scale.shape != expected_w2_scale_shape:
            raise ValueError(f"Expected {w2_scale.shape=} to be {expected_w2_scale_shape}.")
        if w2_scale.dtype != jnp.float32:
            w2_scale = w2_scale.astype(jnp.float32)

    if w3_scale is not None:
        if subc_quant_wsz is None:
            raise ValueError("Expected subc_quant_wsz to be set when w3_scale is provided.")
        expected_w3_scale_shape = (
            num_experts,
            hidden_size // subc_quant_wsz,
            1,
            intermediate_size,
        )
        if w3_scale.shape != expected_w3_scale_shape:
            raise ValueError(f"Expected {w3_scale.shape=} to be {expected_w3_scale_shape}.")
        if w3_scale.dtype != jnp.float32:
            w3_scale = w3_scale.astype(jnp.float32)

    if b1 is not None:
        expected_b1_shape = (num_experts, 1, intermediate_size)
        if b1.shape != expected_b1_shape:
            raise ValueError(f"Expected {b1.shape=} to be {expected_b1_shape}.")
        if b1.dtype != jnp.float32:
            b1 = b1.astype(jnp.float32)

    if b2 is not None:
        expected_b2_shape = (num_experts, 1, hidden_size)
        if b2.shape != expected_b2_shape:
            raise ValueError(f"Expected {b2.shape=} to be {expected_b2_shape}.")

    if b3 is not None:
        expected_b3_shape = (num_experts, 1, intermediate_size)
        if b3.shape != expected_b3_shape:
            raise ValueError(f"Expected {b3.shape=} to be {expected_b3_shape}.")
        if b3.dtype != jnp.float32:
            b3 = b3.astype(jnp.float32)

    if bias is not None and bias.ndim != 1:
        raise ValueError(f"bias must be 1D, got {bias.shape}")

    if use_grouped_topk:
        if num_groups <= 0:
            raise ValueError(f"Expected num_groups > 0, got {num_groups}")
        if top_k_groups <= 0:
            raise ValueError(f"Expected top_k_groups > 0, got {top_k_groups}")
        if top_k_groups > num_groups:
            raise ValueError(
                f"top_k_groups ({top_k_groups}) cannot be larger than num_groups ({num_groups})"
            )
        num_experts = w2.shape[0]
        if num_experts % num_groups != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by num_groups ({num_groups})"
            )


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "top_k",
        "use_grouped_topk",
        "num_groups",
        "top_k_groups",
        "renormalize_topk_logits",
        "routed_scaling_factor",
        "act_fn",
        "subc_quant_wsz",
        "block_config",
        "ep_axis_name",
        "balanced_topk",
    ],
)
def fused_ep_moe(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    w3: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    use_grouped_topk: bool = False,
    num_groups: int = 1,
    top_k_groups: int = 1,
    bias: jax.Array | None = None,
    renormalize_topk_logits: bool = False,
    routed_scaling_factor: float | None = None,
    balanced_topk: bool = False,
    act_fn: str = "silu",
    subc_quant_wsz: int | None = None,
    w1_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // subc_quant_wsz, 1, intermediate_size)
    w2_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, intermediate_size // subc_quant_wsz, 1, hidden_size)
    w3_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // subc_quant_wsz, 1, intermediate_size)
    b1: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
    b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
    b3: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
    block_config: FusedMoEBlockConfig | None = None,
    ep_axis_name: str = "tensor",
):
    ep_size = mesh.shape[ep_axis_name]
    if block_config is None:
        from .tuned_block_configs import get_tuned_fused_moe_block_config

        num_tokens, hidden_size = tokens.shape
        num_experts, intermediate_size, _ = w2.shape
        block_config = get_tuned_fused_moe_block_config(
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=tokens.dtype,
            ep_size=ep_size,
        )
    block_config = block_config.effective_for(
        num_tokens=tokens.shape[0],
        ep_size=ep_size,
        dtype=tokens.dtype,
        subc_quant_wsz=subc_quant_wsz,
    )
    _validate_fused_ep_moe_args(
        mesh=mesh,
        tokens=tokens,
        w1=w1,
        w2=w2,
        w3=w3,
        gating_output=gating_output,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        num_groups=num_groups,
        top_k_groups=top_k_groups,
        bias=bias,
        subc_quant_wsz=subc_quant_wsz,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w3_scale=w3_scale,
        b1=b1,
        b2=b2,
        b3=b3,
        block_config=block_config,
        ep_axis_name=ep_axis_name,
    )

    num_devices = ep_size

    num_tokens, hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape

    local_num_tokens = num_tokens // ep_size
    bt = block_config.bt
    if bt <= 0:
        raise ValueError(f"Expected {bt=} to be > 0.")
    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    t_dtype = tokens.dtype
    gating_dtype = gating_output.dtype
    t_packing = get_dtype_packing(t_dtype)
    hidden_per_pack = hidden_size // t_packing
    # With run_bt tiling in the pallas kernel, a2a scratch only needs to cover one bt tile.
    a2a_max_tokens = align_to(bt * top_k + 1, block_config.bts)
    bd1_per_pack = block_config.bd1 // t_packing
    bd2_per_pack = block_config.bd2 // t_packing

    # Note: we should dump scale as the kernel expected shape in the
    # checkpoint offline or reshape right after weight loading.
    if w1_scale is not None and w1_scale.dtype != jnp.float32:
        w1_scale = w1_scale.astype(jnp.float32)
    if w2_scale is not None and w2_scale.dtype != jnp.float32:
        w2_scale = w2_scale.astype(jnp.float32)
    if w3_scale is not None and w3_scale.dtype != jnp.float32:
        w3_scale = w3_scale.astype(jnp.float32)
    if b1 is not None and b1.dtype != jnp.float32:
        b1 = b1.astype(jnp.float32)
    if b2 is not None and b2.dtype != jnp.float32:
        b2 = b2.astype(jnp.float32)
    if b3 is not None and b3.dtype != jnp.float32:
        b3 = b3.astype(jnp.float32)

    # Prepare inputs for the kernel.
    if padded_num_experts != gating_output.shape[-1]:
        gating_output = jnp.pad(
            gating_output,
            ((0, 0), (0, padded_num_experts - gating_output.shape[-1])),
            constant_values=-jnp.inf,
        )

    if bias is not None:
        if bias.dtype != jnp.float32:
            bias = bias.astype(jnp.float32)
        if padded_num_experts != bias.shape[0]:
            bias = jnp.pad(bias, (0, padded_num_experts - bias.shape[0]), constant_values=0.0)

    tokens = tokens.reshape(-1, t_packing, hidden_size // t_packing)

    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    renorm_str = "-renorm_k" if renormalize_topk_logits else ""
    scope_name = (
        f"fused-moe-k_{top_k}{renorm_str}"
        f"-bt_{block_config.bt}_{block_config.bts}_{block_config.btc}-bf_{block_config.bf}_{block_config.bfc}"
        f"-bd1_{block_config.bd1}_{block_config.bd1c}-bd2_{block_config.bd2}_{block_config.bd2c}"
    )
    if use_grouped_topk:
        scope_name += f"-grp_{num_groups}_{top_k_groups}"

    w1_scale_scratch = None
    if w1_scale is not None:
        assert subc_quant_wsz is not None
        w1_scale_scratch = pltpu.VMEM(
            (2, t_packing, bd1_per_pack // subc_quant_wsz, 1, block_config.bf),
            jnp.float32,
        )
    w3_scale_scratch = None
    if w3_scale is not None:
        assert subc_quant_wsz is not None
        w3_scale_scratch = pltpu.VMEM(
            (2, t_packing, bd1_per_pack // subc_quant_wsz, 1, block_config.bf),
            jnp.float32,
        )

    w2_scale_scratch = None
    if w2_scale is not None:
        assert subc_quant_wsz is not None
        w2_scale_scratch = pltpu.VMEM(
            (2, t_packing, block_config.bf // subc_quant_wsz, 1, bd2_per_pack),
            jnp.float32,
        )

    b1_scratch = None if b1 is None else pltpu.VMEM((2, 1, block_config.bf), jnp.float32)
    b3_scratch = None if b3 is None else pltpu.VMEM((2, 1, block_config.bf), jnp.float32)
    b2_scratch = None if b2 is None else pltpu.VMEM((2, t_packing, 1, bd2_per_pack), jnp.float32)
    scratch_shapes = (
        # Routing / metadata.
        pltpu.SMEM((2, bt, padded_top_k), jnp.int32),  # t2e_routing_x2_smem
        pltpu.SMEM((2, num_devices, 1, padded_num_experts), jnp.int32),  # d2e_count_x2_smem
        pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),  # expert_offsets_x2_smem
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),  # expert_starts_x2_smem
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),  # expert_sizes_x2_smem
        pltpu.SMEM((2,), jnp.int32),  # a2a_s_sends_x2_smem
        pltpu.VMEM(
            (1, top_k, math.gcd(bt, 16), t_packing, hidden_per_pack),
            t_dtype,
        ),  # a2a_g_acc_vmem
        pltpu.VMEM((bt, top_k), jnp.float32),  # top_k_logits_vmem
        # Expert compute scratch.
        pltpu.VMEM((2, bt, padded_num_experts), gating_dtype),  # b_gating_x2_vmem
        pltpu.VMEM((2, bt, hidden_size), t_dtype),  # b_output_x2_vmem
        pltpu.VMEM((2, t_packing, bd1_per_pack, block_config.bf), w1.dtype),  # b_w1_x2_vmem
        pltpu.VMEM((2, t_packing, bd1_per_pack, block_config.bf), w3.dtype),  # b_w3_x2_vmem
        pltpu.VMEM((2, t_packing, block_config.bf, bd2_per_pack), w2.dtype),  # b_w2_x2_vmem
        w1_scale_scratch,  # b_w1_scale_x2_vmem
        w3_scale_scratch,  # b_w3_scale_x2_vmem
        w2_scale_scratch,  # b_w2_scale_x2_vmem
        b1_scratch,  # b_b1_x2_vmem
        b3_scratch,  # b_b3_x2_vmem
        b2_scratch,  # b_b2_x2_vmem
        pltpu.VMEM((2, a2a_max_tokens, 1, block_config.bf), jnp.float32),  # b_acc_vmem
        pltpu.VMEM((2, block_config.bts, t_packing, bd1_per_pack), t_dtype),  # t_stage_x2_vmem
        pltpu.VMEM(
            (3, block_config.bts, t_packing, bd2_per_pack),
            t_dtype,
        ),  # a2a_s_acc_stage_x3_vmem
        (None if bias is None else pltpu.VMEM((padded_num_experts,), jnp.float32)),  # b_bias_vmem
        # Semaphores.
        pltpu.SemaphoreType.DMA((2,)),  # token_stage_x2_sems
        pltpu.SemaphoreType.DMA((3,)),  # acc_stage_x3_sems
        pltpu.SemaphoreType.DMA((2, 5)),  # local_sems
        pltpu.SemaphoreType.DMA((2,)),  # send_x2_sems
        pltpu.SemaphoreType.DMA((2,)),  # recv_x2_sems
        pltpu.SemaphoreType.DMA,  # a2a_gather_sem
        pltpu.SemaphoreType.DMA((1,)),  # a2a_acc_sems
        pltpu.SemaphoreType.BARRIER,  # barrier_sem
    )
    fused_moe = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _fused_ep_moe_kernel,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                num_groups=num_groups,
                top_k_groups=top_k_groups,
                renormalize_topk_logits=renormalize_topk_logits,
                routed_scaling_factor=routed_scaling_factor,
                balanced_topk=balanced_topk,
                ep_axis_name=ep_axis_name,
                act_fn=act_fn,
                subc_quant_wsz=subc_quant_wsz,
                bt=bt,
                bf=block_config.bf,
                bd1=block_config.bd1,
                bd2=block_config.bd2,
                bts=block_config.bts,
                btc=block_config.btc,
                bfc=block_config.bfc,
                bd1c=block_config.bd1c,
                bd2c=block_config.bd2c,
            ),
            out_shape=jax.ShapeDtypeStruct((local_num_tokens, hidden_size), t_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm_block_spec,  # tokens_hbm
                    hbm_block_spec,  # w1_hbm
                    hbm_block_spec,  # w2_hbm
                    hbm_block_spec,  # w3_hbm
                    None if w1_scale is None else hbm_block_spec,  # w1_scale_hbm
                    None if w2_scale is None else hbm_block_spec,  # w2_scale_hbm
                    None if w3_scale is None else hbm_block_spec,  # w3_scale_hbm
                    None if b1 is None else hbm_block_spec,  # b1_hbm
                    None if b2 is None else hbm_block_spec,  # b2_hbm
                    None if b3 is None else hbm_block_spec,  # b3_hbm
                    hbm_block_spec,  # gating_output_hbm
                    hbm_block_spec,  # a2a_s_x2_hbm
                    hbm_block_spec,  # a2a_s_acc_x2_hbm
                    hbm_block_spec,  # a2a_g_hbm
                    None if bias is None else hbm_block_spec,  # bias_hbm
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                allow_collective_id_without_custom_barrier=True,
                has_side_effects=True,
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
            name=scope_name,
        )
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P(ep_axis_name),  # tokens_hbm
            P(ep_axis_name),  # w1_hbm
            P(ep_axis_name),  # w2_hbm
            P(ep_axis_name),  # w3_hbm
            None if w1_scale is None else P(ep_axis_name),  # w1_scale_hbm
            None if w2_scale is None else P(ep_axis_name),  # w2_scale_hbm
            None if w3_scale is None else P(ep_axis_name),  # w3_scale_hbm
            None if b1 is None else P(ep_axis_name),  # b1_hbm
            None if b2 is None else P(ep_axis_name),  # b2_hbm
            None if b3 is None else P(ep_axis_name),  # b3_hbm
            P(ep_axis_name),  # gating_output_hbm
            P(),  # a2a_s_x2_hbm
            P(),  # a2a_s_acc_x2_hbm
            P(),  # a2a_g_hbm
            None if bias is None else P(),
        ),
        out_specs=P(ep_axis_name),
        check_vma=False,
    )
    def kernel(
        tokens,
        w1,
        w2,
        w3,
        w1_scale,
        w2_scale,
        w3_scale,
        b1,
        b2,
        b3,
        gating_output,
        a2a_s_x2_hbm_scratch,
        a2a_s_acc_x2_hbm_scratch,
        a2a_g_hbm_scratch,
        bias,
    ):
        local_output = fused_moe(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),  # tokens_hbm
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),  # w1_hbm
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),  # w2_hbm
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),  # w3_hbm
            (
                None
                if w1_scale is None
                else pltpu.with_memory_space_constraint(w1_scale, pltpu.HBM)
            ),  # w1_scale_hbm
            (
                None
                if w2_scale is None
                else pltpu.with_memory_space_constraint(w2_scale, pltpu.HBM)
            ),  # w2_scale_hbm
            (
                None
                if w3_scale is None
                else pltpu.with_memory_space_constraint(w3_scale, pltpu.HBM)
            ),  # w3_scale_hbm
            (None if b1 is None else pltpu.with_memory_space_constraint(b1, pltpu.HBM)),  # b1_hbm
            (None if b2 is None else pltpu.with_memory_space_constraint(b2, pltpu.HBM)),  # b2_hbm
            (None if b3 is None else pltpu.with_memory_space_constraint(b3, pltpu.HBM)),  # b3_hbm
            pltpu.with_memory_space_constraint(gating_output, pltpu.HBM),  # gating_output_hbm
            pltpu.with_memory_space_constraint(a2a_s_x2_hbm_scratch, pltpu.HBM),  # a2a_s_x2_hbm
            pltpu.with_memory_space_constraint(
                a2a_s_acc_x2_hbm_scratch, pltpu.HBM
            ),  # a2a_s_acc_x2_hbm
            pltpu.with_memory_space_constraint(a2a_g_hbm_scratch, pltpu.HBM),  # a2a_g_hbm
            (None if bias is None else pltpu.with_memory_space_constraint(bias, pltpu.HBM)),
        )
        return local_output

    a2a_s_x2_hbm_scratch = pl.empty(
        (2, a2a_max_tokens, t_packing, hidden_size // t_packing), t_dtype
    )
    a2a_s_acc_x2_hbm_scratch = pl.empty(
        (2, a2a_max_tokens, t_packing, hidden_size // t_packing), t_dtype
    )
    a2a_g_hbm_scratch = pl.empty((num_experts, bt, t_packing, hidden_size // t_packing), t_dtype)
    return kernel(
        tokens,
        w1,
        w2,
        w3,
        w1_scale,
        w2_scale,
        w3_scale,
        b1,
        b2,
        b3,
        gating_output,
        a2a_s_x2_hbm_scratch,
        a2a_s_acc_x2_hbm_scratch,
        a2a_g_hbm_scratch,
        bias,
    )
