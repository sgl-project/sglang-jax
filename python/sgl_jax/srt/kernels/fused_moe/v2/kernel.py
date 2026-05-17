"""Fused EP MoE v2 kernel — Strix-style double-buffer expert FFN.

Key differences from v1:
- Strix-style weight streaming: tile along intermediate dim (bf), keep tokens
  persistent in VMEM, double-buffer W1/W3/W2 with deferred W2 wait
- No bd1/bd2 hidden-dim tiling for weights — full hidden_size loaded per bf tile
- Gate/up accumulators enable W2 DMA overlap with gate/up MXU compute
- fp8 support via dequant-in-VMEM: load fp8 from HBM, dequant to bf16 in VMEM,
  then single large dot (same compute path as bf16)
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec
cdiv = pl.cdiv

_A2A_HBM_FRACTION = 0.03


@functools.lru_cache(maxsize=1)
def _device_hbm_bytes() -> int:
    return jax.local_devices()[0].memory_stats()["bytes_limit"]


# ---------------------------------------------------------------------------
# Block config
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FusedMoEBlockConfig:
    bt: int
    bf: int
    btc: int
    bse: int
    bts: int | None = None

    def effective_for(
        self,
        *,
        num_tokens: int,
        ep_size: int,
    ) -> FusedMoEBlockConfig:
        if ep_size <= 0:
            raise ValueError(f"Expected {ep_size=} > 0.")
        if num_tokens % ep_size != 0:
            raise ValueError(f"Expected {num_tokens=} divisible by {ep_size=}.")

        local_num_tokens = num_tokens // ep_size
        bt = min(self.bt, local_num_tokens)
        bt = math.gcd(bt, local_num_tokens)
        max_bts = bt * ep_size
        bts = bt if self.bts is None else min(self.bts, max_bts)
        btc = min(self.btc, bts)
        if bts % btc != 0:
            raise ValueError(f"Expected {bts=} divisible by {btc=}.")
        bse = self.bf if self.bse is None else self.bse
        return FusedMoEBlockConfig(bt=bt, bf=self.bf, btc=btc, bse=bse, bts=bts)

    def tree_flatten(self):
        bts = 0 if self.bts is None else int(self.bts)
        return (), (self.bt, self.bf, self.btc, self.bse, bts)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        bt, bf, btc, bse, bts = aux_data
        bts = None if bts == 0 else int(bts)
        return cls(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def align_to(x, a):
    return cdiv(x, a) * a


def _pad128(x):
    return ((x + 127) // 128) * 128


def get_dtype_packing(dtype):
    bits = jnp.dtype(dtype).itemsize * 8
    return 32 // bits


def swigluoai(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
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
        raise RuntimeError(f"Unsupported activation: {act_fn}")


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _align_local_tokens_for_decode(local_num_tokens: int) -> int:
    if local_num_tokens <= 0:
        raise ValueError(f"Expected {local_num_tokens=} > 0.")
    # TPU Mosaic lowering needs the token x hidden tile shape to preserve the
    # small-dimension alignment; keep decode shapes at least 8 x 128 aligned.
    if local_num_tokens <= 8:
        return 8
    return ((local_num_tokens + 7) // 8) * 8


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def ref_moe(
    tokens, w1, w2, w3, topk_weights, topk_ids, top_k, *, act_fn="silu",
    w1_shared=None, w2_shared=None, w3_shared=None,
    quant_block_k=None, w1_scale=None, w2_scale=None, w3_scale=None,
):
    num_tokens = tokens.shape[0]
    hidden_size = tokens.shape[1]
    num_experts = w1.shape[0]

    tokens_f32 = tokens.astype(jnp.float32)
    output = jnp.zeros_like(tokens_f32)

    def _dequant(w, scale, qbk):
        if scale is None:
            return w.astype(jnp.float32)
        w_f32 = w.astype(jnp.float32)
        s = jnp.repeat(scale, qbk, axis=0).squeeze(1)
        return w_f32 * s

    for t_id in range(num_tokens):
        for k_id in range(top_k):
            e_id = int(topk_ids[t_id, k_id])
            if e_id < 0 or e_id >= num_experts:
                continue
            weight = float(topk_weights[t_id, k_id])
            x = tokens_f32[t_id : t_id + 1]
            gate = x @ _dequant(w1[e_id], w1_scale[e_id] if w1_scale is not None else None, quant_block_k)
            up = x @ _dequant(w3[e_id], w3_scale[e_id] if w3_scale is not None else None, quant_block_k)
            act = activation_fn(gate, up, act_fn)
            out = act @ _dequant(w2[e_id], w2_scale[e_id] if w2_scale is not None else None, quant_block_k)
            output = output.at[t_id].add(out[0] * weight)

    if w1_shared is not None:
        gate_se = tokens_f32 @ w1_shared.astype(jnp.float32)
        up_se = tokens_f32 @ w3_shared.astype(jnp.float32)
        act_se = activation_fn(gate_se, up_se, act_fn)
        out_se = act_se @ w2_shared.astype(jnp.float32)
        output = output + out_se

    return output.astype(tokens.dtype)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_fused_moe_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    ep_size: int,
    block_config: FusedMoEBlockConfig,
):
    bc = block_config.effective_for(num_tokens=num_tokens, ep_size=ep_size)
    bt, bf, btc, bse = bc.bt, bc.bf, bc.btc, bc.bse
    bts = bc.bts

    local_num_tokens = num_tokens // ep_size
    if local_num_tokens % bt != 0:
        raise ValueError(f"{local_num_tokens=} must be divisible by {bt=}.")
    if intermediate_size % bf != 0:
        raise ValueError(f"{intermediate_size=} must be divisible by {bf=}.")
    if hidden_size % 128 != 0:
        raise ValueError(f"{hidden_size=} must be aligned to 128.")
    if bf % 128 != 0:
        raise ValueError(f"{bf=} must be aligned to 128.")
    if btc % 8 != 0:
        raise ValueError(f"{btc=} must be aligned to 8 (VREG sublane).")
    if bts % btc != 0:
        raise ValueError(f"{bts=} must be divisible by {btc=}.")


def get_ep_size(mesh: jax.sharding.Mesh, dp_axis_name, tp_axis_name):
    dp_size = mesh.shape[dp_axis_name]
    tp_size = mesh.shape[tp_axis_name]
    return dp_size * tp_size


# ---------------------------------------------------------------------------
# Pallas kernel
# ---------------------------------------------------------------------------

def _fused_ep_moe_kernel(
    # --- HBM inputs ---
    tokens_hbm,           # (local_num_tokens, t_packing, h_per_t)
    w1_hbm,               # (local_num_experts, hidden_size, intermediate_size)
    w2_hbm,               # (local_num_experts, intermediate_size, hidden_size)
    w3_hbm,               # (local_num_experts, hidden_size, intermediate_size)
    w1_scale_hbm,         # None | (local_num_experts, H // quant_block_k, 1, I)
    w2_scale_hbm,         # None | (local_num_experts, I // quant_block_k, 1, H)
    w3_scale_hbm,         # None | (local_num_experts, H // quant_block_k, 1, I)
    topk_weights_hbm,     # (local_num_tokens, top_k)
    topk_ids_hbm,         # (local_num_tokens, top_k)
    a2a_s_x2_hbm,         # (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t)
    a2a_s_acc_x2_hbm,     # (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t)
    a2a_g_hbm,            # (num_experts, bt, t_packing, h_per_t)
    w1_shared_hbm,        # None | (hidden_size, se_intermediate_size)
    w3_shared_hbm,        # None | (hidden_size, se_intermediate_size)
    w2_shared_hbm,        # None | (se_intermediate_size, hidden_size)
    metadata_starts_hbm,  # None | (num_bt, 1, padded_num_experts) int32
    metadata_sizes_hbm,   # None | (num_bt, 1, padded_num_experts) int32
    metadata_d2e_counts_hbm,  # None | (num_bt, num_devices, 1, padded_num_experts) int32
    # --- HBM output ---
    output_hbm,           # (local_num_tokens, hidden_size)
    # --- SMEM scratch ---
    t2e_routing_x2_smem,       # (2, bt, padded_top_k)
    d2e_count_x2_smem,         # (2, num_devices, 1, padded_num_experts)
    expert_offsets_x2_smem,    # (2, 2, padded_num_experts)
    expert_starts_x2_smem,    # (2, 1, padded_num_experts)
    expert_sizes_x2_smem,     # (2, 1, padded_num_experts)
    a2a_s_sends_x2_smem,      # (expert_buffer_count,)
    # --- VMEM scratch ---
    a2a_g_acc_vmem,        # (2, top_k, acc_bt, t_packing, h_per_t)
    b_topk_weights_x2_vmem,  # (2, bt, padded_top_k)
    b_topk_ids_x2_vmem,      # (2, bt, padded_top_k)
    b_output_x2_vmem,        # (2, bt, hidden_size)
    # Weight double buffers — (2, t_packing, h_per_t, bf) or (2, t_packing, bf, h_per_t)
    b_w1_x2_vmem,          # (2, t_packing, h_per_t, bf)
    b_w3_x2_vmem,          # (2, t_packing, h_per_t, bf)
    b_w2_x2_vmem,          # (2, t_packing, bf, h_per_t)
    # Scale double buffers (None when not quantized)
    b_w1_scale_x2_vmem,    # None | (2, t_packing, h_per_t // qbk, 1, bf) f32
    b_w3_scale_x2_vmem,    # None | (2, t_packing, h_per_t // qbk, 1, bf) f32
    b_w2_scale_x2_vmem,    # None | (2, t_packing, bf // qbk, 1, h_per_t) f32
    # Dequant scratch (single-buf, populated after DMA wait, None when not quantized)
    b_w1_dq_vmem,          # None | (t_packing, h_per_t, bf) bf16
    b_w3_dq_vmem,          # None | (t_packing, h_per_t, bf) bf16
    b_w2_dq_vmem,          # None | (t_packing, bf, h_per_t) bf16
    # Gate/up accumulators (per bts tile)
    b_gate_acc_vmem,       # (bts, bf) f32
    b_up_acc_vmem,         # (bts, bf) f32
    # Token staging per bts tile
    b_x_vmem,              # (bts, t_packing, h_per_t) bf16
    # Output accumulator per bts tile
    b_y_acc_vmem,          # (bts, t_packing, h_per_t) f32
    # Output staging for HBM read-modify-write per bts tile
    b_y_stage_vmem,        # (bts, t_packing, h_per_t) bf16
    # Shared expert buffers
    b_se_tokens_vmem,      # None | (2, 2, bt, t_packing, h_per_t)
    b_se_w1_x2_vmem,       # None | (2, t_packing, h_per_t, bse)
    b_se_w3_x2_vmem,       # None | (2, t_packing, h_per_t, bse)
    b_se_w2_x2_vmem,       # None | (2, t_packing, bse, h_per_t)
    b_se_acc_vmem,         # None | (2, bt, hidden_size) f32
    # --- Semaphores ---
    x_stage_sem,           # DMA(1,) — token staging
    y_store_sem,           # DMA(1,) — output store from y_acc
    local_sems,            # DMA(2, 10) — weight + topk + output + metadata
    send_x2_sems,         # DMA(expert_buffer_count,)
    recv_x2_sems,         # DMA(expert_buffer_count,)
    gather_send_x2_sems,  # DMA(expert_buffer_count,)
    a2a_gather_sem,        # DMA scalar
    a2a_acc_sems,          # DMA(1,)
    barrier_sem,           # BARRIER
    *,
    # Static params
    top_k: int,
    dp_axis_name: str,
    tp_axis_name: str,
    act_fn: str,
    disable_a2a: bool = False,
    disable_shared_expert: bool = False,
    disable_sync_barrier: bool = False,
    disable_weight_load: bool = False,
    disable_dynamic_ffn1: bool = False,
    disable_dynamic_ffn2: bool = False,
    disable_acc_and_store: bool = False,
    use_jax_allreduce_metadata: bool = True,
    decode_mode: bool = False,
    direct_scaled_dot: bool = False,
    skip_decode_sync_barrier: bool = False,
    bt: int,
    bf: int,
    btc: int,
    bts: int,
    bse: int,
    quant_block_k: int | None = None,
):
    # ===== Dimension extraction =====
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = tp_size * dp_size
    local_num_tokens = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    expert_buffer_count = a2a_s_x2_hbm.shape[0]
    assert expert_buffer_count >= 1
    assert expert_buffer_count <= local_num_experts
    assert local_num_tokens % bt == 0
    num_bt = local_num_tokens // bt
    a2a_max_tokens = a2a_s_x2_hbm.shape[1]
    num_experts = a2a_g_hbm.shape[0]
    padded_num_experts = d2e_count_x2_smem.shape[-1]
    padded_top_k = t2e_routing_x2_smem.shape[-1]
    assert padded_num_experts == align_to(num_experts, 128)
    assert padded_top_k == align_to(top_k, 128)

    t_dtype = tokens_hbm.dtype
    t_packing = get_dtype_packing(t_dtype)
    h_per_t = hidden_size // t_packing

    num_bf = cdiv(intermediate_size, bf)

    n_sg = h_per_t // quant_block_k if quant_block_k is not None else 1
    n_sg2 = bf // quant_block_k if quant_block_k is not None else 1

    se_inter_size = 0
    se_total_blocks = 0
    if w1_shared_hbm is not None:
        se_inter_size = w2_shared_hbm.shape[0]
        se_total_blocks = cdiv(se_inter_size, bse)

    # ===== Mesh device ID — returns tuple for DeviceIdType.MESH =====
    def get_mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    # ===== Sync barrier — signal ALL devices =====
    skip_sync_barrier = skip_decode_sync_barrier and num_bt == 1

    def sync_barrier(*, force: bool = False):
        if disable_sync_barrier or (skip_sync_barrier and not force):
            return
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    # ===== Topk fetch/wait =====
    def start_fetch_topk(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        bt_start = bt_id * bt
        pltpu.make_async_copy(
            src_ref=topk_weights_hbm.at[pl.ds(bt_start, bt)],
            dst_ref=b_topk_weights_x2_vmem.at[bt_sem_id, pl.ds(0, bt)],
            sem=local_sems.at[bt_sem_id, 0],
        ).start(priority=priority)
        pltpu.make_async_copy(
            src_ref=topk_ids_hbm.at[pl.ds(bt_start, bt)],
            dst_ref=b_topk_ids_x2_vmem.at[bt_sem_id, pl.ds(0, bt)],
            sem=local_sems.at[bt_sem_id, 0],
        ).start(priority=priority)

    def wait_fetch_topk(*, bt_id):
        bt_sem_id = bt_id & jnp.int32(1)
        pltpu.make_async_copy(
            src_ref=b_topk_weights_x2_vmem.at[bt_sem_id],
            dst_ref=b_topk_weights_x2_vmem.at[bt_sem_id],
            sem=local_sems.at[bt_sem_id, 0],
        ).wait()
        pltpu.make_async_copy(
            src_ref=b_topk_ids_x2_vmem.at[bt_sem_id],
            dst_ref=b_topk_ids_x2_vmem.at[bt_sem_id],
            sem=local_sems.at[bt_sem_id, 0],
        ).wait()

    # ===== All-reduce metadata =====
    # Copies routing + metadata into SMEM via VMEM staging (HBM→VMEM→SMEM).
    def all_reduce_metadata(*, bt_id, bt_sem_id, t2e_routing):
        if disable_a2a:
            return

        offsets_sem = local_sems.at[bt_sem_id, 8]
        routing_sem = local_sems.at[bt_sem_id, 9]

        if use_jax_allreduce_metadata and metadata_starts_hbm is not None:
            def _copy_precomputed(
                t2e_routing_vmem,
                d2e_count_vmem,
                offsets_vmem,
                starts_vmem,
                sizes_vmem,
            ):
                offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
                t2e_routing_vmem[...] = t2e_routing

                starts_load = pltpu.async_copy(
                    src_ref=metadata_starts_hbm.at[bt_id],
                    dst_ref=starts_vmem,
                    sem=local_sems.at[bt_sem_id, 1],
                )
                sizes_load = pltpu.async_copy(
                    src_ref=metadata_sizes_hbm.at[bt_id],
                    dst_ref=sizes_vmem,
                    sem=local_sems.at[bt_sem_id, 2],
                )
                d2e_count_load = pltpu.async_copy(
                    src_ref=metadata_d2e_counts_hbm.at[bt_id],
                    dst_ref=d2e_count_vmem,
                    sem=local_sems.at[bt_sem_id, 3],
                )

                offsets_copy = pltpu.async_copy(
                    src_ref=offsets_vmem,
                    dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                    sem=offsets_sem,
                )
                t2e_routing_copy = pltpu.async_copy(
                    src_ref=t2e_routing_vmem,
                    dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                    sem=routing_sem,
                )

                starts_load.wait()
                sizes_load.wait()
                d2e_count_load.wait()
                starts_copy = pltpu.async_copy(
                    src_ref=starts_vmem,
                    dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                    sem=local_sems.at[bt_sem_id, 1],
                )
                sizes_copy = pltpu.async_copy(
                    src_ref=sizes_vmem,
                    dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                    sem=local_sems.at[bt_sem_id, 2],
                )
                d2e_count_copy = pltpu.async_copy(
                    src_ref=d2e_count_vmem,
                    dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                    sem=local_sems.at[bt_sem_id, 3],
                )

                t2e_routing_copy.wait()
                offsets_copy.wait()
                starts_copy.wait()
                sizes_copy.wait()
                d2e_count_copy.wait()

            pl.run_scoped(
                _copy_precomputed,
                pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
                pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
                pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
                pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
                pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
            )
            return

        # --- In-kernel metadata allreduce (no JAX-level lax.all_gather) ---
        md_send_sem = send_x2_sems.at[0]
        md_recv_sem = recv_x2_sems.at[0]

        def _inkernel_allreduce(
            t2e_routing_vmem,
            d2e_count_vmem,
            offsets_vmem,
            starts_vmem,
            sizes_vmem,
        ):
            offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=offsets_sem,
            )
            t2e_routing_vmem[...] = t2e_routing
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=routing_sem,
            )

            expert_iota = lax.broadcasted_iota(
                jnp.int32, (1, 1, padded_num_experts), 2,
            )
            routing_expanded = jnp.expand_dims(
                t2e_routing[:, :top_k], axis=2,
            )
            mask = (routing_expanded == expert_iota).astype(jnp.int32)
            local_sizes = jnp.sum(
                mask, axis=(0, 1), keepdims=True,
            ).reshape(1, padded_num_experts)

            d2e_count_vmem[...] = jnp.zeros_like(d2e_count_vmem)
            d2e_count_vmem[my_id] = local_sizes

            sync_barrier(force=True)

            if num_devices > 0 and (num_devices & (num_devices - 1)) == 0:
                rounds = int(math.log2(num_devices))
                for round_id in range(rounds):
                    sync_barrier(force=True)

                    chunk = 1 << round_id
                    chunk_i32 = jnp.int32(chunk)
                    peer_id = my_id ^ chunk_i32

                    send_start = (my_id >> round_id) << round_id
                    recv_start = (peer_id >> round_id) << round_id

                    pltpu.make_async_remote_copy(
                        src_ref=d2e_count_vmem.at[
                            pl.ds(send_start, chunk),
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        dst_ref=d2e_count_vmem.at[
                            pl.ds(send_start, chunk),
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        send_sem=md_send_sem,
                        recv_sem=md_recv_sem,
                        device_id=get_mesh_device_id(peer_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                    recv_ref = d2e_count_vmem.at[
                        pl.ds(recv_start, chunk),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=recv_ref, dst_ref=recv_ref,
                        sem=md_recv_sem,
                    ).wait()

                    send_ref = d2e_count_vmem.at[
                        pl.ds(send_start, chunk),
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=send_ref, dst_ref=send_ref,
                        sem=md_send_sem,
                    ).wait()
            else:
                for step in range(1, num_devices):
                    peer_id = (my_id + step) % num_devices
                    pltpu.make_async_remote_copy(
                        src_ref=d2e_count_vmem.at[
                            my_id,
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        dst_ref=d2e_count_vmem.at[
                            my_id,
                            pl.ds(0, 1),
                            pl.ds(0, padded_num_experts),
                        ],
                        send_sem=md_send_sem,
                        recv_sem=md_recv_sem,
                        device_id=get_mesh_device_id(peer_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                    src_peer = (my_id + num_devices - step) % num_devices
                    recv_ref = d2e_count_vmem.at[
                        src_peer,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=recv_ref, dst_ref=recv_ref,
                        sem=md_recv_sem,
                    ).wait()

                    send_ref = d2e_count_vmem.at[
                        my_id,
                        pl.ds(0, 1),
                        pl.ds(0, padded_num_experts),
                    ]
                    pltpu.make_async_copy(
                        src_ref=send_ref, dst_ref=send_ref,
                        sem=md_send_sem,
                    ).wait()

            sync_barrier(force=True)

            reduced_sizes = jnp.zeros((1, padded_num_experts), dtype=jnp.int32)
            reduced_starts = jnp.zeros((1, padded_num_experts), dtype=jnp.int32)
            for dev_id in range(num_devices):
                dev_sizes = d2e_count_vmem[dev_id]
                reduced_sizes += dev_sizes
                reduced_starts += lax.select(
                    dev_id < my_id, dev_sizes,
                    jnp.zeros_like(dev_sizes),
                )

            starts_vmem[...] = reduced_starts
            sizes_vmem[...] = reduced_sizes

            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 1],
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 2],
            )
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 3],
            )

            t2e_routing_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()
            d2e_count_copy.wait()

        pl.run_scoped(
            _inkernel_allreduce,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
        )

    # ===== A2A scatter (batch — all experts at once) =====

    def start_a2a_scatter_batch(*, bt_sem_id, bt_start):
        if disable_a2a:
            return
        for slot in range(expert_buffer_count):
            a2a_s_sends_x2_smem[slot] = jnp.int32(0)

        def _scatter_one_batch(t_id, _, bt_start=bt_start):
            src_t_id = bt_start + t_id
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]
                is_valid = e_id >= 0
                e_id_safe = lax.select(is_valid, e_id, jnp.int32(0))
                e_sem_id_k = e_id_safe % jnp.int32(local_num_experts)
                recv_id = e_id_safe // local_num_experts
                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id_safe]
                sz = lax.select(is_valid, jnp.int32(1), jnp.int32(0))
                is_local = recv_id == my_id
                local_sz = lax.select(is_local, sz, jnp.int32(0))
                remote_sz = lax.select(is_local, jnp.int32(0), sz)
                expert_offsets_x2_smem[bt_sem_id, 0, e_id_safe] = offset + local_sz + remote_sz
                start = expert_starts_x2_smem[bt_sem_id, 0, e_id_safe] + offset
                cur_sends = a2a_s_sends_x2_smem[e_sem_id_k]
                a2a_s_sends_x2_smem[e_sem_id_k] = cur_sends + remote_sz

                @pl.when(local_sz != 0)
                def _local_copy(
                    src_t_id=src_t_id, start=start, local_sz=local_sz, e_sem_id_k=e_sem_id_k
                ):
                    pltpu.make_async_copy(
                        src_ref=tokens_hbm.at[pl.ds(src_t_id, local_sz)],
                        dst_ref=a2a_s_x2_hbm.at[e_sem_id_k, pl.ds(start, local_sz)],
                        sem=recv_x2_sems.at[e_sem_id_k],
                    ).start()

                @pl.when(remote_sz != 0)
                def _remote_copy(
                    src_t_id=src_t_id,
                    start=start,
                    remote_sz=remote_sz,
                    e_sem_id_k=e_sem_id_k,
                    recv_id=recv_id,
                ):
                    pltpu.make_async_remote_copy(
                        src_ref=tokens_hbm.at[pl.ds(src_t_id, remote_sz)],
                        dst_ref=a2a_s_x2_hbm.at[e_sem_id_k, pl.ds(start, remote_sz)],
                        send_sem=send_x2_sems.at[e_sem_id_k],
                        recv_sem=recv_x2_sems.at[e_sem_id_k],
                        device_id=get_mesh_device_id(recv_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

            return None

        lax.fori_loop(0, bt, _scatter_one_batch, None, unroll=False)

    # ===== A2A scatter (pipelined — one expert at a time) =====

    def start_a2a_scatter(*, bt_sem_id, e_sem_id, local_e_id, bt_start):
        if disable_a2a:
            return
        e_id = my_id * local_num_experts + local_e_id

        def _scatter_one(t_id, _, bt_start=bt_start, e_id=e_id):
            src_t_id = bt_start + t_id
            for k_id in range(top_k):
                expert_of_k = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]

                @pl.when(expert_of_k == e_id)
                def _():
                    offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id]
                    expert_offsets_x2_smem[bt_sem_id, 0, e_id] = offset + 1
                    start = expert_starts_x2_smem[bt_sem_id, 0, e_id] + offset
                    target_device = e_id // local_num_experts
                    is_local = target_device == my_id
                    sz = jnp.int32(1)
                    local_sz = lax.select(is_local, sz, jnp.int32(0))
                    remote_sz = lax.select(is_local, jnp.int32(0), sz)
                    cur_sends = a2a_s_sends_x2_smem[e_sem_id]
                    a2a_s_sends_x2_smem[e_sem_id] = cur_sends + remote_sz

                    @pl.when(local_sz != 0)
                    def _local_copy():
                        pltpu.make_async_copy(
                            src_ref=tokens_hbm.at[pl.ds(src_t_id, 1)],
                            dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(start, 1)],
                            sem=recv_x2_sems.at[e_sem_id],
                        ).start()

                    @pl.when(remote_sz != 0)
                    def _remote_copy():
                        pltpu.make_async_remote_copy(
                            src_ref=tokens_hbm.at[pl.ds(src_t_id, 1)],
                            dst_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(start, 1)],
                            send_sem=send_x2_sems.at[e_sem_id],
                            recv_sem=recv_x2_sems.at[e_sem_id],
                            device_id=get_mesh_device_id(target_device),
                            device_id_type=pltpu.DeviceIdType.MESH,
                        ).start()

            return None

        lax.fori_loop(0, bt, _scatter_one, None, unroll=False)

    # ===== A2A scatter wait =====

    def wait_a2a_scatter_send_batch():
        if disable_a2a:
            return

        def _wait_one(slot, _):
            scatter_send_sz = a2a_s_sends_x2_smem[slot]

            @pl.when(scatter_send_sz != 0)
            def _():
                ref = a2a_s_x2_hbm.at[slot, pl.ds(0, scatter_send_sz)]
                pltpu.make_async_copy(
                    src_ref=ref, dst_ref=ref, sem=send_x2_sems.at[slot],
                ).wait()

            return None

        lax.fori_loop(0, jnp.int32(expert_buffer_count), _wait_one, None, unroll=False)

    def wait_a2a_scatter_recv(*, bt_sem_id, e_sem_id, local_e_id):
        if disable_a2a:
            return
        e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]

        @pl.when(sz != 0)
        def _():
            ref = a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, sz)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=recv_x2_sems.at[e_sem_id],
            ).wait()

    def wait_a2a_scatter_send(*, bt_sem_id, e_sem_id, local_e_id):
        if disable_a2a:
            return
        scatter_send_sz = a2a_s_sends_x2_smem[e_sem_id]

        @pl.when(scatter_send_sz != 0)
        def _():
            ref = a2a_s_x2_hbm.at[e_sem_id, pl.ds(0, scatter_send_sz)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=send_x2_sems.at[e_sem_id],
            ).wait()

    # ===== A2A gather (static for loop, prefix sum, dst position 0) =====

    def start_a2a_gather(*, bt_sem_id, e_sem_id, local_e_id):
        if disable_a2a:
            return
        my_e_id = my_id * local_num_experts + local_e_id
        src_ref = a2a_s_acc_x2_hbm
        start = 0
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, 0)
            remote_sz = lax.select(is_local, 0, sz)

            @pl.when(local_sz != 0)
            def _local_copy(
                start=start,
                local_sz=local_sz,
                my_e_id=my_e_id,
                e_sem_id=e_sem_id,
            ):
                pltpu.make_async_copy(
                    src_ref=src_ref.at[e_sem_id, pl.ds(start, local_sz)],
                    dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                    sem=a2a_gather_sem,
                ).start()

            @pl.when(remote_sz != 0)
            def _remote_copy(
                start=start,
                remote_sz=remote_sz,
                my_e_id=my_e_id,
                e_sem_id=e_sem_id,
                recv_id=recv_id,
            ):
                pltpu.make_async_remote_copy(
                    src_ref=src_ref.at[e_sem_id, pl.ds(start, remote_sz)],
                    dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                    send_sem=gather_send_x2_sems.at[e_sem_id],
                    recv_sem=a2a_gather_sem,
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            start += sz

    def wait_a2a_gather_send(*, bt_sem_id, e_sem_id, local_e_id):
        if disable_a2a or num_devices <= 1:
            return
        my_e_id = my_id * local_num_experts + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(local_e_id >= 0, local_e_id < local_num_experts)
        remote_sz = lax.select(is_valid, remote_sz, 0)

        @pl.when(remote_sz != 0)
        def _():
            ref = a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(0, remote_sz)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=gather_send_x2_sems.at[e_sem_id],
            ).wait()

    def wait_a2a_gather_recv_all(*, bt_sem_id):
        if disable_a2a:
            return

        def _wait_one_expert(e_id, _):
            sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, e_id]

            @pl.when(sz != 0)
            def _():
                ref = a2a_g_hbm.at[e_id, pl.ds(0, sz)]
                pltpu.make_async_copy(
                    src_ref=ref, dst_ref=ref, sem=a2a_gather_sem,
                ).wait()
            return None

        lax.fori_loop(0, num_experts, _wait_one_expert, None, unroll=False)

    # ===== Weight DMA (per-t_packing loop, matching v1 pattern) =====

    def start_fetch_w1(local_e_id, slot, bf_id, priority=1):
        if disable_weight_load:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w1_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w1_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 4],
            ).start(priority=priority)
            if w1_scale_hbm is not None:
                pltpu.make_async_copy(
                    src_ref=w1_scale_hbm.at[
                        local_e_id,
                        pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w1_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 4],
                ).start(priority=priority)

    def wait_fetch_w1(slot):
        if disable_weight_load:
            return
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[slot],
            dst_ref=b_w1_x2_vmem.at[slot],
            sem=local_sems.at[slot, 4],
        ).wait()
        if w1_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w1_scale_x2_vmem.at[slot],
                dst_ref=b_w1_scale_x2_vmem.at[slot],
                sem=local_sems.at[slot, 4],
            ).wait()

    def start_fetch_w3(local_e_id, slot, bf_id, priority=1):
        if disable_weight_load:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w3_hbm.at[
                    local_e_id,
                    pl.ds(p * h_per_t, h_per_t),
                    pl.ds(bf_id * bf, bf),
                ],
                dst_ref=b_w3_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 5],
            ).start(priority=priority)
            if w3_scale_hbm is not None:
                pltpu.make_async_copy(
                    src_ref=w3_scale_hbm.at[
                        local_e_id,
                        pl.ds(p * h_per_t // quant_block_k, h_per_t // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(bf_id * bf, bf),
                    ],
                    dst_ref=b_w3_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 5],
                ).start(priority=priority)

    def wait_fetch_w3(slot):
        if disable_weight_load:
            return
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[slot],
            dst_ref=b_w3_x2_vmem.at[slot],
            sem=local_sems.at[slot, 5],
        ).wait()
        if w3_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w3_scale_x2_vmem.at[slot],
                dst_ref=b_w3_scale_x2_vmem.at[slot],
                sem=local_sems.at[slot, 5],
            ).wait()

    def start_fetch_w2(local_e_id, slot, bf_id, priority=0):
        if disable_weight_load:
            return
        for p in range(t_packing):
            pltpu.make_async_copy(
                src_ref=w2_hbm.at[
                    local_e_id,
                    pl.ds(bf_id * bf, bf),
                    pl.ds(p * h_per_t, h_per_t),
                ],
                dst_ref=b_w2_x2_vmem.at[slot, p],
                sem=local_sems.at[slot, 6],
            ).start(priority=priority)
            if w2_scale_hbm is not None:
                pltpu.make_async_copy(
                    src_ref=w2_scale_hbm.at[
                        local_e_id,
                        pl.ds(bf_id * bf // quant_block_k, bf // quant_block_k),
                        pl.ds(0, 1),
                        pl.ds(p * h_per_t, h_per_t),
                    ],
                    dst_ref=b_w2_scale_x2_vmem.at[slot, p],
                    sem=local_sems.at[slot, 6],
                ).start(priority=priority)

    def wait_fetch_w2(slot):
        if disable_weight_load:
            return
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[slot],
            dst_ref=b_w2_x2_vmem.at[slot],
            sem=local_sems.at[slot, 6],
        ).wait()
        if w2_scale_hbm is not None:
            pltpu.make_async_copy(
                src_ref=b_w2_scale_x2_vmem.at[slot],
                dst_ref=b_w2_scale_x2_vmem.at[slot],
                sem=local_sems.at[slot, 6],
            ).wait()

    # ===== Dequant fp8 → bf16 in VMEM (after DMA wait, before dot) =====

    def dequant_w1(slot):
        if w1_scale_hbm is None or direct_scaled_dot:
            return
        for p in range(t_packing):
            def _dq_w1(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w1_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                s = b_w1_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
                s = s.reshape(1, bf)
                w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, bf))).astype(jnp.bfloat16)
                b_w1_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg, _dq_w1, None, unroll=n_sg)

    def dequant_w3(slot):
        if w3_scale_hbm is None or direct_scaled_dot:
            return
        for p in range(t_packing):
            def _dq_w3(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w3_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)]
                s = b_w3_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, bf)]
                s = s.reshape(1, bf)
                w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, bf))).astype(jnp.bfloat16)
                b_w3_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, bf)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg, _dq_w3, None, unroll=n_sg)

    def dequant_w2(slot):
        if w2_scale_hbm is None or direct_scaled_dot:
            return
        for p in range(t_packing):
            def _dq_w2(sg_id, _):
                sg_off = sg_id * quant_block_k
                w_fp8 = b_w2_x2_vmem[slot, p, pl.ds(sg_off, quant_block_k), pl.ds(0, h_per_t)]
                s = b_w2_scale_x2_vmem[slot, p, pl.ds(sg_id, 1), 0, pl.ds(0, h_per_t)]
                s = s.reshape(1, h_per_t)
                w_bf16 = (w_fp8.astype(jnp.float32) * jnp.broadcast_to(s, (quant_block_k, h_per_t))).astype(jnp.bfloat16)
                b_w2_dq_vmem.at[p, pl.ds(sg_off, quant_block_k), pl.ds(0, h_per_t)][...] = w_bf16
                return None
            lax.fori_loop(0, n_sg2, _dq_w2, None, unroll=n_sg2)

    # ===== Expert FFN: Strix-style double-buffer pipeline =====

    def expert_ffn(bt_sem_id, e_sem_id, local_e_id):
        e_id = my_id * local_num_experts + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id]
        has_tokens = dyn_sz > 0

        def _run_inactive(_):
            @pl.when(local_e_id >= expert_buffer_count)
            def _():
                wait_a2a_gather_send(
                    bt_sem_id=bt_sem_id,
                    e_sem_id=e_sem_id,
                    local_e_id=local_e_id - expert_buffer_count,
                )
            return jnp.int32(0)

        def _run_active(_):

            dyn_sz_i32 = dyn_sz.astype(jnp.int32)
            num_bts_tiles = (dyn_sz_i32 + (bts - 1)) // bts
            num_btc_per_bts = bts // btc

            # bts loop is OUTER (dynamic), bf loop is INNER (static unroll).
            # Tokens load once per bts tile, reused across all bf tiles.
            # Weights re-prefetch per bts tile (redundant when num_bts_tiles=1,
            # which is the common case — avg ~20 tokens per expert).
            def bts_body(bts_id, __):
                tile_start = bts_id * bts

                # Load tokens for this bts tile (once, reused across all bf tiles)
                pltpu.make_async_copy(
                    src_ref=a2a_s_x2_hbm.at[e_sem_id, pl.ds(tile_start, bts)],
                    dst_ref=b_x_vmem,
                    sem=x_stage_sem.at[0],
                ).start(priority=1)
                pltpu.make_async_copy(
                    src_ref=b_x_vmem, dst_ref=b_x_vmem,
                    sem=x_stage_sem.at[0],
                ).wait()

                # Weight prologue (double-buffer only)
                if not decode_mode:
                    start_fetch_w1(local_e_id, 0, 0, priority=1)
                    start_fetch_w3(local_e_id, 0, 0, priority=1)
                    start_fetch_w2(local_e_id, 0, 0, priority=1)
                    if num_bf >= 2:
                        start_fetch_w1(local_e_id, 1, 1)
                        start_fetch_w3(local_e_id, 1, 1)
                        start_fetch_w2(local_e_id, 1, 1, priority=1)

                for bf_id in range(num_bf):
                    if decode_mode:
                        slot = 0
                        start_fetch_w1(local_e_id, 0, bf_id, priority=1)
                        start_fetch_w3(local_e_id, 0, bf_id, priority=1)
                        start_fetch_w2(local_e_id, 0, bf_id, priority=1)
                    else:
                        slot = bf_id % 2

                    wait_fetch_w1(slot)
                    wait_fetch_w3(slot)

                    dequant_w1(slot)
                    dequant_w3(slot)

                    # Gate/up
                    def gate_up_btc(btc_id, ___):
                        gate = jnp.zeros((btc, bf), dtype=jnp.float32)
                        up = jnp.zeros((btc, bf), dtype=jnp.float32)
                        if not disable_dynamic_ffn1:
                            if direct_scaled_dot and w1_scale_hbm is not None:
                                for p_id in range(t_packing):
                                    def _ffn1_sg_body(sg_id, carry):
                                        gate_acc, up_acc = carry
                                        sg_off = sg_id * quant_block_k
                                        x_slice = b_x_vmem[
                                            pl.ds(btc_id * btc, btc),
                                            p_id,
                                            pl.ds(sg_off, quant_block_k),
                                        ]
                                        w1_tile = b_w1_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_off, quant_block_k),
                                            pl.ds(0, bf),
                                        ]
                                        w3_tile = b_w3_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_off, quant_block_k),
                                            pl.ds(0, bf),
                                        ]
                                        d1 = jnp.dot(
                                            x_slice, w1_tile,
                                            preferred_element_type=jnp.float32,
                                        )
                                        s1 = b_w1_scale_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_id, 1),
                                            0,
                                            pl.ds(0, bf),
                                        ].reshape(1, bf)
                                        gate_acc += d1 * jnp.broadcast_to(s1, d1.shape)

                                        d3 = jnp.dot(
                                            x_slice, w3_tile,
                                            preferred_element_type=jnp.float32,
                                        )
                                        s3 = b_w3_scale_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_id, 1),
                                            0,
                                            pl.ds(0, bf),
                                        ].reshape(1, bf)
                                        up_acc += d3 * jnp.broadcast_to(s3, d3.shape)
                                        return gate_acc, up_acc

                                    gate, up = lax.fori_loop(
                                        0, n_sg, _ffn1_sg_body, (gate, up), unroll=n_sg,
                                    )
                            else:
                                for p_id in range(t_packing):
                                    x_slice = b_x_vmem[
                                        pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                                    ]
                                    w1_tile = b_w1_dq_vmem[p_id] if w1_scale_hbm is not None else b_w1_x2_vmem[slot, p_id]
                                    w3_tile = b_w3_dq_vmem[p_id] if w3_scale_hbm is not None else b_w3_x2_vmem[slot, p_id]
                                    gate += jnp.dot(x_slice, w1_tile, preferred_element_type=jnp.float32)
                                    up += jnp.dot(x_slice, w3_tile, preferred_element_type=jnp.float32)
                        b_gate_acc_vmem.at[pl.ds(btc_id * btc, btc), pl.ds(0, bf)][...] = gate
                        b_up_acc_vmem.at[pl.ds(btc_id * btc, btc), pl.ds(0, bf)][...] = up
                        return None

                    lax.fori_loop(0, num_btc_per_bts, gate_up_btc, None)

                    # W2 DMA is started with W1/W3. Defer its wait until the
                    # down projection so W2 transfer can overlap gate/up.
                    wait_fetch_w2(slot)
                    dequant_w2(slot)

                    # Act+down — accumulate in VMEM f32 across bf tiles
                    def act_down_btc(btc_id, ___):
                        use_direct_w2 = direct_scaled_dot and w2_scale_hbm is not None
                        if not use_direct_w2:
                            gate = b_gate_acc_vmem[pl.ds(btc_id * btc, btc), pl.ds(0, bf)]
                            up_val = b_up_acc_vmem[pl.ds(btc_id * btc, btc), pl.ds(0, bf)]
                            act = activation_fn(gate, up_val, act_fn)
                        if not disable_dynamic_ffn2:
                            for p_id in range(t_packing):
                                if use_direct_w2:
                                    def _ffn2_sg_body(sg_id, partial_acc):
                                        sg_off = sg_id * quant_block_k
                                        gate_slice = b_gate_acc_vmem[
                                            pl.ds(btc_id * btc, btc),
                                            pl.ds(sg_off, quant_block_k),
                                        ]
                                        up_slice = b_up_acc_vmem[
                                            pl.ds(btc_id * btc, btc),
                                            pl.ds(sg_off, quant_block_k),
                                        ]
                                        act_slice = activation_fn(gate_slice, up_slice, act_fn)
                                        w2_tile = b_w2_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_off, quant_block_k),
                                            pl.ds(0, h_per_t),
                                        ]
                                        d = jnp.dot(
                                            act_slice, w2_tile,
                                            preferred_element_type=jnp.float32,
                                        )
                                        s = b_w2_scale_x2_vmem[
                                            slot,
                                            p_id,
                                            pl.ds(sg_id, 1),
                                            0,
                                            pl.ds(0, h_per_t),
                                        ].reshape(1, h_per_t)
                                        return partial_acc + d * jnp.broadcast_to(s, d.shape)

                                    partial = lax.fori_loop(
                                        0,
                                        n_sg2,
                                        _ffn2_sg_body,
                                        jnp.zeros((btc, h_per_t), dtype=jnp.float32),
                                        unroll=n_sg2,
                                    )
                                else:
                                    w2_tile = b_w2_dq_vmem[p_id] if w2_scale_hbm is not None else b_w2_x2_vmem[slot, p_id]
                                    partial = jnp.dot(
                                        act, w2_tile, preferred_element_type=jnp.float32,
                                    )
                                acc_ref = b_y_acc_vmem.at[
                                    pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                                ]
                                if bf_id == 0:
                                    acc_ref[...] = partial
                                else:
                                    acc_ref[...] = acc_ref[...] + partial
                        return None

                    lax.fori_loop(0, num_btc_per_bts, act_down_btc, None)

                    # Prefetch next bf tile (double-buffer only)
                    if not decode_mode:
                        next_bf_id = bf_id + 2
                        if next_bf_id < num_bf:
                            start_fetch_w1(local_e_id, slot, next_bf_id)
                            start_fetch_w3(local_e_id, slot, next_bf_id)
                            start_fetch_w2(local_e_id, slot, next_bf_id, priority=1)

                # Final writeback: f32 → bf16, then DMA to HBM (once per bts tile)
                def writeback_btc(btc_id, ___):
                    for p_id in range(t_packing):
                        acc_slice = b_y_acc_vmem[
                            pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                        ]
                        b_y_stage_vmem.at[
                            pl.ds(btc_id * btc, btc), p_id, pl.ds(0, h_per_t)
                        ][...] = acc_slice.astype(t_dtype)
                    return None

                lax.fori_loop(0, num_btc_per_bts, writeback_btc, None)

                pltpu.make_async_copy(
                    src_ref=b_y_stage_vmem,
                    dst_ref=a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(tile_start, bts)],
                    sem=y_store_sem.at[0],
                ).start()
                pltpu.make_async_copy(
                    src_ref=b_y_stage_vmem, dst_ref=b_y_stage_vmem,
                    sem=y_store_sem.at[0],
                ).wait()

                return None

            lax.fori_loop(0, num_bts_tiles, bts_body, None)

            return jnp.int32(0)

        lax.cond(has_tokens, _run_active, _run_inactive, None)

    # ===== Output accumulation =====

    def acc_and_store_output(*, bt_sem_id, out_buf_id):
        acc_bt = a2a_g_acc_vmem.shape[2]
        assert bt % acc_bt == 0
        num_acc_tiles = bt // acc_bt

        def start_load_acc_bt(*, tile_start, buf_id):
            def _load_one(t_i, _):
                t_id = tile_start + t_i
                token_e0 = t2e_routing_x2_smem[bt_sem_id, t_id, 0]
                is_valid = token_e0 >= 0

                @pl.when(is_valid)
                def _():
                    for k_id in range(top_k):
                        e_id = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]
                        offset = expert_offsets_x2_smem[bt_sem_id, 1, e_id]
                        expert_offsets_x2_smem[bt_sem_id, 1, e_id] = offset + 1
                        pltpu.make_async_copy(
                            src_ref=a2a_g_hbm.at[e_id, pl.ds(offset, 1)],
                            dst_ref=a2a_g_acc_vmem.at[buf_id, k_id, pl.ds(t_i, 1)],
                            sem=a2a_acc_sems.at[0],
                        ).start()

                @pl.when(jnp.logical_not(is_valid))
                def _():
                    zeros = jnp.zeros((1, t_packing, h_per_t), dtype=a2a_g_acc_vmem.dtype)
                    for k_id in range(top_k):
                        a2a_g_acc_vmem.at[buf_id, k_id, pl.ds(t_i, 1)][...] = zeros
                return None

            lax.fori_loop(0, acc_bt, _load_one, None, unroll=False)

        def wait_load_acc_bt(*, buf_id, tile_start):
            def _count_valid(t_i, acc):
                token_e0 = t2e_routing_x2_smem[bt_sem_id, tile_start + t_i, 0]
                return acc + (token_e0 >= 0).astype(jnp.int32)

            num_valid = lax.fori_loop(0, acc_bt, _count_valid, jnp.int32(0), unroll=False)

            @pl.when(num_valid != 0)
            def _():
                def _wait_one(_, __):
                    ref = a2a_g_acc_vmem.at[buf_id, 0, pl.ds(0, 1)]
                    pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=a2a_acc_sems.at[0]).wait()
                    return None
                lax.fori_loop(0, num_valid * jnp.int32(top_k), _wait_one, None, unroll=False)

        def acc_gather_to_output(*, tile_start, out_offset, buf_id):
            output_tile = jnp.zeros((acc_bt, t_packing, h_per_t), dtype=jnp.float32)
            logits_tile = b_topk_weights_x2_vmem[
                bt_sem_id, pl.ds(tile_start, acc_bt), pl.ds(0, top_k)
            ]
            for k_id in range(top_k):
                acc_tile = a2a_g_acc_vmem[buf_id, k_id, :acc_bt].astype(jnp.float32)
                logits = logits_tile[:, k_id].reshape(acc_bt, 1, 1)
                output_tile += acc_tile * logits

            out_offset = pl.multiple_of(out_offset, 16)

            if w1_shared_hbm is not None and not disable_shared_expert:
                se_tile = b_se_acc_vmem[
                    out_buf_id, pl.ds(out_offset, acc_bt), pl.ds(0, hidden_size)
                ]
                output_tile = output_tile.reshape(acc_bt, hidden_size) + se_tile

            target = b_output_x2_vmem.at[
                out_buf_id, pl.ds(out_offset, acc_bt), pl.ds(0, hidden_size)
            ]
            target[...] = output_tile.reshape(acc_bt, hidden_size).astype(output_hbm.dtype)

        start_load_acc_bt(tile_start=0, buf_id=0)

        def run_acc_pipeline(i, _):
            curr_buf = i % 2
            next_buf = (i + 1) % 2
            curr_start = i * acc_bt
            next_start = (i + 1) * acc_bt

            @pl.when(i + 1 < num_acc_tiles)
            def _():
                start_load_acc_bt(tile_start=next_start, buf_id=next_buf)

            wait_load_acc_bt(buf_id=curr_buf, tile_start=curr_start)
            acc_gather_to_output(tile_start=curr_start, out_offset=i * acc_bt, buf_id=curr_buf)
            return None

        lax.fori_loop(0, num_acc_tiles, run_acc_pipeline, None, unroll=False)

    # ===== Output DMA =====

    def start_send_bo(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        bt_start = bt_id * bt
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=output_hbm.at[pl.ds(bt_start, bt)],
            sem=local_sems.at[bt_sem_id, 7],
        ).start(priority=priority)

    def wait_store_output(*, bt_id):
        is_valid = jnp.logical_and(bt_id >= 0, bt_id < num_bt)
        bt_sem_id = bt_id & 1

        @pl.when(is_valid)
        def _():
            bt_start = bt_id * bt
            ref = output_hbm.at[pl.ds(bt_start, bt)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=local_sems.at[bt_sem_id, 7],
            ).wait()

    # ===== Shared expert (reads weights directly from HBM refs) =====

    def start_fetch_se_tokens(bt_id):
        if w1_shared_hbm is None or disable_shared_expert:
            return
        bt_start = bt_id * bt
        bt_sem_id = bt_id & jnp.int32(1)
        for p_id in range(t_packing):
            pltpu.make_async_copy(
                src_ref=tokens_hbm.at[
                    pl.ds(bt_start, bt), p_id, pl.ds(0, h_per_t)
                ],
                dst_ref=b_se_tokens_vmem.at[bt_sem_id, 0, pl.ds(0, bt), p_id, pl.ds(0, h_per_t)],
                sem=local_sems.at[bt_sem_id, 0],
            ).start()

    def wait_fetch_se_tokens(bt_id):
        if w1_shared_hbm is None or disable_shared_expert:
            return
        bt_sem_id = bt_id & jnp.int32(1)
        for _ in range(t_packing):
            ref = b_se_tokens_vmem.at[bt_sem_id, 0, pl.ds(0, bt), 0, pl.ds(0, h_per_t)]
            pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=local_sems.at[bt_sem_id, 0]).wait()

    def run_shared_expert_slice(block_id, bt_id, bt_sem_id, out_buf_id):
        if w1_shared_hbm is None or disable_shared_expert:
            return

        @pl.when(block_id < se_total_blocks)
        def _():
            gate_acc = jnp.zeros((bt, bse), dtype=jnp.float32)
            up_acc = jnp.zeros((bt, bse), dtype=jnp.float32)

            for p_id in range(t_packing):
                t_slice = b_se_tokens_vmem[
                    bt_sem_id, 0, pl.ds(0, bt), p_id, pl.ds(0, h_per_t)
                ]
                w1_slice = w1_shared_hbm.at[
                    pl.ds(p_id * h_per_t, h_per_t), pl.ds(block_id * bse, bse)
                ]
                w3_slice = w3_shared_hbm.at[
                    pl.ds(p_id * h_per_t, h_per_t), pl.ds(block_id * bse, bse)
                ]
                gate_acc += jnp.dot(t_slice, w1_slice[...], preferred_element_type=jnp.float32)
                up_acc += jnp.dot(t_slice, w3_slice[...], preferred_element_type=jnp.float32)

            act = activation_fn(gate_acc, up_acc, act_fn)

            for p_id in range(t_packing):
                w2_slice = w2_shared_hbm.at[
                    pl.ds(block_id * bse, bse), pl.ds(p_id * h_per_t, h_per_t)
                ]
                partial = jnp.dot(act, w2_slice[...], preferred_element_type=jnp.float32)
                se_ref = b_se_acc_vmem.at[
                    out_buf_id, pl.ds(0, bt), pl.ds(p_id * h_per_t, h_per_t)
                ]

                @pl.when(block_id == 0)
                def _(se_ref=se_ref, partial=partial):
                    se_ref[...] = partial

                @pl.when(block_id > 0)
                def _(se_ref=se_ref, partial=partial):
                    se_ref[...] = se_ref[...] + partial

    # ===== run_bt =====

    if num_bt >= 1:
        start_fetch_topk(bt_id=jnp.int32(0))
        start_fetch_se_tokens(bt_id=jnp.int32(0))

    def run_bt(bt_id, e_sem_id):
        bt_start = bt_id * bt
        bt_sem_id = bt_id & jnp.int32(1)
        next_bt_id = bt_id + jnp.int32(1)
        out_buf_id = bt_id & jnp.int32(1)

        @pl.when(next_bt_id < num_bt)
        def _():
            start_fetch_topk(bt_id=next_bt_id)
            start_fetch_se_tokens(next_bt_id)

        wait_fetch_topk(bt_id=bt_id)

        t2e_routing = b_topk_ids_x2_vmem[bt_sem_id]

        all_reduce_metadata(
            bt_id=bt_id, bt_sem_id=bt_sem_id, t2e_routing=t2e_routing,
        )

        wait_store_output(bt_id=bt_id - 2)

        se_per_expert = (
            max(2, cdiv(se_total_blocks, local_num_experts)) if se_total_blocks > 0 else 2
        )
        se_before = se_per_expert // 2
        se_after = se_per_expert - se_before

        if expert_buffer_count >= local_num_experts:
            # === BATCH SCATTER ===
            start_a2a_scatter_batch(bt_sem_id=bt_sem_id, bt_start=bt_start)

            init_carry = jnp.int32(0)

            def compute_expert_batch(local_e_id, curr_se_block):
                e_sem_id_local = local_e_id

                for _ in range(se_before):
                    run_shared_expert_slice(curr_se_block, bt_id, bt_sem_id, out_buf_id)
                    curr_se_block += 1

                wait_a2a_scatter_recv(
                    bt_sem_id=bt_sem_id, e_sem_id=e_sem_id_local,
                    local_e_id=local_e_id,
                )
                expert_ffn(bt_sem_id, e_sem_id_local, local_e_id)
                start_a2a_gather(
                    bt_sem_id=bt_sem_id, e_sem_id=e_sem_id_local,
                    local_e_id=local_e_id,
                )

                for _ in range(se_after):
                    run_shared_expert_slice(curr_se_block, bt_id, bt_sem_id, out_buf_id)
                    curr_se_block += 1

                return curr_se_block

            final_se_block = lax.fori_loop(
                0, local_num_experts, compute_expert_batch, init_carry, unroll=False,
            )

            def cleanup_body(block_idx, _):
                run_shared_expert_slice(block_idx, bt_id, bt_sem_id, out_buf_id)
                return None

            lax.fori_loop(final_se_block, se_total_blocks, cleanup_body, None)

            wait_a2a_scatter_send_batch()
            wait_a2a_gather_recv_all(bt_sem_id=bt_sem_id)
            sync_barrier()

            if not disable_acc_and_store:
                acc_and_store_output(bt_sem_id=bt_sem_id, out_buf_id=out_buf_id)
            start_send_bo(bt_id=bt_id)

            tail_start = max(local_num_experts - expert_buffer_count, 0)
            for tail_e_id in range(tail_start, local_num_experts):
                wait_a2a_gather_send(
                    bt_sem_id=bt_sem_id, e_sem_id=tail_e_id,
                    local_e_id=tail_e_id,
                )

            @pl.when(bt_id + 1 < num_bt)
            def _():
                sync_barrier()

            final_e_sem_id = e_sem_id

        else:
            # === PIPELINED SCATTER ===
            start_a2a_scatter(
                bt_sem_id=bt_sem_id, e_sem_id=e_sem_id,
                local_e_id=0, bt_start=bt_start,
            )

            init_carry = (e_sem_id, jnp.int32(0))

            def run_per_expert_pipelined(local_e_id, carry):
                curr_e_sem_id, curr_se_block = carry
                next_e_sem_id = (curr_e_sem_id + jnp.int32(1)) % jnp.int32(expert_buffer_count)
                next_local_e_id = local_e_id + 1

                @pl.when(next_local_e_id < local_num_experts)
                def _():
                    @pl.when(
                        (next_local_e_id >= expert_buffer_count)
                        & (next_local_e_id % expert_buffer_count == 0)
                    )
                    def _():
                        sync_barrier()

                    start_a2a_scatter(
                        bt_sem_id=bt_sem_id, e_sem_id=next_e_sem_id,
                        local_e_id=next_local_e_id, bt_start=bt_start,
                    )

                for _ in range(se_before):
                    run_shared_expert_slice(curr_se_block, bt_id, bt_sem_id, out_buf_id)
                    curr_se_block += 1

                wait_a2a_scatter_recv(
                    bt_sem_id=bt_sem_id, e_sem_id=curr_e_sem_id,
                    local_e_id=local_e_id,
                )
                expert_ffn(bt_sem_id, curr_e_sem_id, local_e_id)
                start_a2a_gather(
                    bt_sem_id=bt_sem_id, e_sem_id=curr_e_sem_id,
                    local_e_id=local_e_id,
                )

                for _ in range(se_after):
                    run_shared_expert_slice(curr_se_block, bt_id, bt_sem_id, out_buf_id)
                    curr_se_block += 1

                wait_a2a_scatter_send(
                    bt_sem_id=bt_sem_id, e_sem_id=curr_e_sem_id,
                    local_e_id=local_e_id,
                )
                return (next_e_sem_id, curr_se_block)

            final_carry = lax.fori_loop(
                0, local_num_experts, run_per_expert_pipelined, init_carry, unroll=False,
            )
            final_e_sem_id, final_se_block = final_carry

            def cleanup_body(block_idx, _):
                run_shared_expert_slice(block_idx, bt_id, bt_sem_id, out_buf_id)
                return None

            lax.fori_loop(final_se_block, se_total_blocks, cleanup_body, None)

            wait_a2a_gather_recv_all(bt_sem_id=bt_sem_id)
            sync_barrier()

            if not disable_acc_and_store:
                acc_and_store_output(bt_sem_id=bt_sem_id, out_buf_id=out_buf_id)
            start_send_bo(bt_id=bt_id)

            tail_start = max(local_num_experts - expert_buffer_count, 0)
            for tail_e_id in range(tail_start, local_num_experts):
                tail_sem = (e_sem_id + tail_e_id) % expert_buffer_count
                wait_a2a_gather_send(
                    bt_sem_id=bt_sem_id, e_sem_id=tail_sem,
                    local_e_id=tail_e_id,
                )

            @pl.when(bt_id + 1 < num_bt)
            def _():
                sync_barrier()

            final_e_sem_id = final_e_sem_id

        return final_e_sem_id

    # ===== Kernel start =====
    sync_barrier()

    lax.fori_loop(0, num_bt, run_bt, jnp.int32(0), unroll=False)
    wait_store_output(bt_id=jnp.int32(num_bt - 2))
    wait_store_output(bt_id=jnp.int32(num_bt - 1))


# ---------------------------------------------------------------------------
# Pre-kernel metadata (JAX-level all-reduce)
# ---------------------------------------------------------------------------

def compute_local_expert_sizes(topk_ids: jax.Array, num_experts: int) -> jax.Array:
    flat_ids = topk_ids.flatten()
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = jnp.where(valid, flat_ids, num_experts)
    counts = jnp.bincount(safe_ids, length=num_experts + 1)[:num_experts]
    return counts[None, :].astype(jnp.int32)


def jax_allreduce_metadata_by_bt(
    topk_ids, num_experts, bt, num_devices, dp_axis_name, tp_axis_name,
):
    num_tokens = topk_ids.shape[0]
    if num_tokens % bt != 0:
        raise ValueError(f"{num_tokens=} must be divisible by {bt=}.")

    topk_ids_by_bt = topk_ids.reshape(num_tokens // bt, bt, topk_ids.shape[-1])
    local_sizes = jax.vmap(compute_local_expert_sizes, in_axes=(0, None))(
        topk_ids_by_bt, num_experts,
    )

    all_sizes = lax.all_gather(
        local_sizes, axis_name=(dp_axis_name, tp_axis_name), axis=1, tiled=True,
    ).astype(jnp.int32)
    sizes = jnp.sum(all_sizes, axis=1, keepdims=True).astype(jnp.int32)

    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    device_ids = lax.broadcasted_iota(jnp.int32, (num_devices,), 0)
    prefix_mask = device_ids < my_id
    starts = jnp.sum(
        jnp.where(prefix_mask[None, :, None], all_sizes, jnp.zeros_like(all_sizes)),
        axis=1, keepdims=True,
    ).astype(jnp.int32)

    d2e_counts = all_sizes[:, :, None, :]
    return starts, sizes, d2e_counts


# ---------------------------------------------------------------------------
# Outer function
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh", "top_k", "act_fn",
        "disable_a2a", "disable_shared_expert", "disable_sync_barrier",
        "disable_weight_load", "disable_dynamic_ffn1", "disable_dynamic_ffn2",
        "disable_acc_and_store",
        "use_jax_allreduce_metadata",
        "block_config", "dp_axis_name", "tp_axis_name",
        "quant_block_k", "decode_mode", "direct_scaled_dot",
        "skip_decode_sync_barrier",
    ],
)
def fused_ep_moe_v2(
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    top_k: int,
    *,
    act_fn: str = "silu",
    disable_a2a: bool = False,
    disable_shared_expert: bool = False,
    disable_sync_barrier: bool = False,
    disable_weight_load: bool = False,
    disable_dynamic_ffn1: bool = False,
    disable_dynamic_ffn2: bool = False,
    disable_acc_and_store: bool = False,
    use_jax_allreduce_metadata: bool = True,
    w1_shared: jax.Array | None = None,
    w2_shared: jax.Array | None = None,
    w3_shared: jax.Array | None = None,
    quant_block_k: int | None = None,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    block_config: FusedMoEBlockConfig | None = None,
    decode_mode: bool = False,
    direct_scaled_dot: bool = False,
    skip_decode_sync_barrier: bool = False,
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
):
    ep_size = get_ep_size(mesh, dp_axis_name, tp_axis_name)
    num_devices = ep_size

    num_tokens, hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape
    local_num_experts = num_experts // ep_size
    se_inter_size = w2_shared.shape[0] if w2_shared is not None else 0

    local_num_tokens = num_tokens // ep_size

    orig_local_num_tokens = local_num_tokens
    aligned_local_num_tokens = _align_local_tokens_for_decode(local_num_tokens)
    pad_local = aligned_local_num_tokens - local_num_tokens
    if pad_local > 0:
        local_num_tokens = local_num_tokens + pad_local
        num_tokens = local_num_tokens * ep_size

    if block_config is None:
        block_config = FusedMoEBlockConfig(
            bt=min(16, local_num_tokens),
            bf=256,
            btc=128,
            bse=256,
        )

    block_config = block_config.effective_for(
        num_tokens=num_tokens, ep_size=ep_size,
    )
    bt = block_config.bt
    bts = block_config.bts
    bf = block_config.bf
    btc = block_config.btc
    bse = block_config.bse

    validate_fused_moe_block_config(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        dtype=tokens.dtype, ep_size=ep_size, block_config=block_config,
    )

    if w1_scale is not None:
        if quant_block_k is None:
            raise ValueError("quant_block_k required when w1_scale is provided.")
    if quant_block_k is not None:
        if quant_block_k % 128 != 0:
            raise ValueError(f"{quant_block_k=} must be aligned to 128.")
        _qbk_t_packing = get_dtype_packing(tokens.dtype)
        _qbk_h_per_t = hidden_size // _qbk_t_packing
        if _qbk_h_per_t % quant_block_k != 0:
            raise ValueError(f"h_per_t={_qbk_h_per_t} must be divisible by {quant_block_k=}.")
        if bf % quant_block_k != 0:
            raise ValueError(f"{bf=} must be divisible by {quant_block_k=}.")
        expected_w1_scale = (num_experts, hidden_size // quant_block_k, 1, intermediate_size)
        if w1_scale is not None and w1_scale.shape != expected_w1_scale:
            raise ValueError(f"{w1_scale.shape=} != {expected_w1_scale}")
        expected_w2_scale = (num_experts, intermediate_size // quant_block_k, 1, hidden_size)
        if w2_scale is not None and w2_scale.shape != expected_w2_scale:
            raise ValueError(f"{w2_scale.shape=} != {expected_w2_scale}")
        expected_w3_scale = (num_experts, hidden_size // quant_block_k, 1, intermediate_size)
        if w3_scale is not None and w3_scale.shape != expected_w3_scale:
            raise ValueError(f"{w3_scale.shape=} != {expected_w3_scale}")

    needs_jax_allreduce = use_jax_allreduce_metadata and ep_size > 1

    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    t_dtype = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)
    h_per_t = hidden_size // t_packing

    a2a_max_tokens = align_to(bt * num_devices, bts)

    a2a_scratch_budget = int(_device_hbm_bytes() * _A2A_HBM_FRACTION)
    bytes_per_slot = 2 * a2a_max_tokens * hidden_size * jnp.dtype(t_dtype).itemsize
    expert_buffer_count = min(local_num_experts, max(2, a2a_scratch_budget // bytes_per_slot))

    tokens = tokens.reshape(-1, t_packing, h_per_t)

    if padded_top_k > top_k:
        topk_ids = jnp.pad(
            topk_ids, ((0, 0), (0, padded_top_k - top_k)),
            mode="constant", constant_values=-1,
        )
        topk_weights = jnp.pad(
            topk_weights, ((0, 0), (0, padded_top_k - top_k)),
            mode="constant", constant_values=0,
        )

    acc_bt = math.gcd(bt, 16)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    wb_slots = 1 if decode_mode else 2
    use_w1_dequant_scratch = w1_scale is not None and not direct_scaled_dot
    use_w3_dequant_scratch = w3_scale is not None and not direct_scaled_dot
    use_w2_dequant_scratch = w2_scale is not None and not direct_scaled_dot

    scope_name = f"fused-moe-v2-k_{top_k}-bt_{bt}_{bts}_{btc}-bf_{bf}"
    if direct_scaled_dot:
        scope_name += "-direct_scaled_dot"
    if skip_decode_sync_barrier:
        scope_name += "-skip_decode_sync"
    if w1_shared is not None:
        scope_name += f"-se_bse_{bse}"

    scratch_shapes = (
        # SMEM: routing/metadata
        pltpu.SMEM((2, bt, padded_top_k), jnp.int32),                      # t2e_routing
        pltpu.SMEM((2, num_devices, 1, padded_num_experts), jnp.int32),     # d2e_count
        pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),                  # expert_offsets
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),                  # expert_starts
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),                  # expert_sizes
        pltpu.SMEM((expert_buffer_count,), jnp.int32),                      # a2a_s_sends
        # VMEM: gather accumulation
        pltpu.VMEM((2, top_k, acc_bt, t_packing, h_per_t), t_dtype),       # a2a_g_acc
        # VMEM: topk
        pltpu.VMEM((2, bt, padded_top_k), jnp.float32),                    # topk_weights
        pltpu.VMEM((2, bt, padded_top_k), jnp.int32),                      # topk_ids
        # VMEM: output double buffer
        pltpu.VMEM((2, bt, hidden_size), t_dtype),                          # output
        # VMEM: weight double buffers
        pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w1.dtype),                 # W1
        pltpu.VMEM((wb_slots, t_packing, h_per_t, bf), w3.dtype),                 # W3
        pltpu.VMEM((wb_slots, t_packing, bf, h_per_t), w2.dtype),                 # W2
        # VMEM: scale double buffers (None when not quantized)
        (None if w1_scale is None else
            pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32)),  # W1 scale
        (None if w3_scale is None else
            pltpu.VMEM((wb_slots, t_packing, h_per_t // quant_block_k, 1, bf), jnp.float32)),  # W3 scale
        (None if w2_scale is None else
            pltpu.VMEM((wb_slots, t_packing, bf // quant_block_k, 1, h_per_t), jnp.float32)),  # W2 scale
        # VMEM: dequant scratch (single-buf, None when not quantized)
        (None if not use_w1_dequant_scratch else
            pltpu.VMEM((t_packing, h_per_t, bf), jnp.bfloat16)),                    # W1 dequant
        (None if not use_w3_dequant_scratch else
            pltpu.VMEM((t_packing, h_per_t, bf), jnp.bfloat16)),                    # W3 dequant
        (None if not use_w2_dequant_scratch else
            pltpu.VMEM((t_packing, bf, h_per_t), jnp.bfloat16)),                    # W2 dequant
        # VMEM: gate/up accumulators (per bts tile)
        pltpu.VMEM((bts, bf), jnp.float32),                                 # gate_acc
        pltpu.VMEM((bts, bf), jnp.float32),                                 # up_acc
        # VMEM: token staging per bts tile
        pltpu.VMEM((bts, t_packing, h_per_t), t_dtype),                    # x
        # VMEM: output accumulator per bts tile (fp32)
        pltpu.VMEM((bts, t_packing, h_per_t), jnp.float32),                # y_acc
        # VMEM: output staging for HBM read-modify-write per bts tile
        pltpu.VMEM((bts, t_packing, h_per_t), t_dtype),                    # y_stage
        # VMEM: shared expert
        (None if w1_shared is None else
            pltpu.VMEM((2, 2, bt, t_packing, h_per_t), t_dtype)),           # se_tokens
        (None if w1_shared is None else
            pltpu.VMEM((2, t_packing, h_per_t, bse), w1.dtype)),            # se_w1
        (None if w3_shared is None else
            pltpu.VMEM((2, t_packing, h_per_t, bse), w3.dtype)),            # se_w3
        (None if w2_shared is None else
            pltpu.VMEM((2, t_packing, bse, h_per_t), w2.dtype)),            # se_w2
        (None if w1_shared is None else
            pltpu.VMEM((2, bt, hidden_size), jnp.float32)),                 # se_acc
        # Semaphores
        pltpu.SemaphoreType.DMA((1,)),                                      # x_stage
        pltpu.SemaphoreType.DMA((1,)),                                      # y_store
        pltpu.SemaphoreType.DMA((2, 10)),                                   # local_sems
        pltpu.SemaphoreType.DMA((expert_buffer_count,)),                    # send
        pltpu.SemaphoreType.DMA((expert_buffer_count,)),                    # recv
        pltpu.SemaphoreType.DMA((expert_buffer_count,)),                    # gather_send
        pltpu.SemaphoreType.DMA,                                            # a2a_gather
        pltpu.SemaphoreType.DMA((1,)),                                      # a2a_acc
        pltpu.SemaphoreType.BARRIER,                                        # barrier
    )

    fused_moe = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _fused_ep_moe_kernel,
                top_k=top_k,
                dp_axis_name=dp_axis_name,
                tp_axis_name=tp_axis_name,
                act_fn=act_fn,
                disable_a2a=disable_a2a,
                disable_shared_expert=disable_shared_expert,
                disable_sync_barrier=disable_sync_barrier,
                disable_weight_load=disable_weight_load,
                disable_dynamic_ffn1=disable_dynamic_ffn1,
                disable_dynamic_ffn2=disable_dynamic_ffn2,
                disable_acc_and_store=disable_acc_and_store,
                use_jax_allreduce_metadata=use_jax_allreduce_metadata,
                decode_mode=decode_mode,
                direct_scaled_dot=direct_scaled_dot,
                skip_decode_sync_barrier=skip_decode_sync_barrier,
                bt=bt, bf=bf, btc=btc, bts=bts, bse=bse,
                quant_block_k=quant_block_k,
            ),
            out_shape=jax.ShapeDtypeStruct((local_num_tokens, hidden_size), t_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm_spec,                                               # tokens
                    hbm_spec,                                               # w1
                    hbm_spec,                                               # w2
                    hbm_spec,                                               # w3
                    None if w1_scale is None else hbm_spec,                 # w1_scale
                    None if w2_scale is None else hbm_spec,                 # w2_scale
                    None if w3_scale is None else hbm_spec,                 # w3_scale
                    hbm_spec,                                               # topk_weights
                    hbm_spec,                                               # topk_ids
                    hbm_spec,                                               # a2a_s
                    hbm_spec,                                               # a2a_s_acc
                    hbm_spec,                                               # a2a_g
                    None if w1_shared is None else hbm_spec,                # w1_shared
                    None if w3_shared is None else hbm_spec,                # w3_shared
                    None if w2_shared is None else hbm_spec,                # w2_shared
                    None if not needs_jax_allreduce else hbm_spec,          # metadata_starts
                    None if not needs_jax_allreduce else hbm_spec,          # metadata_sizes
                    None if not needs_jax_allreduce else hbm_spec,          # metadata_d2e_counts
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
            P((dp_axis_name, tp_axis_name)),     # tokens
            P((dp_axis_name, tp_axis_name)),     # w1
            P((dp_axis_name, tp_axis_name)),     # w2
            P((dp_axis_name, tp_axis_name)),     # w3
            None if w1_scale is None else P((dp_axis_name, tp_axis_name)),  # w1_scale
            None if w2_scale is None else P((dp_axis_name, tp_axis_name)),  # w2_scale
            None if w3_scale is None else P((dp_axis_name, tp_axis_name)),  # w3_scale
            P((dp_axis_name, tp_axis_name)),     # topk_weights
            P((dp_axis_name, tp_axis_name)),     # topk_ids
            P(),                                  # a2a_s
            P(),                                  # a2a_s_acc
            P(),                                  # a2a_g
            None if w1_shared is None else P(),   # w1_shared
            None if w3_shared is None else P(),   # w3_shared
            None if w2_shared is None else P(),   # w2_shared
        ),
        out_specs=P((dp_axis_name, tp_axis_name)),
        check_vma=False,
    )
    def kernel(
        tokens, w1, w2, w3,
        w1_scale_arg, w2_scale_arg, w3_scale_arg,
        topk_weights, topk_ids,
        a2a_s_hbm_scratch, a2a_s_acc_hbm_scratch, a2a_g_hbm_scratch,
        w1_shared=None, w3_shared=None, w2_shared=None,
    ):
        if pad_local > 0:
            tokens = jnp.pad(tokens, ((0, pad_local), (0, 0), (0, 0)))
            topk_weights = jnp.pad(topk_weights, ((0, pad_local), (0, 0)),
                                   constant_values=0.0)
            topk_ids = jnp.pad(
                topk_ids,
                ((0, pad_local), (0, 0)),
                mode="constant",
                constant_values=-1,
            )

        if needs_jax_allreduce:
            md_starts, md_sizes, md_d2e = jax_allreduce_metadata_by_bt(
                topk_ids[:, :top_k], padded_num_experts, bt,
                num_devices, dp_axis_name, tp_axis_name,
            )
            md_starts_arg = pltpu.with_memory_space_constraint(md_starts, pltpu.HBM)
            md_sizes_arg = pltpu.with_memory_space_constraint(md_sizes, pltpu.HBM)
            md_d2e_arg = pltpu.with_memory_space_constraint(md_d2e, pltpu.HBM)
        else:
            md_starts_arg = None
            md_sizes_arg = None
            md_d2e_arg = None

        out = fused_moe(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
            (None if w1_scale_arg is None else
                pltpu.with_memory_space_constraint(w1_scale_arg, pltpu.HBM)),
            (None if w2_scale_arg is None else
                pltpu.with_memory_space_constraint(w2_scale_arg, pltpu.HBM)),
            (None if w3_scale_arg is None else
                pltpu.with_memory_space_constraint(w3_scale_arg, pltpu.HBM)),
            pltpu.with_memory_space_constraint(topk_weights, pltpu.HBM),
            pltpu.with_memory_space_constraint(topk_ids, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_s_hbm_scratch, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_s_acc_hbm_scratch, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_g_hbm_scratch, pltpu.HBM),
            (None if w1_shared is None else
                pltpu.with_memory_space_constraint(w1_shared, pltpu.HBM)),
            (None if w3_shared is None else
                pltpu.with_memory_space_constraint(w3_shared, pltpu.HBM)),
            (None if w2_shared is None else
                pltpu.with_memory_space_constraint(w2_shared, pltpu.HBM)),
            md_starts_arg,
            md_sizes_arg,
            md_d2e_arg,
        )
        if pad_local > 0:
            out = out[:orig_local_num_tokens]
        return out

    a2a_s_hbm_scratch = pl.empty(
        (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t), t_dtype,
    )
    a2a_s_acc_hbm_scratch = pl.empty(
        (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t), t_dtype,
    )
    a2a_g_hbm_scratch = pl.empty(
        (num_experts, bt, t_packing, h_per_t), t_dtype,
    )

    return kernel(
        tokens, w1, w2, w3,
        w1_scale, w2_scale, w3_scale,
        topk_weights, topk_ids,
        a2a_s_hbm_scratch, a2a_s_acc_hbm_scratch, a2a_g_hbm_scratch,
        w1_shared, w3_shared, w2_shared,
    )
