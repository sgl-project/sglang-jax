"""Fused EP MoE V2 — Double-Buffer Expert FFN + Direct Scatter → VMEM.

Key optimization over V1:
- Scatter tokens directly from HBM → VMEM (skip HBM intermediate buffer)
- Double-buffered persistent tokens in VMEM for scatter-FFN overlap
- Expert FFN tiles along intermediate dimension (gate/up/down fused per tile)
- fp32 accumulation across tiles, bf16 writeback

M3 scope: bf16 tokens/weights, no quantization, no shared expert, no bias.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

cdiv = pl.cdiv


def align_to(x, a):
    return cdiv(x, a) * a


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
        raise RuntimeError(f"Unsupported activation function: {act_fn}")


def compute_local_expert_sizes(topk_ids: jax.Array, num_experts: int) -> jax.Array:
    flat_ids = topk_ids.flatten()
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    safe_ids = jnp.where(valid, flat_ids, num_experts)
    counts = jnp.bincount(safe_ids, length=num_experts + 1)[:num_experts]
    return counts[None, :].astype(jnp.int32)


def jax_allreduce_metadata_by_bt(
    topk_ids: jax.Array,
    num_experts: int,
    bt: int,
    num_devices: int,
    dp_axis_name: str,
    tp_axis_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_tokens = topk_ids.shape[0]
    if num_tokens % bt != 0:
        raise ValueError(
            f"Expected local topk_ids tokens ({num_tokens}) to be divisible by {bt=}."
        )

    topk_ids_by_bt = topk_ids.reshape(num_tokens // bt, bt, topk_ids.shape[-1])
    local_sizes = jax.vmap(compute_local_expert_sizes, in_axes=(0, None))(
        topk_ids_by_bt,
        num_experts,
    )

    all_sizes = lax.all_gather(
        local_sizes,
        axis_name=(dp_axis_name, tp_axis_name),
        axis=1,
        tiled=True,
    )
    all_sizes = all_sizes.astype(jnp.int32)
    if all_sizes.shape[1] != num_devices:
        raise ValueError(
            f"Expected gathered metadata axis to be {num_devices}, got {all_sizes.shape}."
        )

    sizes = jnp.sum(all_sizes, axis=1, keepdims=True).astype(jnp.int32)

    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    device_ids = lax.broadcasted_iota(jnp.int32, (num_devices,), 0)
    prefix_mask = device_ids < my_id
    starts = jnp.sum(
        jnp.where(
            prefix_mask[None, :, None],
            all_sizes,
            jnp.zeros_like(all_sizes),
        ),
        axis=1,
        keepdims=True,
    ).astype(jnp.int32)

    d2e_counts = all_sizes[:, :, None, :]
    return starts, sizes, d2e_counts


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------


def _fused_ep_moe_v2_kernel(
    # HBM Inputs.
    tokens_hbm,              # (local_num_tokens, t_packing, h_per_t) bf16
    w1_hbm,                  # (local_num_experts, hidden_size, intermediate_size) bf16
    w2_hbm,                  # (local_num_experts, intermediate_size, hidden_size) bf16
    w3_hbm,                  # (local_num_experts, hidden_size, intermediate_size) bf16
    topk_weights_hbm,        # (local_num_tokens, top_k) f32
    topk_ids_hbm,            # (local_num_tokens, top_k) int32
    a2a_s_acc_x2_hbm,        # (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t) bf16
    a2a_g_hbm,               # (num_experts, bt, t_packing, h_per_t) bf16
    metadata_starts_hbm,     # None | (num_bt, 1, padded_num_experts) int32
    metadata_sizes_hbm,      # None | (num_bt, 1, padded_num_experts) int32
    metadata_d2e_counts_hbm, # None | (num_bt, num_devices, 1, padded_num_experts) int32
    # HBM Output.
    output_hbm,              # (local_num_tokens, t_packing, h_per_t) bf16
    # SMEM Scratch — routing / metadata.
    t2e_routing_x2_smem,     # (2, bt, padded_top_k) int32
    d2e_count_x2_smem,       # (2, num_devices, 1, padded_num_experts) int32
    expert_offsets_x2_smem,  # (2, 2, padded_num_experts) int32
    expert_starts_x2_smem,   # (2, 1, padded_num_experts) int32
    expert_sizes_x2_smem,    # (2, 1, padded_num_experts) int32
    a2a_s_sends_x2_smem,     # (2,) int32 — per x_buf_id
    # VMEM Scratch — token double buffer (v2 innovation).
    b_x_x2_vmem,             # (2, a2a_max_tokens, t_packing, h_per_t) bf16
    b_y_acc_vmem,            # (a2a_max_tokens, hidden_size) f32
    b_y_out_vmem,            # (a2a_max_tokens, t_packing, h_per_t) bf16 — staging for DMA
    # VMEM Scratch — weight double buffer.
    b_w1_x2_vmem,            # (2, hidden_size, bf) weight_dtype
    b_w3_x2_vmem,            # (2, hidden_size, bf) weight_dtype
    b_w2_x2_vmem,            # (2, bf, hidden_size) weight_dtype
    # VMEM Scratch — gather accumulation & output.
    a2a_g_acc_vmem,          # (2, top_k, acc_bt, t_packing, h_per_t) bf16
    b_topk_weights_x2_vmem,  # (2, bt, padded_top_k) f32
    b_topk_ids_x2_vmem,     # (2, bt, padded_top_k) int32
    b_output_x2_vmem,       # (2, bt, t_packing, h_per_t) bf16
    # Semaphores.
    weight_sems,             # DMA(2, 3) — per w_slot, per {w1, w3, w2}
    x_recv_sems,             # DMA(2,) — per x_buf_id
    x_send_sems,             # DMA(2,) — per x_buf_id
    y_out_sem,               # DMA(1,)
    local_sems,              # DMA(2, 7) — topk/output/metadata
    gather_send_x2_sems,     # DMA(expert_buffer_count,)
    a2a_gather_sem,          # DMA
    a2a_acc_sems,            # DMA(1,)
    barrier_sem,             # BARRIER
    *,
    # Static params.
    top_k: int,
    dp_axis_name: str,
    tp_axis_name: str,
    act_fn: str,
    bt: int,
    bf: int,
):
    # -- Derive shapes from refs --
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = tp_size * dp_size

    local_num_tokens = tokens_hbm.shape[0]
    local_num_experts, intermediate_size, hidden_size = w2_hbm.shape
    t_packing = tokens_hbm.shape[1]
    h_per_t = tokens_hbm.shape[2]
    expert_buffer_count = a2a_s_acc_x2_hbm.shape[0]
    a2a_max_tokens = a2a_s_acc_x2_hbm.shape[1]
    num_experts = a2a_g_hbm.shape[0]
    padded_num_experts = d2e_count_x2_smem.shape[-1]
    padded_top_k = t2e_routing_x2_smem.shape[-1]

    assert expert_buffer_count >= 1
    assert expert_buffer_count <= local_num_experts
    assert local_num_tokens % bt == 0
    num_bt = local_num_tokens // bt
    assert padded_num_experts == align_to(num_experts, 128)
    assert padded_top_k == align_to(top_k, 128)

    t_dtype = tokens_hbm.dtype
    n_w = intermediate_size // bf

    # local_sems layout:
    #   [bt_sem_id, 0]: topk fetch (shared for weights + ids)
    #   [bt_sem_id, 1]: output store
    #   [bt_sem_id, 2]: metadata_starts
    #   [bt_sem_id, 3]: metadata_sizes
    #   [bt_sem_id, 4]: metadata_d2e_counts
    #   [bt_sem_id, 5]: metadata_offsets
    #   [bt_sem_id, 6]: metadata_routing

    # -- Infrastructure closures (Step 3) --

    def get_mesh_device_id(ep_rank):
        dp_rank = ep_rank // tp_size
        tp_rank = ep_rank % tp_size
        return (dp_rank, tp_rank)

    def sync_barrier():
        for i in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(i),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    def start_fetch_topk(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        bt_start = bt_id * bt
        topk_bits = jnp.dtype(topk_weights_hbm.dtype).itemsize * 8
        tile0 = math.gcd(256 // topk_bits, local_num_tokens)
        bt_start = pl.multiple_of(bt_start, tile0)
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

    # -- Metadata (Step 4) --

    def all_reduce_metadata(*, bt_id, bt_sem_id, t2e_routing):
        assert metadata_starts_hbm is not None

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
                sem=local_sems.at[bt_sem_id, 2],
            )
            sizes_load = pltpu.async_copy(
                src_ref=metadata_sizes_hbm.at[bt_id],
                dst_ref=sizes_vmem,
                sem=local_sems.at[bt_sem_id, 3],
            )
            d2e_count_load = pltpu.async_copy(
                src_ref=metadata_d2e_counts_hbm.at[bt_id],
                dst_ref=d2e_count_vmem,
                sem=local_sems.at[bt_sem_id, 4],
            )

            offsets_copy = pltpu.async_copy(
                src_ref=offsets_vmem,
                dst_ref=expert_offsets_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 5],
            )
            t2e_routing_copy = pltpu.async_copy(
                src_ref=t2e_routing_vmem,
                dst_ref=t2e_routing_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 6],
            )

            starts_load.wait()
            sizes_load.wait()
            d2e_count_load.wait()
            starts_copy = pltpu.async_copy(
                src_ref=starts_vmem,
                dst_ref=expert_starts_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 2],
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_vmem,
                dst_ref=expert_sizes_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 3],
            )
            d2e_count_copy = pltpu.async_copy(
                src_ref=d2e_count_vmem,
                dst_ref=d2e_count_x2_smem.at[bt_sem_id],
                sem=local_sems.at[bt_sem_id, 4],
            )

            t2e_routing_copy.wait()
            d2e_count_copy.wait()
            offsets_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

        pl.run_scoped(
            _copy_precomputed,
            pltpu.VMEM(t2e_routing_x2_smem.shape[1:], t2e_routing_x2_smem.dtype),
            pltpu.VMEM(d2e_count_x2_smem.shape[1:], d2e_count_x2_smem.dtype),
            pltpu.VMEM(expert_offsets_x2_smem.shape[1:], expert_offsets_x2_smem.dtype),
            pltpu.VMEM(expert_starts_x2_smem.shape[1:], expert_starts_x2_smem.dtype),
            pltpu.VMEM(expert_sizes_x2_smem.shape[1:], expert_sizes_x2_smem.dtype),
        )

    # -- Scatter to VMEM (Step 5) --

    def start_a2a_scatter_to_vmem(*, bt_sem_id, x_buf_id, local_e_id, bt_start):
        def _scatter_one(t_id, send_sz):
            src_t_id = bt_start + t_id
            for k_id in range(top_k):
                e_id = t2e_routing_x2_smem[bt_sem_id, t_id, k_id]
                is_valid = e_id >= 0
                e_id_safe = lax.select(is_valid, e_id, jnp.int32(0))
                recv_id = e_id_safe // jnp.int32(local_num_experts)
                local_e = e_id_safe % jnp.int32(local_num_experts)
                is_mine = local_e == local_e_id

                offset = expert_offsets_x2_smem[bt_sem_id, 0, e_id_safe]
                dest_pos = expert_starts_x2_smem[bt_sem_id, 0, e_id_safe] + offset

                @pl.when(is_valid & is_mine)
                def _():
                    expert_offsets_x2_smem[bt_sem_id, 0, e_id_safe] = offset + 1

                is_local = recv_id == my_id

                @pl.when(is_valid & is_mine & is_local)
                def _():
                    pltpu.make_async_copy(
                        src_ref=tokens_hbm.at[pl.ds(src_t_id, 1)],
                        dst_ref=b_x_x2_vmem.at[x_buf_id, pl.ds(dest_pos, 1)],
                        sem=x_recv_sems.at[x_buf_id],
                    ).start()

                @pl.when(is_valid & is_mine & ~is_local)
                def _():
                    pltpu.make_async_remote_copy(
                        src_ref=tokens_hbm.at[pl.ds(src_t_id, 1)],
                        dst_ref=b_x_x2_vmem.at[x_buf_id, pl.ds(dest_pos, 1)],
                        send_sem=x_send_sems.at[x_buf_id],
                        recv_sem=x_recv_sems.at[x_buf_id],
                        device_id=get_mesh_device_id(recv_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                send_sz = send_sz + lax.select(
                    is_valid & is_mine & ~is_local, jnp.int32(1), jnp.int32(0)
                )
            return send_sz

        send_sz = lax.fori_loop(0, bt, _scatter_one, jnp.int32(0), unroll=False)
        a2a_s_sends_x2_smem[x_buf_id] = send_sz

    def wait_a2a_scatter_recv(*, bt_sem_id, x_buf_id, local_e_id):
        e_id = my_id * jnp.int32(local_num_experts) + local_e_id
        total = expert_sizes_x2_smem[bt_sem_id, 0, e_id]

        @pl.when(total != 0)
        def _():
            def _wait_one(_, __):
                pltpu.make_async_copy(
                    src_ref=b_x_x2_vmem.at[x_buf_id, pl.ds(0, 1)],
                    dst_ref=b_x_x2_vmem.at[x_buf_id, pl.ds(0, 1)],
                    sem=x_recv_sems.at[x_buf_id],
                ).wait()
                return None
            lax.fori_loop(0, total, _wait_one, None, unroll=False)

    def wait_a2a_scatter_send(*, x_buf_id):
        send_sz = a2a_s_sends_x2_smem[x_buf_id]

        @pl.when(send_sz != 0)
        def _():
            def _wait_one(_, __):
                pltpu.make_async_remote_copy(
                    src_ref=tokens_hbm.at[pl.ds(0, 1)],
                    dst_ref=b_x_x2_vmem.at[x_buf_id, pl.ds(0, 1)],
                    send_sem=x_send_sems.at[x_buf_id],
                    recv_sem=x_recv_sems.at[x_buf_id],
                    device_id=get_mesh_device_id(0),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).wait()
                return None
            lax.fori_loop(0, send_sz, _wait_one, None, unroll=False)

    # -- Gather (Step 6) --

    def start_a2a_gather(*, bt_sem_id, e_sem_id, local_e_id):
        my_e_id = my_id * jnp.int32(local_num_experts) + local_e_id
        start = 0
        for recv_id in range(num_devices):
            sz = d2e_count_x2_smem[bt_sem_id, recv_id, 0, my_e_id]
            is_local = recv_id == my_id
            local_sz = lax.select(is_local, sz, jnp.int32(0))
            remote_sz = lax.select(is_local, jnp.int32(0), sz)

            @pl.when(local_sz != 0)
            def _(start=start, local_sz=local_sz, my_e_id=my_e_id, e_sem_id=e_sem_id):
                pltpu.make_async_copy(
                    src_ref=a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(start, local_sz)],
                    dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, local_sz)],
                    sem=a2a_gather_sem,
                ).start()

            @pl.when(remote_sz != 0)
            def _(
                start=start, remote_sz=remote_sz, my_e_id=my_e_id,
                e_sem_id=e_sem_id, recv_id=recv_id,
            ):
                pltpu.make_async_remote_copy(
                    src_ref=a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(start, remote_sz)],
                    dst_ref=a2a_g_hbm.at[my_e_id, pl.ds(0, remote_sz)],
                    send_sem=gather_send_x2_sems.at[e_sem_id],
                    recv_sem=a2a_gather_sem,
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

            start += sz

    def wait_a2a_gather_recv_all(*, bt_sem_id):
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

    def wait_a2a_gather_send(*, bt_sem_id, e_sem_id, local_e_id):
        my_e_id = my_id * jnp.int32(local_num_experts) + local_e_id
        sz = expert_sizes_x2_smem[bt_sem_id, 0, my_e_id]
        local_sz = d2e_count_x2_smem[bt_sem_id, my_id, 0, my_e_id]
        remote_sz = sz - local_sz
        is_valid = jnp.logical_and(
            local_e_id >= 0, local_e_id < local_num_experts
        )
        remote_sz = lax.select(is_valid, remote_sz, jnp.int32(0))

        @pl.when(remote_sz != 0)
        def _():
            ref = a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(0, remote_sz)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=gather_send_x2_sems.at[e_sem_id],
            ).wait()

    # -- Expert FFN v2 (Step 7) --

    def start_fetch_w1(w_slot, local_e_id, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[local_e_id, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w1_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 0],
        ).start(priority=priority)

    def wait_fetch_w1(w_slot):
        pltpu.make_async_copy(
            src_ref=b_w1_x2_vmem.at[w_slot],
            dst_ref=b_w1_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 0],
        ).wait()

    def start_fetch_w3(w_slot, local_e_id, tile_idx, priority=1):
        pltpu.make_async_copy(
            src_ref=w3_hbm.at[local_e_id, :, pl.ds(tile_idx * bf, bf)],
            dst_ref=b_w3_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 1],
        ).start(priority=priority)

    def wait_fetch_w3(w_slot):
        pltpu.make_async_copy(
            src_ref=b_w3_x2_vmem.at[w_slot],
            dst_ref=b_w3_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 1],
        ).wait()

    def start_fetch_w2(w_slot, local_e_id, tile_idx, priority=0):
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[local_e_id, pl.ds(tile_idx * bf, bf), :],
            dst_ref=b_w2_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 2],
        ).start(priority=priority)

    def wait_fetch_w2(w_slot):
        pltpu.make_async_copy(
            src_ref=b_w2_x2_vmem.at[w_slot],
            dst_ref=b_w2_x2_vmem.at[w_slot],
            sem=weight_sems.at[w_slot, 2],
        ).wait()

    def compute_tile(x_buf_id, w_slot, is_first_tile):
        x = b_x_x2_vmem[x_buf_id].reshape(a2a_max_tokens, hidden_size)
        w1 = b_w1_x2_vmem[w_slot]
        w3 = b_w3_x2_vmem[w_slot]
        gate = jnp.dot(x, w1, preferred_element_type=jnp.float32)
        up = jnp.dot(x, w3, preferred_element_type=jnp.float32)
        act = activation_fn(gate, up, act_fn)
        wait_fetch_w2(w_slot)
        w2 = b_w2_x2_vmem[w_slot]
        partial = jnp.dot(act, w2, preferred_element_type=jnp.float32)
        if is_first_tile:
            b_y_acc_vmem[...] = partial
        else:
            b_y_acc_vmem[...] = b_y_acc_vmem[...] + partial

    def expert_ffn_v2(bt_sem_id, x_buf_id, e_sem_id, local_e_id):
        e_id = my_id * jnp.int32(local_num_experts) + local_e_id
        dyn_sz = expert_sizes_x2_smem[bt_sem_id, 0, e_id].astype(jnp.int32)
        has_tokens = dyn_sz != 0

        def _run_active(_):
            start_fetch_w1(0, local_e_id, 0, priority=1)
            start_fetch_w3(0, local_e_id, 0, priority=1)
            start_fetch_w2(0, local_e_id, 0, priority=1)
            if n_w >= 2:
                start_fetch_w1(1, local_e_id, 1)
                start_fetch_w3(1, local_e_id, 1)
                start_fetch_w2(1, local_e_id, 1)

            wait_fetch_w1(0)
            wait_fetch_w3(0)
            compute_tile(x_buf_id, slot=0, is_first_tile=True)

            for tile in range(1, n_w - 1):
                slot = tile % 2
                next_slot = 1 - slot
                start_fetch_w1(next_slot, local_e_id, tile + 1)
                start_fetch_w3(next_slot, local_e_id, tile + 1)
                start_fetch_w2(next_slot, local_e_id, tile + 1)
                wait_fetch_w1(slot)
                wait_fetch_w3(slot)
                compute_tile(x_buf_id, slot, is_first_tile=False)

            if n_w >= 2:
                last_slot = (n_w - 1) % 2
                wait_fetch_w1(last_slot)
                wait_fetch_w3(last_slot)
                compute_tile(x_buf_id, last_slot, is_first_tile=False)

            b_y_out_vmem[...] = b_y_acc_vmem[...].astype(t_dtype).reshape(
                a2a_max_tokens, t_packing, h_per_t
            )
            pltpu.make_async_copy(
                src_ref=b_y_out_vmem,
                dst_ref=a2a_s_acc_x2_hbm.at[e_sem_id, pl.ds(0, a2a_max_tokens)],
                sem=y_out_sem.at[0],
            ).start()
            pltpu.make_async_copy(
                src_ref=b_y_out_vmem,
                dst_ref=b_y_out_vmem,
                sem=y_out_sem.at[0],
            ).wait()

        lax.cond(has_tokens, _run_active, lambda _: None, operand=None)

    # -- Accumulation & output store (Step 8) --

    def acc_and_store_output(*, bt_sem_id, out_buf_id):
        acc_bt = a2a_g_acc_vmem.shape[2]
        assert bt % acc_bt == 0
        num_acc_tiles = bt // acc_bt

        def start_load_acc_bt(*, tile_start, buf_id):
            def _load_one(t_i, _):
                t_id = tile_start + t_i
                token_e0 = t2e_routing_x2_smem[bt_sem_id, t_id, 0]
                is_valid_token = token_e0 >= 0

                @pl.when(is_valid_token)
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

                @pl.when(jnp.logical_not(is_valid_token))
                def _():
                    zeros = jnp.zeros(
                        (1, t_packing, h_per_t), dtype=a2a_g_acc_vmem.dtype
                    )
                    for k_id in range(top_k):
                        a2a_g_acc_vmem.at[buf_id, k_id, pl.ds(t_i, 1)][...] = zeros

                return None

            for t_i in range(acc_bt):
                _load_one(t_i, None)

        def wait_load_acc_bt(*, buf_id, tile_start):
            def _count_valid(t_i, acc):
                token_e0 = t2e_routing_x2_smem[bt_sem_id, tile_start + t_i, 0]
                return acc + (token_e0 >= 0).astype(jnp.int32)

            num_valid = lax.fori_loop(
                0, acc_bt, _count_valid, jnp.int32(0), unroll=False,
            )

            @pl.when(num_valid != 0)
            def _():
                def _wait_one(_, __):
                    ref = a2a_g_acc_vmem.at[buf_id, 0, pl.ds(0, 1)]
                    pltpu.make_async_copy(
                        src_ref=ref, dst_ref=ref, sem=a2a_acc_sems.at[0],
                    ).wait()
                    return None
                lax.fori_loop(
                    0, num_valid * jnp.int32(top_k), _wait_one, None, unroll=False,
                )

        def acc_gather_to_output(*, tile_start, out_offset, buf_id):
            output_tile = jnp.zeros((acc_bt, hidden_size), dtype=jnp.float32)
            logits_tile = b_topk_weights_x2_vmem[
                bt_sem_id, pl.ds(tile_start, acc_bt), pl.ds(0, top_k)
            ]
            for k_id in range(top_k):
                acc_tile = a2a_g_acc_vmem[buf_id, k_id, :acc_bt].reshape(
                    acc_bt, hidden_size
                ).astype(jnp.float32)
                logits = logits_tile[:, k_id].reshape(acc_bt, 1)
                output_tile += acc_tile * logits

            out_offset = pl.multiple_of(out_offset, 16)
            target_slice = b_output_x2_vmem.at[
                out_buf_id, pl.ds(out_offset, acc_bt)
            ]
            target_slice[...] = output_tile.astype(output_hbm.dtype).reshape(
                acc_bt, t_packing, h_per_t
            )

        start_load_acc_bt(tile_start=0, buf_id=0)

        def run_acc_pipeline(i, _):
            curr_buf_id = i % 2
            next_buf_id = (i + 1) % 2
            curr_tile_start = i * acc_bt
            next_tile_start = (i + 1) * acc_bt
            out_offset = i * acc_bt

            @pl.when(i + 1 < num_acc_tiles)
            def _():
                start_load_acc_bt(tile_start=next_tile_start, buf_id=next_buf_id)

            wait_load_acc_bt(buf_id=curr_buf_id, tile_start=curr_tile_start)
            acc_gather_to_output(
                tile_start=curr_tile_start, out_offset=out_offset, buf_id=curr_buf_id,
            )
            return None

        lax.fori_loop(0, num_acc_tiles, run_acc_pipeline, None, unroll=False)

    def start_send_bo(*, bt_id, priority=0):
        bt_sem_id = bt_id & jnp.int32(1)
        bt_start = bt_id * bt
        pltpu.make_async_copy(
            src_ref=b_output_x2_vmem.at[bt_sem_id],
            dst_ref=output_hbm.at[pl.ds(bt_start, bt)],
            sem=local_sems.at[bt_sem_id, 1],
        ).start(priority=priority)

    def wait_store_output(*, bt_id):
        is_valid = jnp.logical_and(bt_id >= 0, bt_id < num_bt)
        bt_sem_id = bt_id & 1

        @pl.when(is_valid)
        def _():
            bt_start = bt_id * bt
            ref = output_hbm.at[pl.ds(bt_start, bt)]
            pltpu.make_async_copy(
                src_ref=ref, dst_ref=ref, sem=local_sems.at[bt_sem_id, 1],
            ).wait()

    # -- Pipelined run_bt (Step 9) --

    def run_bt(bt_id, e_sem_id):
        bt_start = bt_id * bt
        bt_sem_id = bt_id & jnp.int32(1)

        @pl.when(bt_id + 1 < num_bt)
        def _():
            start_fetch_topk(bt_id=bt_id + 1)

        wait_fetch_topk(bt_id=bt_id)
        t2e_routing = b_topk_ids_x2_vmem[bt_sem_id]
        all_reduce_metadata(
            bt_id=bt_id, bt_sem_id=bt_sem_id, t2e_routing=t2e_routing,
        )
        wait_store_output(bt_id=bt_id - 2)

        start_a2a_scatter_to_vmem(
            bt_sem_id=bt_sem_id, x_buf_id=0, local_e_id=jnp.int32(0),
            bt_start=bt_start,
        )

        init_carry = (e_sem_id, jnp.int32(0))

        def run_per_expert(local_e_id, carry):
            curr_e_sem_id, curr_x_buf = carry
            next_e_sem_id = (curr_e_sem_id + 1) % expert_buffer_count
            next_x_buf = jnp.int32(1) - curr_x_buf

            @pl.when(local_e_id + 1 < local_num_experts)
            def _():
                @pl.when(
                    (local_e_id + 1 >= expert_buffer_count)
                    & ((local_e_id + 1) % expert_buffer_count == 0)
                )
                def _():
                    sync_barrier()

                start_a2a_scatter_to_vmem(
                    bt_sem_id=bt_sem_id, x_buf_id=next_x_buf,
                    local_e_id=local_e_id + 1, bt_start=bt_start,
                )

            wait_a2a_scatter_recv(
                bt_sem_id=bt_sem_id, x_buf_id=curr_x_buf, local_e_id=local_e_id,
            )
            expert_ffn_v2(bt_sem_id, curr_x_buf, curr_e_sem_id, local_e_id)
            start_a2a_gather(
                bt_sem_id=bt_sem_id, e_sem_id=curr_e_sem_id, local_e_id=local_e_id,
            )
            wait_a2a_scatter_send(x_buf_id=curr_x_buf)

            return (next_e_sem_id, next_x_buf)

        final_carry = lax.fori_loop(
            0, local_num_experts, run_per_expert, init_carry, unroll=False,
        )
        final_e_sem_id = final_carry[0]

        wait_a2a_gather_recv_all(bt_sem_id=bt_sem_id)
        sync_barrier()
        acc_and_store_output(bt_sem_id=bt_sem_id, out_buf_id=bt_sem_id)
        start_send_bo(bt_id=bt_id)

        tail_start = max(local_num_experts - expert_buffer_count, 0)
        for tail_e in range(tail_start, local_num_experts):
            tail_sem = (e_sem_id + tail_e) % expert_buffer_count
            wait_a2a_gather_send(
                bt_sem_id=bt_sem_id, e_sem_id=tail_sem, local_e_id=jnp.int32(tail_e),
            )

        @pl.when(bt_id + 1 < num_bt)
        def _():
            sync_barrier()

        return final_e_sem_id

    # -- Kernel entry point --
    sync_barrier()
    start_fetch_topk(bt_id=0)

    lax.fori_loop(0, num_bt, run_bt, jnp.int32(0), unroll=False)

    wait_store_output(bt_id=jnp.int32(num_bt - 1))
    wait_store_output(bt_id=jnp.int32(num_bt - 2))


# ---------------------------------------------------------------------------
# Outer function
# ---------------------------------------------------------------------------

_A2A_HBM_FRACTION = 0.03


@functools.lru_cache(maxsize=1)
def _device_hbm_bytes() -> int:
    return jax.local_devices()[0].memory_stats()["bytes_limit"]


def get_ep_size(mesh: jax.sharding.Mesh, dp_axis_name, tp_axis_name):
    assert len(mesh.shape) == 2
    dp_size = mesh.shape[dp_axis_name]
    tp_size = mesh.shape[tp_axis_name]
    return dp_size * tp_size


@functools.partial(
    jax.jit,
    static_argnames=[
        "mesh",
        "top_k",
        "act_fn",
        "bt",
        "bf",
        "dp_axis_name",
        "tp_axis_name",
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
    bt: int = 16,
    bf: int = 256,
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
):
    ep_size = get_ep_size(mesh, dp_axis_name, tp_axis_name)
    num_devices = ep_size

    num_tokens, hidden_size = tokens.shape
    num_experts, intermediate_size, _ = w2.shape
    local_num_experts = num_experts // ep_size
    local_num_tokens = num_tokens // ep_size

    bt = min(bt, local_num_tokens)
    bt = math.gcd(bt, local_num_tokens)
    if bt <= 0:
        raise ValueError(f"Expected {bt=} > 0 after clamping.")

    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    t_dtype = tokens.dtype
    t_packing = get_dtype_packing(t_dtype)
    h_per_t = hidden_size // t_packing
    a2a_max_tokens = align_to(bt * num_devices, 16)
    acc_bt = math.gcd(bt, 16)

    a2a_scratch_budget = int(_device_hbm_bytes() * _A2A_HBM_FRACTION)
    bytes_per_slot = a2a_max_tokens * hidden_size * jnp.dtype(t_dtype).itemsize
    expert_buffer_count = min(
        local_num_experts, max(2, a2a_scratch_budget // bytes_per_slot)
    )

    if padded_top_k > top_k:
        topk_ids = jnp.pad(
            topk_ids,
            ((0, 0), (0, padded_top_k - top_k)),
            mode="constant",
            constant_values=-1,
        )
        topk_weights = jnp.pad(
            topk_weights,
            ((0, 0), (0, padded_top_k - top_k)),
            mode="constant",
            constant_values=0,
        )

    needs_jax_allreduce = True

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    scope_name = (
        f"fused-moe-v2-k{top_k}-bt{bt}-bf{bf}"
        f"-d{hidden_size}-f{intermediate_size}"
    )

    scratch_shapes = (
        pltpu.SMEM((2, bt, padded_top_k), jnp.int32),
        pltpu.SMEM((2, num_devices, 1, padded_num_experts), jnp.int32),
        pltpu.SMEM((2, 2, padded_num_experts), jnp.int32),
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
        pltpu.SMEM((2, 1, padded_num_experts), jnp.int32),
        pltpu.SMEM((2,), jnp.int32),
        pltpu.VMEM((2, a2a_max_tokens, t_packing, h_per_t), t_dtype),
        pltpu.VMEM((a2a_max_tokens, hidden_size), jnp.float32),
        pltpu.VMEM((a2a_max_tokens, t_packing, h_per_t), t_dtype),
        pltpu.VMEM((2, hidden_size, bf), w1.dtype),
        pltpu.VMEM((2, hidden_size, bf), w3.dtype),
        pltpu.VMEM((2, bf, hidden_size), w2.dtype),
        pltpu.VMEM((2, top_k, acc_bt, t_packing, h_per_t), t_dtype),
        pltpu.VMEM((2, bt, padded_top_k), jnp.float32),
        pltpu.VMEM((2, bt, padded_top_k), jnp.int32),
        pltpu.VMEM((2, bt, t_packing, h_per_t), t_dtype),
        pltpu.SemaphoreType.DMA((2, 3)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((2, 7)),
        pltpu.SemaphoreType.DMA((expert_buffer_count,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.BARRIER,
    )

    fused_moe = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _fused_ep_moe_v2_kernel,
                top_k=top_k,
                dp_axis_name=dp_axis_name,
                tp_axis_name=tp_axis_name,
                act_fn=act_fn,
                bt=bt,
                bf=bf,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (local_num_tokens, t_packing, h_per_t), t_dtype
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm,  # tokens
                    hbm,  # w1
                    hbm,  # w2
                    hbm,  # w3
                    hbm,  # topk_weights
                    hbm,  # topk_ids
                    hbm,  # a2a_s_acc_x2
                    hbm,  # a2a_g
                    hbm if needs_jax_allreduce else None,  # metadata_starts
                    hbm if needs_jax_allreduce else None,  # metadata_sizes
                    hbm if needs_jax_allreduce else None,  # metadata_d2e_counts
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

    ep_spec = P((dp_axis_name, tp_axis_name))

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            ep_spec,  # tokens
            ep_spec,  # w1
            ep_spec,  # w2
            ep_spec,  # w3
            ep_spec,  # topk_weights
            ep_spec,  # topk_ids
            P(),      # a2a_s_acc scratch
            P(),      # a2a_g scratch
        ),
        out_specs=ep_spec,
        check_vma=False,
    )
    def kernel(
        tokens, w1, w2, w3, topk_weights, topk_ids,
        a2a_s_acc_scratch, a2a_g_scratch,
    ):
        tokens_3d = tokens.reshape(local_num_tokens, t_packing, h_per_t)

        if needs_jax_allreduce:
            metadata_starts, metadata_sizes, metadata_d2e_counts = (
                jax_allreduce_metadata_by_bt(
                    topk_ids[:, :top_k],
                    padded_num_experts,
                    bt,
                    num_devices,
                    dp_axis_name,
                    tp_axis_name,
                )
            )
            ms_arg = pltpu.with_memory_space_constraint(metadata_starts, pltpu.HBM)
            msz_arg = pltpu.with_memory_space_constraint(metadata_sizes, pltpu.HBM)
            md_arg = pltpu.with_memory_space_constraint(metadata_d2e_counts, pltpu.HBM)
        else:
            ms_arg = None
            msz_arg = None
            md_arg = None

        result_3d = fused_moe(
            pltpu.with_memory_space_constraint(tokens_3d, pltpu.HBM),
            pltpu.with_memory_space_constraint(w1, pltpu.HBM),
            pltpu.with_memory_space_constraint(w2, pltpu.HBM),
            pltpu.with_memory_space_constraint(w3, pltpu.HBM),
            pltpu.with_memory_space_constraint(topk_weights, pltpu.HBM),
            pltpu.with_memory_space_constraint(topk_ids, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_s_acc_scratch, pltpu.HBM),
            pltpu.with_memory_space_constraint(a2a_g_scratch, pltpu.HBM),
            ms_arg,
            msz_arg,
            md_arg,
        )
        return result_3d.reshape(local_num_tokens, hidden_size)

    a2a_s_acc_scratch = pl.empty(
        (expert_buffer_count, a2a_max_tokens, t_packing, h_per_t), t_dtype,
    )
    a2a_g_scratch = pl.empty(
        (num_experts, bt, t_packing, h_per_t), t_dtype,
    )

    return kernel(
        tokens, w1, w2, w3, topk_weights, topk_ids,
        a2a_s_acc_scratch, a2a_g_scratch,
    )


# ---------------------------------------------------------------------------
# Reference implementation (for correctness testing)
# ---------------------------------------------------------------------------


def ref_moe_simple(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    top_k: int,
    *,
    act_fn: str = "silu",
) -> jax.Array:
    num_tokens, hidden_size = tokens.shape
    num_experts = w1.shape[0]

    output = jnp.zeros((num_tokens, hidden_size), dtype=jnp.float32)
    for i in range(num_tokens):
        tok = tokens[i : i + 1].astype(jnp.float32)
        for k in range(top_k):
            eid = topk_ids[i, k]
            wt = topk_weights[i, k]
            gate = tok @ w1[eid].astype(jnp.float32)
            up = tok @ w3[eid].astype(jnp.float32)
            act = activation_fn(gate, up, act_fn)
            down = act @ w2[eid].astype(jnp.float32)
            output = output.at[i].add(down[0] * wt)

    return output.astype(tokens.dtype)


# ---------------------------------------------------------------------------
# __main__: single-device correctness test (ep_size=1)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    configs = [
        {"name": "small-test", "d": 768, "f": 256, "E": 4, "top_k": 2},
        {"name": "MiMo-V2-Pro-EP1", "d": 6144, "f": 2048, "E": 16, "top_k": 8},
    ]
    bt_sizes = [16]
    bf_arg = 256

    num_devices = jax.device_count()
    print(f"Devices: {num_devices}")

    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(1, 1), ("data", "tensor")
    )

    for cfg in configs:
        d, f = cfg["d"], cfg["f"]
        E = cfg["E"]
        top_k = cfg["top_k"]

        if f % bf_arg != 0:
            bf_test = 128
        else:
            bf_test = bf_arg

        for bt in bt_sizes:
            num_tokens = max(bt, 16)
            key = jax.random.key(42)
            k1, k2, k3, k4, k5 = jax.random.split(key, 5)

            tokens = jax.random.normal(k1, (num_tokens, d), dtype=jnp.bfloat16)
            w1_arr = jax.random.normal(k2, (E, d, f), dtype=jnp.bfloat16) * 0.01
            w2_arr = jax.random.normal(k3, (E, f, d), dtype=jnp.bfloat16) * 0.01
            w3_arr = jax.random.normal(k4, (E, d, f), dtype=jnp.bfloat16) * 0.01

            gating = jax.random.normal(k5, (num_tokens, E), dtype=jnp.float32)
            _, topk_idx = lax.top_k(gating, top_k)
            topk_logits = jnp.take_along_axis(gating, topk_idx, axis=-1)
            topk_wts = jax.nn.softmax(topk_logits, axis=-1)

            ep_sharding = jax.sharding.NamedSharding(
                mesh, P(("data", "tensor"))
            )
            rep_sharding = jax.sharding.NamedSharding(mesh, P())

            tokens_s = jax.device_put(tokens, ep_sharding)
            w1_s = jax.device_put(w1_arr, ep_sharding)
            w2_s = jax.device_put(w2_arr, ep_sharding)
            w3_s = jax.device_put(w3_arr, ep_sharding)
            topk_wts_s = jax.device_put(topk_wts, ep_sharding)
            topk_idx_s = jax.device_put(topk_idx, ep_sharding)

            result = fused_ep_moe_v2(
                mesh, tokens_s, w1_s, w2_s, w3_s,
                topk_wts_s, topk_idx_s, top_k,
                bt=bt, bf=bf_test,
            )
            ref = ref_moe_simple(
                tokens, w1_arr, w2_arr, w3_arr,
                topk_wts, topk_idx, top_k,
            )

            result_f32 = result.astype(jnp.float32)
            ref_f32 = ref.astype(jnp.float32)
            max_err = jnp.max(jnp.abs(result_f32 - ref_f32))
            rel_err = max_err / (jnp.max(jnp.abs(ref_f32)) + 1e-6)

            print(
                f"{cfg['name']} bt={bt} E={E} k={top_k} d={d} f={f} bf={bf_test}: "
                f"max_abs_err={max_err:.4f}, rel_err={rel_err:.6f}"
            )
            if rel_err > 0.05:
                print(f"  FAIL (rel_err too high)")
                sys.exit(1)
            print(f"  PASS")

    print("\nAll tests passed.")
