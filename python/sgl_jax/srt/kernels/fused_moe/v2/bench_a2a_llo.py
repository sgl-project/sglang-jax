"""PROTOTYPE: standalone fused-MoE v2 scatter/gather communication probe.

Question answered by this throwaway prototype:
    What remote-DMA topology and payload does the current fused-MoE v2
    ``a2a_scatter`` / ``a2a_gather`` path present to the TPU compiler when
    GLM-5.2 runs with EP=32?

The payload communication below is extracted from ``kernel.py``.  Metadata
all-reduce, expert compute, and final accumulation are deliberately excluded:
the host builds the same per-BT ``d2e_count``, expert start, and expert size
tables, then the two Pallas calls execute the original remote-copy shapes and
semaphore protocol independently.

This file is a diagnostic prototype, not a second fused-MoE implementation.
Delete it after the compiler investigation or absorb the useful reporting into
a maintained benchmark.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

jax = None
jnp = None
lax = None
pl = None
pltpu = None
P = None


def load_jax() -> None:
    global P, jax, jnp, lax, pl, pltpu
    import jax as jax_module
    import jax.numpy as jnp_module
    from jax import lax as lax_module
    from jax.experimental import pallas as pl_module
    from jax.experimental.pallas import tpu as pltpu_module

    jax = jax_module
    jnp = jnp_module
    lax = lax_module
    pl = pl_module
    pltpu = pltpu_module
    P = jax.sharding.PartitionSpec


@dataclass(frozen=True)
class CasePlan:
    tokens: int
    ep_size: int
    num_experts: int
    top_k: int
    hidden_size: int
    bt: int
    local_tokens: int
    num_bt: int
    local_experts: int
    a2a_max_tokens: int
    topk_ids: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    scatter_rows: np.ndarray
    gather_rows: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump standalone fused-MoE v2 scatter/gather LLO on an EP mesh."
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 16384])
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument(
        "--base-bt",
        type=int,
        default=32,
        help="GLM-5.2 falls back to fused-MoE v2's bt=32 default.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="First logical mesh dimension; TP is inferred as EP / DP.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["scatter_fp8", "scatter_bf16", "gather_bf16"],
        default=["scatter_fp8", "scatter_bf16", "gather_bf16"],
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--sample-runs", type=int, default=5)
    parser.add_argument("--output-dir", default="/tmp/fused-moe-a2a-llo")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write logical traffic plans without initializing TPU/JAX distributed runtime.",
    )
    return parser.parse_args()


def resolve_bt(tokens: int, ep_size: int, base_bt: int) -> int:
    if tokens % ep_size:
        raise ValueError(f"{tokens=} must be divisible by {ep_size=}")
    local_tokens = tokens // ep_size
    return int(np.gcd(min(base_bt, local_tokens), local_tokens))


def build_case_plan(
    *,
    tokens: int,
    ep_size: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    base_bt: int,
) -> CasePlan:
    if num_experts % ep_size:
        raise ValueError(f"{num_experts=} must be divisible by {ep_size=}")
    if hidden_size % 128:
        raise ValueError(f"{hidden_size=} must be aligned to 128")
    if top_k > num_experts:
        raise ValueError(f"{top_k=} cannot exceed {num_experts=}")

    local_tokens = tokens // ep_size
    bt = resolve_bt(tokens, ep_size, base_bt)
    num_bt = local_tokens // bt
    local_experts = num_experts // ep_size
    a2a_max_tokens = bt * ep_size

    global_token_ids = np.arange(tokens, dtype=np.int64).reshape(ep_size, local_tokens)
    topk_offsets = np.arange(top_k, dtype=np.int64)
    topk_ids = (global_token_ids[:, :, None] * top_k + topk_offsets[None, None, :]) % num_experts
    topk_ids = topk_ids.astype(np.int32)

    counts = np.zeros((num_bt, ep_size, num_experts), dtype=np.int32)
    for bt_id in range(num_bt):
        token_slice = slice(bt_id * bt, (bt_id + 1) * bt)
        for src_rank in range(ep_size):
            counts[bt_id, src_rank] = np.bincount(
                topk_ids[src_rank, token_slice].reshape(-1),
                minlength=num_experts,
            )

    starts = np.cumsum(counts, axis=1, dtype=np.int32) - counts
    sizes = np.sum(counts, axis=1, dtype=np.int32)

    scatter_rows = np.zeros((ep_size, ep_size), dtype=np.int64)
    for src_rank in range(ep_size):
        per_expert = counts[:, src_rank, :].sum(axis=0, dtype=np.int64)
        for expert_id, rows in enumerate(per_expert):
            scatter_rows[src_rank, expert_id // local_experts] += int(rows)
    gather_rows = scatter_rows.T.copy()

    expected_rows = tokens * top_k
    if int(scatter_rows.sum()) != expected_rows:
        raise AssertionError((scatter_rows.sum(), expected_rows))
    if np.any(sizes > a2a_max_tokens):
        raise AssertionError(
            f"expert capacity overflow: max={int(sizes.max())}, capacity={a2a_max_tokens}"
        )

    return CasePlan(
        tokens=tokens,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        bt=bt,
        local_tokens=local_tokens,
        num_bt=num_bt,
        local_experts=local_experts,
        a2a_max_tokens=a2a_max_tokens,
        topk_ids=topk_ids,
        counts=counts,
        starts=starts,
        sizes=sizes,
        scatter_rows=scatter_rows,
        gather_rows=gather_rows,
    )


def logical_plan_dict(plan: CasePlan) -> dict[str, Any]:
    return {
        "tokens": plan.tokens,
        "ep_size": plan.ep_size,
        "num_experts": plan.num_experts,
        "top_k": plan.top_k,
        "hidden_size": plan.hidden_size,
        "bt": plan.bt,
        "local_tokens": plan.local_tokens,
        "num_bt": plan.num_bt,
        "local_experts": plan.local_experts,
        "a2a_max_tokens": plan.a2a_max_tokens,
        "total_routed_rows": int(plan.tokens * plan.top_k),
        "scatter_rows_matrix": plan.scatter_rows.tolist(),
        "gather_rows_matrix": plan.gather_rows.tolist(),
        "max_expert_rows_per_bt": int(plan.sizes.max()),
        "min_expert_rows_per_bt": int(plan.sizes.min()),
        "source": {
            "kernel": "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py",
            "scatter_lines": "672-780",
            "gather_lines": "784-868",
            "schedule_lines": "2047-2159",
        },
    }


def maybe_initialize_distributed() -> None:
    world_size = int(os.getenv("FALCON_WORLD_SIZE", os.getenv("JAX_PROCESS_COUNT", "1")))
    if world_size > 1:
        jax.distributed.initialize()


def _device_coords(device: Any) -> tuple[int, ...]:
    coords = getattr(device, "coords", ())
    return tuple(int(x) for x in coords)


def topology_dict(mesh: jax.sharding.Mesh) -> dict[str, Any]:
    devices = list(mesh.devices.flat)
    records = []
    for logical_rank, device in enumerate(devices):
        mesh_index = np.unravel_index(logical_rank, mesh.devices.shape)
        records.append(
            {
                "logical_rank": logical_rank,
                "mesh_index": [int(x) for x in mesh_index],
                "jax_device_id": int(device.id),
                "process_index": int(device.process_index),
                "coords": list(_device_coords(device)),
                "core_on_chip": int(getattr(device, "core_on_chip", 0)),
                "slice_index": int(getattr(device, "slice_index", 0)),
            }
        )
    coord_rows = [_device_coords(device) for device in devices]
    torus_dimensions = (
        [max(row[axis] for row in coord_rows) + 1 for axis in range(len(coord_rows[0]))]
        if coord_rows and coord_rows[0]
        else []
    )
    return {
        "mesh_axis_names": list(mesh.axis_names),
        "mesh_shape": {str(k): int(v) for k, v in mesh.shape.items()},
        "visible_devices": len(devices),
        "chips": len({(_device_coords(d), int(getattr(d, "slice_index", 0))) for d in devices}),
        "processes": len({int(d.process_index) for d in devices}),
        "torus_dimensions": torus_dimensions,
        "devices": records,
    }


def _torus_hops(
    src: tuple[int, ...],
    dst: tuple[int, ...],
    dimensions: list[int],
) -> int:
    hops = 0
    for src_coord, dst_coord, size in zip(src, dst, dimensions, strict=True):
        delta = abs(src_coord - dst_coord)
        hops += min(delta, size - delta)
    return hops


def physical_load_dict(
    *,
    plan: CasePlan,
    topology: dict[str, Any],
    phase: str,
    dtype_bytes: int,
) -> dict[str, Any]:
    rows = plan.scatter_rows if phase == "scatter" else plan.gather_rows
    payload = rows * plan.hidden_size * dtype_bytes
    device_rows = topology["devices"]
    torus_dimensions = topology["torus_dimensions"]
    process_ids = sorted({row["process_index"] for row in device_rows})
    process_slot = {process_id: slot for slot, process_id in enumerate(process_ids)}
    host_bytes = np.zeros((len(process_ids), len(process_ids)), dtype=np.int64)
    hop_bytes: dict[int, int] = {}
    class_bytes = {
        "self": 0,
        "same_chip_peer_core": 0,
        "same_host_other_chip": 0,
        "cross_host": 0,
    }

    for src_rank, src in enumerate(device_rows):
        src_coords = tuple(src["coords"])
        for dst_rank, dst in enumerate(device_rows):
            nbytes = int(payload[src_rank, dst_rank])
            host_bytes[
                process_slot[src["process_index"]],
                process_slot[dst["process_index"]],
            ] += nbytes
            if src_rank == dst_rank:
                category = "self"
                hops = 0
            elif src_coords == tuple(dst["coords"]):
                category = "same_chip_peer_core"
                hops = 0
            elif src["process_index"] == dst["process_index"]:
                category = "same_host_other_chip"
                hops = _torus_hops(src_coords, tuple(dst["coords"]), torus_dimensions)
            else:
                category = "cross_host"
                hops = _torus_hops(src_coords, tuple(dst["coords"]), torus_dimensions)
            class_bytes[category] += nbytes
            hop_bytes[hops] = hop_bytes.get(hops, 0) + nbytes

    remote_bytes = int(payload.sum() - np.trace(payload))
    hop_weighted_bytes = sum(hops * nbytes for hops, nbytes in hop_bytes.items())
    return {
        "phase": phase,
        "dtype_bytes": dtype_bytes,
        "payload_bytes_matrix": payload.tolist(),
        "host_payload_bytes_matrix": host_bytes.tolist(),
        "payload_bytes_by_link_class": class_bytes,
        "payload_bytes_by_shortest_torus_hops": {
            str(hops): nbytes for hops, nbytes in sorted(hop_bytes.items())
        },
        "total_payload_bytes_including_local": int(payload.sum()),
        "remote_payload_bytes": remote_bytes,
        "hop_weighted_payload_bytes": int(hop_weighted_bytes),
        "max_directed_rank_pair_bytes": int(payload.max()),
        "per_source_payload_bytes": payload.sum(axis=1).tolist(),
        "per_destination_payload_bytes": payload.sum(axis=0).tolist(),
    }


def _collective_id(tag: str) -> int:
    return 500_000 + zlib.crc32(tag.encode()) % 300_000


def _make_array_from_numpy(
    array: np.ndarray,
    sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])


def _make_constant_array(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    sharding: jax.sharding.NamedSharding,
) -> jax.Array:
    np_dtype = np.dtype(dtype)

    def callback(index):
        local_shape = []
        for axis, part in enumerate(index):
            if isinstance(part, slice):
                start, stop, step = part.indices(shape[axis])
                local_shape.append(len(range(start, stop, step)))
            else:
                local_shape.append(1)
        return np.ones(local_shape, dtype=np_dtype)

    return jax.make_array_from_callback(shape, sharding, callback)


def build_scatter_runner(
    *,
    mesh: jax.sharding.Mesh,
    plan: CasePlan,
    dtype: jnp.dtype,
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    ep_size = plan.ep_size
    local_experts = plan.local_experts
    num_bt = plan.num_bt
    bt = plan.bt
    routing_width = max(128, plan.top_k)
    packing = 32 // (jnp.dtype(dtype).itemsize * 8)
    h_per_pack = plan.hidden_size // packing
    use_banks = num_bt > 1
    local_out_shape = (
        (num_bt, local_experts, plan.a2a_max_tokens, packing, h_per_pack)
        if use_banks
        else (local_experts, plan.a2a_max_tokens, packing, h_per_pack)
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    tag = f"fused_moe_v2_a2a_scatter_t{plan.tokens}_{jnp.dtype(dtype).name}"

    def pallas_kernel(
        tokens_ref,
        topk_ref,
        starts_ref,
        sizes_ref,
        out_ref,
        topk_smem,
        starts_smem,
        sizes_smem,
        metadata_sems,
        offsets_smem,
        sends_smem,
        send_sems,
        recv_sems,
        barrier_sem,
    ):
        dp_rank = lax.axis_index(dp_axis_name)
        tp_rank = lax.axis_index(tp_axis_name)
        tp_size = lax.axis_size(tp_axis_name)
        my_id = dp_rank * tp_size + tp_rank

        def mesh_device_id(rank):
            return (rank // tp_size, rank % tp_size)

        def out_slice(bt_id, expert_slot, start, size):
            if use_banks:
                return out_ref.at[bt_id, expert_slot, pl.ds(start, size)]
            return out_ref.at[expert_slot, pl.ds(start, size)]

        def sync_barrier():
            for peer in range(ep_size):
                pl.semaphore_signal(
                    barrier_sem,
                    device_id=mesh_device_id(peer),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier_sem, ep_size)

        def scatter_bt(bt_id, _):
            bt_start = bt_id * bt
            topk_copy = pltpu.async_copy(
                src_ref=topk_ref.at[pl.ds(bt_start, bt)],
                dst_ref=topk_smem,
                sem=metadata_sems.at[0],
            )
            starts_copy = pltpu.async_copy(
                src_ref=starts_ref.at[bt_id],
                dst_ref=starts_smem,
                sem=metadata_sems.at[1],
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_ref.at[bt_id],
                dst_ref=sizes_smem,
                sem=metadata_sems.at[2],
            )

            def clear_offset(expert_id, _):
                offsets_smem[expert_id] = jnp.int32(0)
                return None

            def clear_send(expert_slot, _):
                sends_smem[expert_slot] = jnp.int32(0)
                return None

            lax.fori_loop(0, plan.num_experts, clear_offset, None, unroll=False)
            lax.fori_loop(0, local_experts, clear_send, None, unroll=False)
            topk_copy.wait()
            starts_copy.wait()
            sizes_copy.wait()

            def scatter_one(token_id, _):
                src_token_id = bt_start + token_id
                for k_id in range(plan.top_k):
                    expert_id = topk_smem[token_id, k_id]
                    expert_slot = expert_id % jnp.int32(local_experts)
                    recv_id = expert_id // jnp.int32(local_experts)
                    offset = offsets_smem[expert_id]
                    offsets_smem[expert_id] = offset + jnp.int32(1)
                    start = starts_smem[my_id, expert_id] + offset
                    is_local = recv_id == my_id

                    with jax.named_scope("a2a_scatter_remote_sends"):
                        sends_smem[expert_slot] += jnp.logical_not(is_local).astype(jnp.int32)

                    with jax.named_scope("a2a_scatter_local_copy"):

                        @pl.when(is_local)
                        def local_copy(
                            _src_token_id=src_token_id,
                            _expert_slot=expert_slot,
                            _start=start,
                        ):
                            pltpu.make_async_copy(
                                src_ref=tokens_ref.at[pl.ds(_src_token_id, 1)],
                                dst_ref=out_slice(bt_id, _expert_slot, _start, 1),
                                sem=recv_sems.at[_expert_slot],
                            ).start()

                    with jax.named_scope("a2a_scatter_remote_copy"):

                        @pl.when(jnp.logical_not(is_local))
                        def remote_copy(
                            _src_token_id=src_token_id,
                            _expert_slot=expert_slot,
                            _start=start,
                            _recv_id=recv_id,
                        ):
                            pltpu.make_async_remote_copy(
                                src_ref=tokens_ref.at[pl.ds(_src_token_id, 1)],
                                dst_ref=out_slice(bt_id, _expert_slot, _start, 1),
                                send_sem=send_sems.at[_expert_slot],
                                recv_sem=recv_sems.at[_expert_slot],
                                device_id=mesh_device_id(_recv_id),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start()

                return None

            with jax.named_scope("a2a_scatter"):
                lax.fori_loop(0, bt, scatter_one, None, unroll=False)

            with jax.named_scope("a2a_scatter_send_wait"):

                def wait_send(expert_slot, _):
                    send_rows = sends_smem[expert_slot]

                    @pl.when(send_rows != 0)
                    def wait(_expert_slot=expert_slot, _send_rows=send_rows):
                        ref = out_slice(bt_id, _expert_slot, 0, _send_rows)
                        pltpu.make_async_copy(
                            src_ref=ref,
                            dst_ref=ref,
                            sem=send_sems.at[_expert_slot],
                        ).wait()

                    return None

                lax.fori_loop(0, local_experts, wait_send, None, unroll=False)

            with jax.named_scope("a2a_scatter_recv_wait"):

                def wait_recv(expert_slot, _):
                    expert_id = my_id * local_experts + expert_slot
                    recv_rows = sizes_smem[0, expert_id]

                    @pl.when(recv_rows != 0)
                    def wait(_expert_slot=expert_slot, _recv_rows=recv_rows):
                        ref = out_slice(bt_id, _expert_slot, 0, _recv_rows)
                        pltpu.make_async_copy(
                            src_ref=ref,
                            dst_ref=ref,
                            sem=recv_sems.at[_expert_slot],
                        ).wait()

                    return None

                lax.fori_loop(0, local_experts, wait_recv, None, unroll=False)
            sync_barrier()
            return None

        sync_barrier()
        lax.fori_loop(0, num_bt, scatter_bt, None, unroll=False)

    scatter_call = pl.pallas_call(
        pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(local_out_shape, dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            # The production kernel stages routing metadata from HBM through
            # VMEM into SMEM before dynamically indexing it.  Keeping the
            # payload in HBM and metadata in VMEM preserves the communication
            # operation while satisfying Mosaic's reference-load rules.
            in_specs=[hbm_spec, vmem_spec, vmem_spec, vmem_spec],
            out_specs=hbm_spec,
            scratch_shapes=[
                # Match fused-MoE v2's default top-k padding so the VMEM-to-SMEM
                # metadata DMA is aligned to TPU's 128-column tiling.
                pltpu.SMEM((bt, routing_width), jnp.int32),
                pltpu.SMEM((ep_size, plan.num_experts), jnp.int32),
                pltpu.SMEM((8, plan.num_experts), jnp.int32),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SMEM((plan.num_experts,), jnp.int32),
                pltpu.SMEM((local_experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((local_experts,)),
                pltpu.SemaphoreType.DMA((local_experts,)),
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=_collective_id(tag),
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=tag,
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P((dp_axis_name, tp_axis_name)),
            P((dp_axis_name, tp_axis_name)),
            P(),
            P(),
        ),
        out_specs=P(),
        check_vma=False,
    )
    def run(tokens, topk_ids, starts, sizes):
        return scatter_call(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(topk_ids, pltpu.HBM),
            pltpu.with_memory_space_constraint(starts, pltpu.HBM),
            pltpu.with_memory_space_constraint(sizes, pltpu.HBM),
        )

    return run


def build_gather_runner(
    *,
    mesh: jax.sharding.Mesh,
    plan: CasePlan,
    dp_axis_name: str = "data",
    tp_axis_name: str = "tensor",
) -> tuple[Callable[[jax.Array, jax.Array, jax.Array], jax.Array], tuple[int, ...]]:
    dtype = jnp.bfloat16
    ep_size = plan.ep_size
    local_experts = plan.local_experts
    num_bt = plan.num_bt
    bt = plan.bt
    packing = 32 // (jnp.dtype(dtype).itemsize * 8)
    h_per_pack = plan.hidden_size // packing
    use_banks = num_bt > 1
    local_source_shape = (
        (num_bt, local_experts, plan.a2a_max_tokens, packing, h_per_pack)
        if use_banks
        else (local_experts, plan.a2a_max_tokens, packing, h_per_pack)
    )
    local_out_shape = (
        (num_bt, plan.num_experts, bt, packing, h_per_pack)
        if use_banks
        else (plan.num_experts, bt, packing, h_per_pack)
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    tag = f"fused_moe_v2_a2a_gather_t{plan.tokens}_bfloat16"

    def pallas_kernel(
        source_ref,
        counts_ref,
        sizes_ref,
        out_ref,
        counts_smem,
        sizes_smem,
        metadata_sems,
        send_sems,
        recv_sem,
        barrier_sem,
    ):
        dp_rank = lax.axis_index(dp_axis_name)
        tp_rank = lax.axis_index(tp_axis_name)
        tp_size = lax.axis_size(tp_axis_name)
        my_id = dp_rank * tp_size + tp_rank

        def mesh_device_id(rank):
            return (rank // tp_size, rank % tp_size)

        def source_slice(bt_id, expert_slot, start, size):
            if use_banks:
                return source_ref.at[bt_id, expert_slot, pl.ds(start, size)]
            return source_ref.at[expert_slot, pl.ds(start, size)]

        def out_slice(bt_id, expert_id, start, size):
            if use_banks:
                return out_ref.at[bt_id, expert_id, pl.ds(start, size)]
            return out_ref.at[expert_id, pl.ds(start, size)]

        def sync_barrier():
            for peer in range(ep_size):
                pl.semaphore_signal(
                    barrier_sem,
                    device_id=mesh_device_id(peer),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier_sem, ep_size)

        def gather_bt(bt_id, _):
            counts_copy = pltpu.async_copy(
                src_ref=counts_ref.at[bt_id],
                dst_ref=counts_smem,
                sem=metadata_sems.at[0],
            )
            sizes_copy = pltpu.async_copy(
                src_ref=sizes_ref.at[bt_id],
                dst_ref=sizes_smem,
                sem=metadata_sems.at[1],
            )
            counts_copy.wait()
            sizes_copy.wait()

            with jax.named_scope("a2a_gather"):

                def gather_expert(expert_slot, _):
                    expert_id = my_id * local_experts + expert_slot
                    start = jnp.int32(0)
                    for recv_id in range(ep_size):
                        rows = counts_smem[recv_id, expert_id]
                        is_local = recv_id == my_id

                        with jax.named_scope("a2a_gather_local_copy"):

                            @pl.when(jnp.logical_and(is_local, rows != 0))
                            def local_copy(
                                _start=start,
                                _rows=rows,
                                _expert_id=expert_id,
                                _expert_slot=expert_slot,
                            ):
                                pltpu.make_async_copy(
                                    src_ref=source_slice(
                                        bt_id,
                                        _expert_slot,
                                        _start,
                                        _rows,
                                    ),
                                    dst_ref=out_slice(
                                        bt_id,
                                        _expert_id,
                                        0,
                                        _rows,
                                    ),
                                    sem=recv_sem,
                                ).start()

                        with jax.named_scope("a2a_gather_remote_copy"):

                            @pl.when(jnp.logical_and(jnp.logical_not(is_local), rows != 0))
                            def remote_copy(
                                _start=start,
                                _rows=rows,
                                _expert_id=expert_id,
                                _expert_slot=expert_slot,
                                _recv_id=recv_id,
                            ):
                                pltpu.make_async_remote_copy(
                                    src_ref=source_slice(
                                        bt_id,
                                        _expert_slot,
                                        _start,
                                        _rows,
                                    ),
                                    dst_ref=out_slice(
                                        bt_id,
                                        _expert_id,
                                        0,
                                        _rows,
                                    ),
                                    send_sem=send_sems.at[_expert_slot],
                                    recv_sem=recv_sem,
                                    device_id=mesh_device_id(_recv_id),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).start()

                        start += rows
                    return None

                lax.fori_loop(0, local_experts, gather_expert, None, unroll=False)

            with jax.named_scope("a2a_gather_send_wait"):

                def wait_one_send(expert_slot, _):
                    expert_id = my_id * local_experts + expert_slot
                    total_rows = sizes_smem[0, expert_id]
                    local_rows = counts_smem[my_id, expert_id]
                    remote_rows = total_rows - local_rows

                    @pl.when(remote_rows != 0)
                    def wait_send(
                        _expert_slot=expert_slot,
                        _remote_rows=remote_rows,
                    ):
                        ref = source_slice(bt_id, _expert_slot, 0, _remote_rows)
                        pltpu.make_async_copy(
                            src_ref=ref,
                            dst_ref=ref,
                            sem=send_sems.at[_expert_slot],
                        ).wait()

                    return None

                lax.fori_loop(0, local_experts, wait_one_send, None, unroll=False)

            with jax.named_scope("a2a_gather_recv_wait"):

                def wait_one_recv(expert_id, _):
                    recv_rows = counts_smem[my_id, expert_id]

                    @pl.when(recv_rows != 0)
                    def wait_recv(
                        _expert_id=expert_id,
                        _recv_rows=recv_rows,
                    ):
                        ref = out_slice(bt_id, _expert_id, 0, _recv_rows)
                        pltpu.make_async_copy(
                            src_ref=ref,
                            dst_ref=ref,
                            sem=recv_sem,
                        ).wait()

                    return None

                lax.fori_loop(0, plan.num_experts, wait_one_recv, None, unroll=False)
            sync_barrier()
            return None

        sync_barrier()
        lax.fori_loop(0, num_bt, gather_bt, None, unroll=False)

    gather_call = pl.pallas_call(
        pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(local_out_shape, dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm_spec, vmem_spec, vmem_spec],
            out_specs=hbm_spec,
            scratch_shapes=[
                pltpu.SMEM((ep_size, plan.num_experts), jnp.int32),
                pltpu.SMEM((8, plan.num_experts), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((local_experts,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=_collective_id(tag),
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=tag,
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def sharded_run(source, counts, sizes):
        return gather_call(
            pltpu.with_memory_space_constraint(source, pltpu.HBM),
            pltpu.with_memory_space_constraint(counts, pltpu.HBM),
            pltpu.with_memory_space_constraint(sizes, pltpu.HBM),
        )

    @jax.jit
    def run(source, counts, sizes):
        return sharded_run(source, counts, sizes)

    return run, local_source_shape


def time_runner(
    run_fn: Callable[[], jax.Array],
    *,
    warmup_runs: int,
    sample_runs: int,
) -> list[float]:
    for _ in range(warmup_runs):
        out = run_fn()
        jax.block_until_ready(out)
        del out
    samples = []
    for _ in range(sample_runs):
        start = time.perf_counter()
        out = run_fn()
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) * 1e3)
        del out
    return samples


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        ep_size = 32
        plans = [
            build_case_plan(
                tokens=tokens,
                ep_size=ep_size,
                num_experts=args.num_experts,
                top_k=args.top_k,
                hidden_size=args.hidden_size,
                base_bt=args.base_bt,
            )
            for tokens in args.tokens
        ]
        write_json(
            output_dir / "logical_communication_plan.json",
            {"cases": [logical_plan_dict(plan) for plan in plans]},
        )
        return

    load_jax()
    maybe_initialize_distributed()
    ep_size = jax.device_count()
    if ep_size != 32:
        raise ValueError(f"This compiler handoff run requires EP=32, got {ep_size=}")
    if ep_size % args.dp_size:
        raise ValueError(f"{ep_size=} must be divisible by dp_size={args.dp_size}")
    tp_size = ep_size // args.dp_size
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices(), dtype=object).reshape(args.dp_size, tp_size),
        ("data", "tensor"),
    )
    topology = topology_dict(mesh)
    write_json(output_dir / "topology.json", topology)

    full_axes = ("data", "tensor")
    token_sharding = jax.sharding.NamedSharding(mesh, P(full_axes, None, None))
    topk_sharding = jax.sharding.NamedSharding(mesh, P(full_axes, None))
    replicated = jax.sharding.NamedSharding(mesh, P())
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    report_cases = []

    for tokens in args.tokens:
        plan = build_case_plan(
            tokens=tokens,
            ep_size=ep_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            base_bt=args.base_bt,
        )
        routing_width = max(128, args.top_k)
        topk_global = np.full((tokens, routing_width), -1, dtype=np.int32)
        topk_global[:, : args.top_k] = plan.topk_ids.reshape(tokens, args.top_k)
        topk_array = _make_array_from_numpy(topk_global, topk_sharding)
        starts_array = _make_array_from_numpy(plan.starts, replicated)
        sizes_staging = np.zeros(
            (plan.num_bt, 8, plan.num_experts),
            dtype=np.int32,
        )
        sizes_staging[:, 0, :] = plan.sizes
        sizes_array = _make_array_from_numpy(sizes_staging, replicated)
        counts_array = _make_array_from_numpy(plan.counts, replicated)
        case_report = logical_plan_dict(plan)
        case_report["physical_load"] = {}

        for variant in args.variants:
            if variant.startswith("scatter"):
                dtype = jnp.float8_e4m3fn if variant == "scatter_fp8" else jnp.bfloat16
                packing = 32 // (jnp.dtype(dtype).itemsize * 8)
                tokens_array = _make_constant_array(
                    (tokens, packing, args.hidden_size // packing),
                    dtype,
                    token_sharding,
                )
                runner = build_scatter_runner(mesh=mesh, plan=plan, dtype=dtype)

                run_variant = functools.partial(
                    runner,
                    tokens_array,
                    topk_array,
                    starts_array,
                    sizes_array,
                )

                phase = "scatter"
                dtype_bytes = jnp.dtype(dtype).itemsize
            else:
                runner, source_shape = build_gather_runner(mesh=mesh, plan=plan)
                source = jax.jit(
                    lambda _shape=source_shape: jnp.ones(_shape, dtype=jnp.bfloat16),
                    out_shardings=replicated,
                )()

                run_variant = functools.partial(runner, source, counts_array, sizes_array)

                phase = "gather"
                dtype_bytes = jnp.dtype(jnp.bfloat16).itemsize

            samples = time_runner(
                run_variant,
                warmup_runs=args.warmup_runs,
                sample_runs=args.sample_runs,
            )
            load = physical_load_dict(
                plan=plan,
                topology=topology,
                phase=phase,
                dtype_bytes=dtype_bytes,
            )
            case_report["physical_load"][variant] = load
            metric = {
                "variant": variant,
                "tokens": tokens,
                "ep_size": ep_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
                "hidden_size": args.hidden_size,
                "bt": plan.bt,
                "num_bt": plan.num_bt,
                "dtype": "fp8" if dtype_bytes == 1 else "bf16",
                "latency_ms": statistics.median(samples),
                "latency_samples_ms": samples,
                "remote_payload_bytes": load["remote_payload_bytes"],
                "hop_weighted_payload_bytes": load["hop_weighted_payload_bytes"],
                "process_index": jax.process_index(),
            }
            append_jsonl(metrics_path, metric)
            if jax.process_index() == 0:
                print(json.dumps(metric, sort_keys=True), flush=True)

        report_cases.append(case_report)

    write_json(
        output_dir / "communication_plan.json",
        {
            "model": "zai-org/GLM-5.2",
            "model_dimensions": {
                "hidden_size": args.hidden_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
            },
            "topology": topology,
            "cases": report_cases,
        },
    )
    handoff_cases = []
    for case in report_cases:
        compact_loads = {}
        for variant, load in case["physical_load"].items():
            compact_loads[variant] = {
                key: value
                for key, value in load.items()
                if key
                not in {
                    "payload_bytes_matrix",
                    "per_source_payload_bytes",
                    "per_destination_payload_bytes",
                }
            }
        handoff_cases.append(
            {key: value for key, value in case.items() if key not in {"physical_load", "source"}}
            | {"physical_load": compact_loads}
        )
    handoff = {
        "model": "zai-org/GLM-5.2",
        "topology": topology,
        "cases": handoff_cases,
    }
    write_json(output_dir / "handoff_summary.json", handoff)
    if jax.process_index() == 0:
        print("A2A_HANDOFF_JSON=" + json.dumps(handoff, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
