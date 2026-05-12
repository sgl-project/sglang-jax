"""Test whether VMEM-to-VMEM remote async copy (over ICI) is supported on TPU.

Tests:
1. HBM → remote HBM (known to work)
2. VMEM → remote HBM (unknown)
3. HBM → remote VMEM (unknown)
4. VMEM → remote VMEM (unknown)

Usage:
    python -m benchmark.moe.test_vmem_remote_copy
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental import shard_map

P = jax.sharding.PartitionSpec


def _make_kernel(src_space, dst_space, num_tokens):
    """Generate a kernel that does remote async copy between specified memory spaces."""

    if src_space == "vmem" and dst_space == "vmem":
        def kernel(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_src, vmem_dst):
            tp_rank = lax.axis_index("tensor")
            tp_size = lax.axis_size("tensor")
            target = (tp_rank + 1) % tp_size

            # Load src from HBM to VMEM first
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

            # VMEM → remote VMEM
            pltpu.make_async_remote_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(0, target),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            pltpu.make_async_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=send_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

        return kernel, 2  # 2 vmem scratch buffers

    elif src_space == "vmem" and dst_space == "hbm":
        def kernel(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_src):
            tp_rank = lax.axis_index("tensor")
            tp_size = lax.axis_size("tensor")
            target = (tp_rank + 1) % tp_size

            # Load src from HBM to VMEM first
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).start()
            pltpu.make_async_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

            # VMEM → remote HBM
            pltpu.make_async_remote_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(0, target),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            pltpu.make_async_copy(
                src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_src.at[pl.ds(0, num_tokens)],
                sem=send_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

        return kernel, 1

    elif src_space == "hbm" and dst_space == "vmem":
        def kernel(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_dst):
            tp_rank = lax.axis_index("tensor")
            tp_size = lax.axis_size("tensor")
            target = (tp_rank + 1) % tp_size

            # HBM → remote VMEM
            pltpu.make_async_remote_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(0, target),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=src_hbm.at[pl.ds(0, num_tokens)],
                sem=send_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

        return kernel, 1  # 1 vmem scratch (dst)

    else:  # hbm → hbm
        def kernel(src_hbm, dst_hbm, _out, send_sem, recv_sem):
            tp_rank = lax.axis_index("tensor")
            tp_size = lax.axis_size("tensor")
            target = (tp_rank + 1) % tp_size

            pltpu.make_async_remote_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(0, target),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=src_hbm.at[pl.ds(0, num_tokens)],
                sem=send_sem,
            ).wait()
            pltpu.make_async_copy(
                src_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                sem=recv_sem,
            ).wait()

        return kernel, 0


def run_test(name, src_space, dst_space, num_tokens, mesh):
    hidden_size = 6144
    t_packing = 2
    hidden_per_pack = hidden_size // t_packing
    dtype = jnp.bfloat16
    buf_shape = (num_tokens, t_packing, hidden_per_pack)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    kernel_fn, num_vmem = _make_kernel(src_space, dst_space, num_tokens)

    scratch = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
    for _ in range(num_vmem):
        scratch.append(pltpu.VMEM(buf_shape, dtype))

    def _pallas_body(src, dst):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(buf_shape, dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec, hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=tuple(scratch),
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        )(src, dst)

    @jax.jit
    def run(src, dst):
        return shard_map.shard_map(
            _pallas_body,
            mesh=mesh,
            in_specs=(P("tensor"), P("tensor")),
            out_specs=P("tensor"),
            check_rep=False,
        )(src, dst)

    ep_size = mesh.shape["tensor"]
    full_shape = (num_tokens * ep_size, t_packing, hidden_per_pack)
    src = jnp.ones(full_shape, dtype=dtype)
    dst = jnp.zeros(full_shape, dtype=dtype)
    sharding = jax.sharding.NamedSharding(mesh, P("tensor"))
    src = jax.device_put(src, sharding)
    dst = jax.device_put(dst, sharding)

    try:
        out = run(src, dst)
        jax.block_until_ready(out)
        print(f"  {name}: SUCCESS")
        return True
    except Exception as e:
        err = str(e)
        if len(err) > 300:
            err = err[:300] + "..."
        print(f"  {name}: FAILED - {err}")
        return False


def main():
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    ep_size = jax.device_count()
    print(f"TPU devices: {ep_size}")

    mesh = create_device_mesh(
        ici_parallelism=[1, ep_size],
        dcn_parallelism=[1, 1],
        devices=jax.devices()[:ep_size],
        mesh_axes=("data", "tensor"),
    )

    num_tokens = 4
    print(f"\nTesting remote async copy between memory spaces (num_tokens={num_tokens}):\n")

    cases = [
        ("HBM → remote HBM", "hbm", "hbm"),
        ("VMEM → remote HBM", "vmem", "hbm"),
        ("HBM → remote VMEM", "hbm", "vmem"),
        ("VMEM → remote VMEM", "vmem", "vmem"),
    ]

    for name, src, dst in cases:
        run_test(name, src, dst, num_tokens, mesh)

    print("\nDone.")


if __name__ == "__main__":
    main()
