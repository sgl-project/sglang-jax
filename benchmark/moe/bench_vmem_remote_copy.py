"""Benchmark remote async copy latency across memory space combinations.

Tests 4 paths over ICI with varying token counts:
1. HBM → remote HBM (baseline)
2. VMEM → remote HBM
3. HBM → remote VMEM
4. VMEM → remote VMEM

Each runs in a separate subprocess to avoid TPU semaphore conflicts.

Usage:
    python -m benchmark.moe.bench_vmem_remote_copy [--ep-size 8]
"""
import subprocess
import sys
import os
import numpy as np


def run_single(num_tokens, hidden_size, num_repeats, src_space, dst_space, ep_size):
    script = f"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "tpu")
if os.path.exists("/tmp/libtpu_lockfile"):
    os.remove("/tmp/libtpu_lockfile")
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
P = jax.sharding.PartitionSpec

num_tokens = {num_tokens}
hidden_size = {hidden_size}
num_repeats = {num_repeats}
src_space = "{src_space}"
dst_space = "{dst_space}"
t_packing = 2
hidden_per_pack = hidden_size // t_packing
dtype = jnp.bfloat16
buf_shape = (num_tokens, t_packing, hidden_per_pack)
hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

def _kernel_hbm_hbm(src_hbm, dst_hbm, _out, send_sem, recv_sem):
    tp_rank = lax.axis_index("tensor")
    tp_size = lax.axis_size("tensor")
    target = (tp_rank + 1) % tp_size
    def _repeat(_, __):
        pltpu.make_async_remote_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            send_sem=send_sem, recv_sem=recv_sem,
            device_id=(0, target), device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                              dst_ref=src_hbm.at[pl.ds(0, num_tokens)], sem=send_sem).wait()
        pltpu.make_async_copy(src_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                              dst_ref=dst_hbm.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)

def _kernel_vmem_hbm(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_src):
    tp_rank = lax.axis_index("tensor")
    tp_size = lax.axis_size("tensor")
    target = (tp_rank + 1) % tp_size
    # preload to vmem
    pltpu.make_async_copy(src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                          dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=recv_sem).start()
    pltpu.make_async_copy(src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                          dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
    def _repeat(_, __):
        pltpu.make_async_remote_copy(
            src_ref=vmem_src.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            send_sem=send_sem, recv_sem=recv_sem,
            device_id=(0, target), device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                              dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=send_sem).wait()
        pltpu.make_async_copy(src_ref=dst_hbm.at[pl.ds(0, num_tokens)],
                              dst_ref=dst_hbm.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)

def _kernel_hbm_vmem(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_dst):
    tp_rank = lax.axis_index("tensor")
    tp_size = lax.axis_size("tensor")
    target = (tp_rank + 1) % tp_size
    def _repeat(_, __):
        pltpu.make_async_remote_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
            send_sem=send_sem, recv_sem=recv_sem,
            device_id=(0, target), device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                              dst_ref=src_hbm.at[pl.ds(0, num_tokens)], sem=send_sem).wait()
        pltpu.make_async_copy(src_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                              dst_ref=vmem_dst.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)

def _kernel_vmem_vmem(src_hbm, dst_hbm, _out, send_sem, recv_sem, vmem_src, vmem_dst):
    tp_rank = lax.axis_index("tensor")
    tp_size = lax.axis_size("tensor")
    target = (tp_rank + 1) % tp_size
    # preload to vmem
    pltpu.make_async_copy(src_ref=src_hbm.at[pl.ds(0, num_tokens)],
                          dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=recv_sem).start()
    pltpu.make_async_copy(src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                          dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
    def _repeat(_, __):
        pltpu.make_async_remote_copy(
            src_ref=vmem_src.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_dst.at[pl.ds(0, num_tokens)],
            send_sem=send_sem, recv_sem=recv_sem,
            device_id=(0, target), device_id_type=pltpu.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(src_ref=vmem_src.at[pl.ds(0, num_tokens)],
                              dst_ref=vmem_src.at[pl.ds(0, num_tokens)], sem=send_sem).wait()
        pltpu.make_async_copy(src_ref=vmem_dst.at[pl.ds(0, num_tokens)],
                              dst_ref=vmem_dst.at[pl.ds(0, num_tokens)], sem=recv_sem).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)

if src_space == "hbm" and dst_space == "hbm":
    kernel_fn = _kernel_hbm_hbm
    scratch = (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA)
elif src_space == "vmem" and dst_space == "hbm":
    kernel_fn = _kernel_vmem_hbm
    scratch = (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.VMEM(buf_shape, dtype))
elif src_space == "hbm" and dst_space == "vmem":
    kernel_fn = _kernel_hbm_vmem
    scratch = (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA, pltpu.VMEM(buf_shape, dtype))
else:
    kernel_fn = _kernel_vmem_vmem
    scratch = (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
               pltpu.VMEM(buf_shape, dtype), pltpu.VMEM(buf_shape, dtype))

mesh = create_device_mesh(
    ici_parallelism=[1, {ep_size}], dcn_parallelism=[1, 1],
    devices=jax.devices()[:{ep_size}], mesh_axes=("data", "tensor"),
)

def _pallas_body(src, dst):
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(buf_shape, dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(src, dst)

@jax.jit
def run(src, dst):
    return jax.shard_map.shard_map(
        _pallas_body, mesh=mesh,
        in_specs=(P("tensor"), P("tensor")),
        out_specs=P("tensor"), check_rep=False,
    )(src, dst)

full_shape = (num_tokens * {ep_size}, t_packing, hidden_per_pack)
sharding = jax.sharding.NamedSharding(mesh, P("tensor"))
src = jax.device_put(jnp.ones(full_shape, dtype=dtype), sharding)
dst = jax.device_put(jnp.zeros(full_shape, dtype=dtype), sharding)

for _ in range(3):
    out = run(src, dst)
    jax.block_until_ready(out)

times_us = []
for _ in range(5):
    start = time.perf_counter()
    out = run(src, dst)
    jax.block_until_ready(out)
    elapsed_us = (time.perf_counter() - start) * 1e6
    times_us.append(elapsed_us / num_repeats)

if len(times_us) > 1:
    times_us = times_us[1:]
med = float(np.median(times_us))
print(f"RESULT={{med:.4f}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().split("\n")[-3:]
        print(f"  ERROR [{src_space}→{dst_space}]: {'; '.join(stderr_tail)}", file=sys.stderr)
        return float("nan")

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT="):
            return float(line.split("=")[1])
    return float("nan")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    args = parser.parse_args()

    modes = [("hbm", "hbm"), ("vmem", "hbm"), ("hbm", "vmem"), ("vmem", "vmem")]
    headers = ["HBM→rHBM", "VMEM→rHBM", "HBM→rVMEM", "VMEM→rVMEM"]

    print(f"Remote DMA benchmark (EP{args.ep_size}, hidden={args.hidden_size}, repeats={args.num_repeats})")
    print(f"{'tokens':>8}", end="")
    for h in headers:
        print(f"  {h + '(µs)':>14}", end="")
    print()
    print("-" * 72)

    for num_tokens in args.sizes:
        row = f"{num_tokens:>8}"
        for src_space, dst_space in modes:
            val = run_single(
                num_tokens, args.hidden_size, args.num_repeats,
                src_space, dst_space, args.ep_size
            )
            if np.isnan(val):
                row += f"  {'ERROR':>14}"
            else:
                row += f"  {val:>14.2f}"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
