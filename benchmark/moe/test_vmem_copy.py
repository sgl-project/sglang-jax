"""Test whether VMEM-to-VMEM async DMA copy is supported on TPU.

Tries three memory space combinations:
1. HBM → HBM (known to work, baseline)
2. HBM → VMEM (known to work)
3. VMEM → VMEM (unknown - this is what we want to test)

Usage:
    python -m benchmark.moe.test_vmem_copy
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel_hbm_to_hbm(src_hbm, dst_hbm, out_hbm, sem):
    """HBM → HBM async copy (baseline, known to work)."""
    pltpu.make_async_copy(
        src_ref=src_hbm.at[pl.ds(0, 4)],
        dst_ref=dst_hbm.at[pl.ds(0, 4)],
        sem=sem,
    ).start()
    pltpu.make_async_copy(
        src_ref=dst_hbm.at[pl.ds(0, 4)],
        dst_ref=dst_hbm.at[pl.ds(0, 4)],
        sem=sem,
    ).wait()


def _kernel_hbm_to_vmem(src_hbm, _dst_hbm, out_hbm, sem, vmem_buf):
    """HBM → VMEM async copy (known to work)."""
    pltpu.make_async_copy(
        src_ref=src_hbm.at[pl.ds(0, 4)],
        dst_ref=vmem_buf.at[pl.ds(0, 4)],
        sem=sem,
    ).start()
    pltpu.make_async_copy(
        src_ref=vmem_buf.at[pl.ds(0, 4)],
        dst_ref=vmem_buf.at[pl.ds(0, 4)],
        sem=sem,
    ).wait()


def _kernel_vmem_to_vmem(src_hbm, _dst_hbm, out_hbm, sem, vmem_a, vmem_b):
    """VMEM → VMEM async copy (the test case)."""
    # First load data into vmem_a from HBM
    pltpu.make_async_copy(
        src_ref=src_hbm.at[pl.ds(0, 4)],
        dst_ref=vmem_a.at[pl.ds(0, 4)],
        sem=sem,
    ).start()
    pltpu.make_async_copy(
        src_ref=vmem_a.at[pl.ds(0, 4)],
        dst_ref=vmem_a.at[pl.ds(0, 4)],
        sem=sem,
    ).wait()

    # Now try VMEM → VMEM async copy
    pltpu.make_async_copy(
        src_ref=vmem_a.at[pl.ds(0, 4)],
        dst_ref=vmem_b.at[pl.ds(0, 4)],
        sem=sem,
    ).start()
    pltpu.make_async_copy(
        src_ref=vmem_b.at[pl.ds(0, 4)],
        dst_ref=vmem_b.at[pl.ds(0, 4)],
        sem=sem,
    ).wait()


def run_test(name, kernel_fn, scratch_shapes):
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    dtype = jnp.bfloat16
    shape = (4, 2, 128)

    @jax.jit
    def run(src, dst):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(shape, dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec, hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=scratch_shapes,
            ),
        )(src, dst)

    src = jnp.ones(shape, dtype=dtype)
    dst = jnp.zeros(shape, dtype=dtype)

    try:
        out = run(src, dst)
        jax.block_until_ready(out)
        print(f"  {name}: SUCCESS")
        return True
    except Exception as e:
        err = str(e)
        # Truncate long errors
        if len(err) > 200:
            err = err[:200] + "..."
        print(f"  {name}: FAILED - {err}")
        return False


def main():
    print(f"TPU devices: {jax.device_count()}")
    print(f"Testing async copy between memory spaces:\n")

    vmem_shape = (4, 2, 128)
    dtype = jnp.bfloat16

    run_test(
        "HBM → HBM",
        _kernel_hbm_to_hbm,
        (pltpu.SemaphoreType.DMA,),
    )

    run_test(
        "HBM → VMEM",
        _kernel_hbm_to_vmem,
        (pltpu.SemaphoreType.DMA, pltpu.VMEM(vmem_shape, dtype)),
    )

    run_test(
        "VMEM → VMEM",
        _kernel_vmem_to_vmem,
        (pltpu.SemaphoreType.DMA, pltpu.VMEM(vmem_shape, dtype), pltpu.VMEM(vmem_shape, dtype)),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
