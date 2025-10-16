import gc
import sys
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import LinearBase

GIGANTIC_SIZE = 16384  # ~512MB tensor


# --- Eager Initialization Class (for comparison) ---
class EagerLinear(nnx.Module):
    def __init__(self, input_size, output_size, *, rngs):
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), ("data", "model"))(
                rngs.params(), (input_size, output_size), jnp.bfloat16
            )
        )


def print_memory_usage(event: str):
    """Helper to print memory stats for the first device."""
    stats = jax.devices()[0].memory_stats()
    # Peak bytes in use is a good indicator of memory allocation
    peak_bytes = stats.get("peak_bytes_in_use", 0)
    print(f"[{event}] Peak memory in use: {peak_bytes / 1e6:.2f} MB")
    return peak_bytes


def print_all_devices_memory(event: str):
    """Print current and peak memory for all devices."""
    for i, d in enumerate(jax.devices()):
        stats = d.memory_stats()
        cur = stats.get("bytes_in_use", 0)
        peak = stats.get("peak_bytes_in_use", 0)
        print(f"[{event}] dev{i} current={cur / 1e6:.2f} MB, peak={peak / 1e6:.2f} MB")


def device_memory_supported() -> bool:
    try:
        stats = jax.devices()[0].memory_stats()
        _ = stats.get("bytes_in_use", None)
        _ = stats.get("peak_bytes_in_use", None)
        return True
    except Exception:
        return False


def get_current_bytes_for_devices(devices):
    return [d.memory_stats().get("bytes_in_use", 0) for d in devices]


class TestMemoryUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if len(jax.devices()) < 4 or not device_memory_supported():
            raise unittest.SkipTest("Requires >=4 devices and memory_stats support")

    def setUp(self):
        all_devs = jax.devices()
        self.used_devices = all_devs[:4]
        self.mesh = Mesh(
            np.array(self.used_devices).reshape((2, 2)), axis_names=("data", "model")
        )

    def test_lazy_construction_minimal_memory(self):
        before = get_current_bytes_for_devices(self.used_devices)

        rngs = nnx.Rngs(0)
        input_size, output_size = 4096, 4096
        model = LinearBase(
            input_size=input_size,
            output_size=output_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=("data", "model"),
            rngs=rngs,
        )

        after = get_current_bytes_for_devices(self.used_devices)

        mb = 1024 * 1024
        for i, (b, a) in enumerate(zip(before, after)):
            self.assertLess(
                a - b, 2 * mb, f"Device {i} allocated too much during lazy construction"
            )

        del model
        gc.collect()
        jnp.zeros(()).block_until_ready()

    def test_materialize_adds_shard_memory(self):
        before = get_current_bytes_for_devices(self.used_devices)

        input_size, output_size = 4096, 4096
        bytes_per_elem = 2  # bf16
        total_param_bytes = input_size * output_size * bytes_per_elem
        num_devices = 4
        expected_per_device = total_param_bytes // num_devices

        model = LinearBase(
            input_size=input_size,
            output_size=output_size,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=("data", "model"),
            rngs=nnx.Rngs(1),
        )

        model.materialize(self.mesh, nnx.Rngs(2))
        model.weight.value.block_until_ready()

        after = get_current_bytes_for_devices(self.used_devices)

        lower = 0.5 * expected_per_device
        upper = 2.0 * expected_per_device
        for i, (b, a) in enumerate(zip(before, after)):
            delta = max(0, a - b)
            self.assertGreaterEqual(
                delta, lower, f"Device {i} shard too small: {delta} < {lower}"
            )
            self.assertLessEqual(
                delta, upper, f"Device {i} shard too large: {delta} > {upper}"
            )

        del model
        gc.collect()
        jnp.zeros(()).block_until_ready()


# --- Demo helpers ---
def run_eager_demo(mesh: Mesh):
    rngs = nnx.Rngs(0)
    print("--- Eager Init + Shard via nnx.jit ---")
    print_all_devices_memory("Before Eager create_model (current)")
    mem_before = print_memory_usage("Before Eager create_model (peak)")

    @nnx.jit
    def create_model(rng: nnx.Rngs):
        model = EagerLinear(GIGANTIC_SIZE, GIGANTIC_SIZE, rngs=rng)
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        # Apply sharding according to partition specs under mesh context
        with mesh:
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

    start = time.perf_counter()
    eager_model = create_model(rngs)
    eager_model.weight.value.block_until_ready()
    secs = time.perf_counter() - start
    print(f"create_model (jit) time: {secs * 1000:.2f} ms")
    print_all_devices_memory("After Eager create_model (current)")
    mem_after = print_memory_usage("After Eager create_model (peak)")
    print(f"Memory increased by: {(mem_after - mem_before) / 1e6:.2f} MB")
    print(f"Sharded Status: {eager_model.weight.value.sharding}\n")


def run_lazy_demo(mesh: Mesh):
    rngs = nnx.Rngs(1)
    print("--- Testing Lazy Initialization (jitted create + materialize) ---")
    print_all_devices_memory("Before Lazy (current)")
    mem_before_lazy = print_memory_usage("Before Lazy (peak)")

    @nnx.jit
    def create_lazy_model(r: nnx.Rngs):
        model = LinearBase(
            input_size=GIGANTIC_SIZE,
            output_size=GIGANTIC_SIZE,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=("data", "model"),
            rngs=r,
        )
        # Materialize within jit under the mesh context so weights are created and sharded
        with mesh:
            model.materialize(mesh, r)
        return model

    start_create = time.perf_counter()
    lazy_model = create_lazy_model(rngs)
    lazy_model.weight.value.block_until_ready()
    create_seconds = time.perf_counter() - start_create
    print(f"create_lazy_model (jit) time: {create_seconds * 1000:.2f} ms")
    print_all_devices_memory("After create_lazy_model (current)")
    _ = print_memory_usage("After create_lazy_model (peak)")
    print(f"Final Sharding Status: {lazy_model.weight.value.sharding}")


if __name__ == "__main__":
    # Run unittests by default. To run interactive demos, use: --eager or --lazy
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--eager", action="store_true", help="Run eager init demo")
    group.add_argument("--lazy", action="store_true", help="Run lazy init demo")
    args = parser.parse_args()

    if args.eager or args.lazy:
        devices = jax.devices()
        if len(devices) < 4:
            raise ValueError("This example requires at least 4 devices for a 2x2 mesh.")
        device_mesh = np.array(devices[:4]).reshape((2, 2))
        mesh = Mesh(device_mesh, axis_names=("data", "model"))

        if args.eager:
            run_eager_demo(mesh)
        else:
            run_lazy_demo(mesh)
    else:
        unittest.main(argv=[sys.argv[0]])
