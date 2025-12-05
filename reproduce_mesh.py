import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def test_mesh_behavior():
    # Attempt to use 4 devices for the test. Adjust if fewer are available.
    try:
        devices = np.array(jax.devices()[:4])
    except RuntimeError:
        print(
            "JAX devices not available. Make sure JAX is configured for your backend (e.g., CPU, GPU, TPU)."
        )
        return

    if len(devices) < 4:
        print(
            f"Not enough devices to test (need 4, found {len(devices)}). Skipping some parts of the test."
        )
        # Proceed with fewer devices if possible, or exit if not.
        if len(devices) < 2:  # Need at least 2 for a minimal 1x2 or 2x1 mesh
            print("Need at least 2 devices for basic mesh tests. Exiting.")
            return
        devices = devices[:2]  # Use 2 devices for a minimal test

    # Define ep_size and tp_size based on available devices
    ep_size = min(2, len(devices))
    tp_size = len(devices) // ep_size
    if tp_size == 0:
        tp_size = 1  # Ensure tp_size is at least 1

    # Original mesh: data=ep_size, tensor=tp_size (or adjusted if not 4 devices)
    # This simulates a typical initial mesh setup.
    print(f"Using {len(devices)} devices. Reshaping to ({ep_size}, {tp_size}) for initial mesh.")
    original_mesh_shape = (ep_size, tp_size)
    if len(devices) == 1:
        original_mesh = Mesh(devices.reshape(1), axis_names=("data",))  # Single device
    else:
        original_mesh = Mesh(devices.reshape(original_mesh_shape), axis_names=("data", "tensor"))

    print(f"\nOriginal Mesh: {original_mesh}")
    print(f"Original Mesh axis names: {original_mesh.axis_names}")

    # Moe mesh: expert=ep_size, tensor=tp_size (same underlying devices)
    devices_flat = original_mesh.devices.flatten()

    # Ensure reshape dimensions match the flattened devices length
    if ep_size * tp_size != len(devices_flat):
        # Adjust ep_size/tp_size for reshape if needed due to device count
        ep_size = len(devices_flat)
        tp_size = 1  # Fallback to a 1D reshape if device count is not divisible
        print(f"Adjusted MoE mesh shape to ({ep_size}, {tp_size}) for {len(devices_flat)} devices.")

    moe_mesh = Mesh(
        devices_flat.reshape(ep_size, tp_size),
        axis_names=("expert", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )

    print(f"\nMoE Mesh: {moe_mesh}")
    print(f"MoE Mesh axis names: {moe_mesh.axis_names}")

    # Check if "data" is in moe_mesh.shape
    print(f"Does 'moe_mesh' have 'data' axis? {'data' in moe_mesh.shape}")

    # Test kernel attempting to access "data" axis with moe_mesh
    @jax.jit
    @jax.shard_map(
        mesh=moe_mesh,
        in_specs=P("expert", "tensor"),
        out_specs=P("expert", "tensor"),
        # check_vma=False # This flag might be for older JAX versions or specific use cases. Removing for broader compatibility.
    )
    def kernel_access_data(x):
        try:
            # This simulates jax.lax.axis_index("data") within _fused_ep_moe_kernel
            dp_rank = jax.lax.axis_index("data")
            return x * 0 + dp_rank
        except NameError:
            # NameError will occur if axis "data" is not found
            return x * 0 - 100.0  # Return a distinct value to indicate failure
        except Exception as e:
            print(f"Unexpected error in kernel_access_data: {e}")
            return x * 0 - 200.0

    # Input array to shard
    x = jnp.ones((ep_size, tp_size))

    print(
        f"\nRunning kernel_access_data with input shape {x.shape} sharded as P('expert', 'tensor')..."
    )
    try:
        result_data_access = kernel_access_data(x)
        print(
            f"Result from kernel_access_data (expected to be -100.0 if 'data' axis is not found):\n{result_data_access}"
        )
    except Exception as e:
        print(f"kernel_access_data failed at JIT compilation/execution: {e}")

    # Test kernel accessing "expert" axis with moe_mesh (expected to succeed)
    @jax.jit
    @jax.shard_map(
        mesh=moe_mesh,
        in_specs=P("expert", "tensor"),
        out_specs=P("expert", "tensor"),
        # check_vma=False
    )
    def kernel_access_expert(x):
        ep_rank = jax.lax.axis_index("expert")
        return x * 0 + ep_rank

    print(
        f"\nRunning kernel_access_expert with input shape {x.shape} sharded as P('expert', 'tensor')..."
    )
    try:
        result_expert_access = kernel_access_expert(x)
        print(
            f"Result from kernel_access_expert (expected to show expert indices):\n{result_expert_access}"
        )
    except Exception as e:
        print(f"kernel_access_expert failed at JIT compilation/execution: {e}")


if __name__ == "__main__":
    # Configure JAX to use CPU for consistency if TPUs/GPUs aren't available
    # Or remove this line if you want to test on available accelerators.
    # jax.config.update('jax_platform_name', 'cpu')

    test_mesh_behavior()
