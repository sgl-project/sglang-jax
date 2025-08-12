# Usage:
# PROCESS_ID=0 NUM_PROCESSES=2 COORDINATOR_ADDRESS=localhost:12345 python -m sgl_jax.test.test_multi_process_radix_cache
# PROCESS_ID=1 NUM_PROCESSES=2 COORDINATOR_ADDRESS=localhost:12345 python -m sgl_jax.test.test_multi_process_radix_cache

import os
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import RadixCache


def print_cache_sharding_info(cache, mesh, req_pool, allocator, process_id):
    """Print cache-related sharding information"""
    print(f"\n{'='*80}")
    print(f"[PROCESS {process_id}] RadixCache multiprocesssharding info")
    print(f"[PROCESS {process_id}] Local device count: {len(jax.local_devices())}")
    print(f"[PROCESS {process_id}] Global device count: {len(jax.devices())}")
    print(f"[PROCESS {process_id}] Mesh axes: {mesh.axis_names}")
    print(
        f"[PROCESS {process_id}] Local mesh device layout: {mesh.local_mesh.devices.shape}"
    )
    print(f"[PROCESS {process_id}] Global mesh device layout: {mesh.devices.shape}")
    print(f"[PROCESS {process_id}] Local mesh: {mesh.local_mesh}")
    print(f"[PROCESS {process_id}] Global mesh: {mesh}")

    def print_sharding(obj, name, prefix=""):
        """Recursively print object's sharding information"""
        full_name = f"{prefix}.{name}" if prefix else name

        if hasattr(obj, "sharding") and obj.sharding is not None:
            print(
                f"[PROCESS {process_id}] [SHARDING] {full_name}: sharding={obj.sharding}"
            )
            if hasattr(obj, "addressable_shards"):
                for i, shard in enumerate(obj.addressable_shards):
                    print(
                        f"[PROCESS {process_id}]     [SHARD] idx={i}, device={shard.device}, index={getattr(shard, 'index', None)}, shape={getattr(shard.data, 'shape', None)}"
                    )
        elif hasattr(obj, "shape"):
            print(
                f"[PROCESS {process_id}] [SHARDING] {full_name}: Unsharded, shape={obj.shape}, device={getattr(obj, 'device', 'unknown')}"
            )
        else:
            print(
                f"[PROCESS {process_id}] [SHARDING] {full_name}: Non-JAX array, type={type(obj)}"
            )

    # Print RadixCache's sharding information
    if hasattr(cache, "kv_cache_sharding"):
        print(
            f"[PROCESS {process_id}] [CACHE] KV cache sharding strategy: {cache.kv_cache_sharding}"
        )
    if hasattr(cache, "token_sharding"):
        print(
            f"[PROCESS {process_id}] [CACHE] Token sharding strategy: {cache.token_sharding}"
        )

    # Print ReqToTokenPool's sharding information
    print_sharding(req_pool.req_to_token, "req_to_token_pool.req_to_token")

    # Print KV Cache's sharding information
    kv_cache = allocator.get_kvcache()
    if hasattr(kv_cache, "k_buffer") and kv_cache.k_buffer:
        print_sharding(kv_cache.k_buffer[0], "kv_cache.k_buffer[0]", "allocator")
    if hasattr(kv_cache, "v_buffer") and kv_cache.v_buffer:
        print_sharding(kv_cache.v_buffer[0], "kv_cache.v_buffer[0]", "allocator")
    if hasattr(kv_cache, "kv_buffer") and kv_cache.kv_buffer:
        print_sharding(kv_cache.kv_buffer[0], "kv_cache.kv_buffer[0]", "allocator")

    print(f"{'='*80}")


def create_multi_process_radix_cache(process_id, tp_size=8):
    """Create multi-process RadixCache configuration"""
    # Cache configuration
    kv_head_num = 32
    head_dim = 128
    layer_num = 24
    max_seq_len = 2048
    dtype = jnp.bfloat16

    # Adjust cache size: each device holds total size / tp_size
    base_pool_size = 8192
    base_req_pool_size = 1024

    # Actual size per device
    pool_size_per_device = base_pool_size // tp_size
    req_pool_size_per_device = base_req_pool_size // tp_size

    print(
        f"[PROCESS {process_id}] Base pool size: {base_pool_size}, pool size per device: {pool_size_per_device}"
    )
    print(
        f"[PROCESS {process_id}] Base request pool size: {base_req_pool_size}, request pool size per device: {req_pool_size_per_device}"
    )

    # Use local devices to create mesh, avoid cross-process sharding of token data
    # Each process independently manages its own RadixCache
    local_devices = jax.local_devices()
    print(f"[PROCESS {process_id}] Local devices: {local_devices}")

    if len(local_devices) >= 2:
        # If local devices are sufficient, create two-axis mesh
        local_tensor_size = len(local_devices)
        local_data_size = 1

        import numpy as np

        devices_reshaped = np.array(local_devices).reshape(
            local_data_size, local_tensor_size
        )
        mesh = Mesh(devices_reshaped, ("data", "tensor"))
    else:
        # If local devices are insufficient, use single axis
        mesh = Mesh(local_devices, ("tensor",))

    print(f"[PROCESS {process_id}] Created local mesh: {mesh}")

    # Create request-to-token pool, use data axis sharding (or no sharding)
    token_partition_axis = "data" if "data" in mesh.axis_names else None
    req_pool = ReqToTokenPool(
        size=req_pool_size_per_device,
        max_context_len=max_seq_len,
        mesh=mesh,
        token_partition_axis=token_partition_axis,
    )

    # Create KV cache, use tensor axis sharding
    kv_cache = MHATokenToKVPool(
        size=pool_size_per_device,
        page_size=1,
        dtype=dtype,
        head_num=kv_head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=mesh,
        kv_partition_axis="tensor",
    )

    # Create allocator
    allocator = TokenToKVPoolAllocator(
        size=pool_size_per_device, dtype=dtype, kvcache=kv_cache
    )

    # Create RadixCache, each process independent
    cache = RadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        kv_head_num=kv_head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )

    return cache, mesh, req_pool, allocator


def test_basic_radix_cache_operations(cache, process_id):
    """Test basic RadixCache operations"""
    print(f"\n[PROCESS {process_id}] Starting basic RadixCache operation tests...")

    # All processes use the same test data to ensure multi-process consistency
    test_keys = [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7], [10, 11, 12, 13, 14]]

    for i, key in enumerate(test_keys):
        print(f"[PROCESS {process_id}] Inserting key {i+1}: {key}")
        prefix_len = cache.insert(key)
        print(f"[PROCESS {process_id}] Prefix match length: {prefix_len}")

        # Test matching
        match_result = cache.match_prefix(key)
        print(
            f"[PROCESS {process_id}] Match result length: {len(match_result.device_indices)}"
        )

        # Test getting KV data
        kv_data, matched_len = cache.get_cached_kv(key)
        print(
            f"[PROCESS {process_id}] KV data shape: {kv_data.shape}, match length: {matched_len}"
        )

    # Test cache size
    print(f"[PROCESS {process_id}] Cache status:")
    print(f"[PROCESS {process_id}]   Total size: {cache.total_size()}")
    print(f"[PROCESS {process_id}]   Evictable size: {cache.evictable_size()}")
    print(f"[PROCESS {process_id}]   Protected size: {cache.protected_size()}")

    # Test cache tree printing
    print(f"[PROCESS {process_id}] Cache tree structure:")
    cache.pretty_print()


def test_memory_usage(cache, process_id, pool_size_per_device):
    """Test memory usage"""
    print(f"\n[PROCESS {process_id}] Testing memory usage...")

    # Calculate theoretical memory usage
    bytes_per_element = 2  # bfloat16
    kv_head_num = 32
    head_dim = 128
    layer_num = 24

    # Theoretical KV cache size per device
    theoretical_kv_size = (
        pool_size_per_device
        * kv_head_num
        * head_dim
        * 2
        * layer_num
        * bytes_per_element
    )  # 2 for K and V
    theoretical_kv_size_gb = theoretical_kv_size / (1024**3)

    print(
        f"[PROCESS {process_id}] Theoretical KV cache size per device: {theoretical_kv_size_gb:.4f} GB"
    )
    print(f"[PROCESS {process_id}] Pool size per device: {pool_size_per_device}")

    # Fill some data to test actual memory usage
    large_keys = []
    for i in range(10):
        key = list(range(i * 100, (i + 1) * 100))  # 100-token sequence
        large_keys.append(key)
        cache.insert(key)

    print(
        f"[PROCESS {process_id}] Cache size after inserting large amount of data: {cache.total_size()}"
    )


def test_cross_process_isolation(cache, process_id):
    """Test cross-process isolation - each process can insert different data"""
    print(f"\n[PROCESS {process_id}] Testing cross-process isolation...")

    # Now each process can safely insert different data, as cross-process sharding is no longer used
    process_specific_keys = [
        [process_id * 1000 + 100 + i for i in range(5)],  # Process-specific key 1
        [process_id * 1000 + 200 + i for i in range(5)],  # Process-specific key 2
        [process_id * 1000 + 300 + i for i in range(5)],  # Process-specific key 3
    ]

    print(f"[PROCESS {process_id}] Inserting process-specific data:")
    for i, key in enumerate(process_specific_keys):
        print(f"[PROCESS {process_id}] Inserting key{i+1}: {key}")
        cache.insert(key)
        match_result = cache.match_prefix(key)
        print(
            f"[PROCESS {process_id}] Key{i+1} match result: {len(match_result.device_indices)}"
        )

    # Test cache status
    print(f"[PROCESS {process_id}] Process-specific cache status:")
    print(f"[PROCESS {process_id}]   Total size: {cache.total_size()}")
    print(f"[PROCESS {process_id}]   Evictable size: {cache.evictable_size()}")

    # Print cache tree structure
    print(f"[PROCESS {process_id}] Process-specific cache tree:")
    cache.pretty_print()


def main():
    # Get multi-process configuration from environment variables
    process_id = int(os.environ.get("PROCESS_ID", "0"))
    num_processes = int(os.environ.get("NUM_PROCESSES", "2"))
    coordinator_address = os.environ.get("COORDINATOR_ADDRESS", "localhost:12345")
    tp_size = int(os.environ.get("TP_SIZE", "8"))

    print(
        f"[PROCESS {process_id}] Starting multi-process environment initialization..."
    )
    print(
        f"[PROCESS {process_id}] Process ID: {process_id}, total processes: {num_processes}"
    )
    print(f"[PROCESS {process_id}] Coordinator address: {coordinator_address}")
    print(f"[PROCESS {process_id}] TP size: {tp_size}")

    # Initialize distributed JAX
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )

    print(f"[PROCESS {process_id}] JAX distributed initialization completed")
    print(f"[PROCESS {process_id}] Local devices: {jax.local_devices()}")
    print(f"[PROCESS {process_id}] Global devices: {jax.devices()}")
    print(
        f"[PROCESS {process_id}] Device count - local: {jax.local_device_count()}, global: {jax.device_count()}"
    )

    # Wait for all processes to be ready
    print(f"[PROCESS {process_id}] Waiting for all processes to synchronize...")
    time.sleep(2)  # Simple synchronization method

    try:
        # Create multi-process RadixCache
        cache, mesh, req_pool, allocator = create_multi_process_radix_cache(
            process_id, tp_size
        )

        # Print sharding information
        print_cache_sharding_info(cache, mesh, req_pool, allocator, process_id)

        # Test basic operations
        test_basic_radix_cache_operations(cache, process_id)

        # Test memory usage
        pool_size_per_device = 8192 // tp_size
        test_memory_usage(cache, process_id, pool_size_per_device)

        # Test cross-process isolation
        test_cross_process_isolation(cache, process_id)

        print(f"[PROCESS {process_id}] All tests completed!")

    except Exception as e:
        print(f"[PROCESS {process_id}] Error during testing: {e}")
        raise

    finally:
        print(f"[PROCESS {process_id}] Testing finished")


if __name__ == "__main__":
    main()
