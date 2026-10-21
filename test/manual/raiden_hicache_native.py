"""Manual native TPU HiCache round-trip and per-rank pressure regression.

Requires a matching tpu-sync wheel; run in a fresh process before importing JAX.
"""

# Native extension must load before all imports that initialize JAX.
# ruff: noqa: E402

import argparse
import json
from functools import partial

from sgl_jax.raiden import preload_raiden

preload_raiden()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.raiden_hicache import create_raiden_hicache
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

args = argparse.ArgumentParser()
args.add_argument("--dp-size", type=int, default=1)
args.add_argument("--rounds", type=int, default=8)
options = args.parse_args()
dp = options.dp_size
mesh = Mesh(np.array(jax.devices()).reshape(dp, -1), ("data", "tensor"))
jax.sharding.set_mesh(mesh)
size, ps = 15 * 128 * dp, 128
pool = MHATokenToKVPool(size, ps, jnp.bfloat16, 8, 128, 2, mesh, dp_size=dp)
allocator = PagedTokenToKVPoolAllocator(size=size, page_size=ps, kvcache=pool, dp_size=dp)
cache = UnifiedRadixCache(
    req_to_token_pool=ReqToTokenPool(8, 2048, np.int32),
    token_to_kv_pool_allocator=allocator,
    page_size=ps,
    kv_head_num=8,
    head_dim=128,
    layer_num=2,
    max_seq_len=2048,
    dtype=jnp.bfloat16,
)
# Initialize each global page with distinct data before native registration.
for i, buffer in enumerate(pool.kv_buffer):
    shape = buffer.shape
    values = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) // 128 + 11 * i) % 127
    pool.kv_buffer[i] = jax.device_put(values.astype(jnp.bfloat16), buffer.sharding)
cache.host_pool, cache.hicache_controller = create_raiden_hicache(pool, 8 * dp, dp)
cache.hicache_enabled = True
cache.write_policy = "write_through"


@partial(jax.jit, donate_argnums=(0,))
def wipe(buffer):
    return buffer * jnp.bfloat16(0)


@partial(jax.jit, donate_argnums=(0,))
def refill(buffer):
    return buffer + jnp.bfloat16(1)


# A device-produced copy avoids np.asarray's cached snapshot after raw H2D.
fresh = jax.jit(lambda buffer: buffer[::-1])


def read():
    return [np.asarray(fresh(buffer))[::-1].copy() for buffer in pool.kv_buffer]


def pointers():
    return [[s.data.unsafe_buffer_pointer() for s in b.addressable_shards] for b in pool.kv_buffer]


registered = pointers()
pages_per_rank = pool.kv_buffer[0].shape[0] // dp
for iteration in range(options.rounds):
    expected = read()
    nodes = []
    for rank in range(dp):
        key = RadixKey(list(range(2 * ps)), extra_key=None, dp_rank=rank)
        indices = allocator.alloc(2 * ps, dp_rank=rank)
        cache.insert(InsertParams(key=key, value=indices))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        assert cache.write_backup(node) == 2
        assert node.backuped and node.component_data[0].lock_ref == 0
        source = np.asarray(indices)[::ps] // ps + rank * pages_per_rank
        nodes.append((rank, key, node, source))
        cache.evict(EvictParams(num_tokens=2 * ps, dp_rank=rank))
        assert node.evicted
    pool.kv_buffer = [wipe(b) for b in pool.kv_buffer]
    jax.block_until_ready(pool.kv_buffer)
    assert pointers() == registered, "donation replaced registered allocation"
    for rank, key, node, source in nodes:
        indices, restored, flush = cache.init_load_back(node, 2 * ps)
        assert restored is node and not flush and not node.evicted
        destination = np.asarray(indices)[::ps] // ps + rank * pages_per_rank
        actual = read()
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a[destination], e[source])
    cache.reset()
    allocator.clear()
    assert cache.host_pool.available_size() == 8 * dp
    pool.kv_buffer = [refill(b) for b in pool.kv_buffer]
    jax.block_until_ready(pool.kv_buffer)
    assert pointers() == registered
    print(
        json.dumps({"iteration": iteration, "dp_size": dp, "status": "passed"}),
        flush=True,
    )
# Fill each rank's dedicated host capacity with two evicted leaves. Pressure
# on one rank must not free another rank's host chunks, even if older in LRU.
for rank in range(dp):
    for segment in range(2):
        key = RadixKey(list(range(100 + segment * 512, 100 + segment * 512 + 4 * ps)), dp_rank=rank)
        indices = allocator.alloc(4 * ps, dp_rank=rank)
        cache.insert(InsertParams(key=key, value=indices))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        assert cache.write_backup(node) == 4
        cache.evict(EvictParams(num_tokens=4 * ps, dp_rank=rank))
    assert cache.host_pool.available_size(rank) == 0
protected = {h for h, page in cache.host_pool._pages.items() if page.rank != 0}
key = RadixKey(list(range(1200, 1200 + 4 * ps)), dp_rank=0)
indices = allocator.alloc(4 * ps, dp_rank=0)
expected = read()
source = np.asarray(indices)[::ps] // ps
cache.insert(InsertParams(key=key, value=indices))
node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
assert cache.write_backup(node) == 4
assert protected <= cache.host_pool._pages.keys()
assert cache.host_pool.available_size(0) == 0
cache.evict(EvictParams(num_tokens=4 * ps, dp_rank=0))
pool.kv_buffer = [wipe(b) for b in pool.kv_buffer]
jax.block_until_ready(pool.kv_buffer)
indices, restored, flush = cache.init_load_back(node, 4 * ps)
assert restored is node and not flush
actual = read()
destination = np.asarray(indices)[::ps] // ps
for a, e in zip(actual, expected):
    np.testing.assert_array_equal(a[destination], e[source])
cache.reset()
allocator.clear()
assert cache.host_pool.available_size() == 8 * dp
assert pointers() == registered
print("NATIVE_HOST_PRESSURE_PASS", json.dumps({"dp_size": dp}), flush=True)
cache.hicache_controller.shutdown()
print(
    "NATIVE_TREE_PASS",
    json.dumps({"dp_size": dp, "rounds": options.rounds, "devices": len(jax.devices())}),
    flush=True,
)
