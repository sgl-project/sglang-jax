"""Multi-device correctness test for fused_ep_moe_v2.

Usage:
  # Single pod (ep=8, bench-4):
  python test_multi.py                    # default: small config
  python test_multi.py mimo-v2-pro        # MiMo-V2-Pro config (needs ep=32 for full test)

  # Multi-pod (ep=32, ablation-16):
  for pod in ablation-16-0-SUFFIX ablation-16-1-SUFFIX ...; do
    kubectl exec $pod -c bench -- python /tmp/v2_test/v2/test_multi.py &
  done
  wait
"""
from __future__ import annotations

import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.distributed.initialize()

num_devices = jax.device_count()
local_devices = jax.local_device_count()
process_id = jax.process_index()
num_processes = jax.process_count()

if process_id == 0:
    print(
        f"[process {process_id}/{num_processes}] "
        f"devices: {num_devices} (local: {local_devices})"
    )

from kernel import fused_ep_moe_v2, ref_moe_simple, align_to

P = jax.sharding.PartitionSpec

devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices

ALL_CONFIGS = {
    "small": {"name": "small", "d": 768, "f": 256, "E": 64, "top_k": 2, "bt": 16, "bf": 256},
    "mimo-v2-pro": {"name": "MiMo-V2-Pro", "d": 6144, "f": 2048, "E": 128, "top_k": 8, "bt": 16, "bf": 256},
}

config_name = sys.argv[1] if len(sys.argv) > 1 else "small"
if config_name not in ALL_CONFIGS:
    print(f"Unknown config '{config_name}'. Available: {list(ALL_CONFIGS.keys())}")
    sys.exit(1)

configs = [ALL_CONFIGS[config_name]]
ep_spec = P(("data", "tensor"))

for cfg in configs:
    d = cfg["d"]
    f = cfg["f"]
    E = cfg["E"]
    top_k = cfg["top_k"]
    bt = cfg["bt"]
    bf = cfg["bf"]

    num_tokens = bt * ep_size
    local_num_experts = E // ep_size

    if process_id == 0:
        print(
            f"\n--- {cfg['name']} (ep={ep_size}) ---\n"
            f"  E={E}, d={d}, f={f}, top_k={top_k}, bt={bt}, bf={bf}\n"
            f"  local_experts={local_num_experts}, num_tokens={num_tokens}"
        )

    key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    tokens_global = jax.random.normal(k1, (num_tokens, d), dtype=jnp.bfloat16)
    w1_global = jax.random.normal(k2, (E, d, f), dtype=jnp.bfloat16) * 0.01
    w2_global = jax.random.normal(k3, (E, f, d), dtype=jnp.bfloat16) * 0.01
    w3_global = jax.random.normal(k4, (E, d, f), dtype=jnp.bfloat16) * 0.01

    gating = jax.random.normal(k5, (num_tokens, E), dtype=jnp.float32)
    _, topk_idx = lax.top_k(gating, top_k)
    topk_logits = jnp.take_along_axis(gating, topk_idx, axis=-1)
    topk_wts = jax.nn.softmax(topk_logits, axis=-1)

    ep_sharding = jax.sharding.NamedSharding(mesh, ep_spec)

    tokens_s = jax.device_put(tokens_global, ep_sharding)
    w1_s = jax.device_put(w1_global, ep_sharding)
    w2_s = jax.device_put(w2_global, ep_sharding)
    w3_s = jax.device_put(w3_global, ep_sharding)
    topk_wts_s = jax.device_put(topk_wts, ep_sharding)
    topk_idx_s = jax.device_put(topk_idx, ep_sharding)

    result = fused_ep_moe_v2(
        mesh, tokens_s, w1_s, w2_s, w3_s,
        topk_wts_s, topk_idx_s, top_k,
        bt=bt, bf=bf,
    )

    ref = ref_moe_simple(
        tokens_global, w1_global, w2_global, w3_global,
        topk_wts, topk_idx, top_k,
    )

    result_gathered = jax.device_get(
        jax.device_put(result, jax.sharding.NamedSharding(mesh, P()))
    )

    if process_id == 0:
        result_f32 = result_gathered.astype(np.float32)
        ref_f32 = np.asarray(ref).astype(np.float32)
        max_err = np.max(np.abs(result_f32 - ref_f32))
        rel_err = float(max_err / (np.max(np.abs(ref_f32)) + 1e-6))
        print(f"  max_abs_err={max_err:.4f}, rel_err={rel_err:.6f}")
        if rel_err > 0.05:
            print("  FAIL")
            sys.exit(1)
        print("  PASS")

if process_id == 0:
    print("\nAll tests passed.")
